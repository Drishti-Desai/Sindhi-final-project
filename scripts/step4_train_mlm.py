"""Step 4: further masked-LM training of XLM-RoBERTa on Sindhi text (transformers Trainer).

Data: the Step 4 TRAIN split only (data/splits/step4/*_train.txt): Wikipedia once, legal + POS sentences repeated
`domain_oversample` times. Sentences are packed into chunks of up to `max_len` tokens without splitting words;
whole words are masked (15%: 80% <mask>, 10% random token, 10% unchanged). The embedding matrix (tied to the output
layer) and the lower `freeze_layers` transformer layers are frozen to fit the 8 GB M2; the upper layers and the
output head's dense layer train. The output
layer (250k-word vocabulary) is applied only at the masked positions: full logits would need ~2 GB per batch.
During training the frozen embedding matrix is held in bfloat16 (halves its 736 MB and the copy the backward pass
makes of it); the exact float32 weights are put back before the model is saved.
Validation loss on the val split (legal + POS + Wikipedia) picks the best checkpoint.

Outputs: checkpoints/step4_xlmr_mlm/{checkpoint-*, best/}, results/step4/mlm_log.json
Usage: python scripts/step4_train_mlm.py [--max-steps N] [--resume]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import REPO_ROOT, load_config  # noqa: E402

_cfg = load_config("step4")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", str(_cfg["mps_memory_ratio"]))
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", str(_cfg["mps_memory_ratio"] * 0.8))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments  # noqa: E402

from common.gates import require_gate  # noqa: E402

SEED = 42


def read(name: str) -> list[str]:
    return [l for l in (REPO_ROOT / "data" / "splits" / "step4" / f"{name}.txt").read_text(encoding="utf-8").splitlines()
            if l.strip()]


def pack(tok, sentences: list[str], max_len: int) -> list[dict]:
    """Chunks of whole sentences (<= max_len tokens incl. <s> </s>); word id per token for whole-word masking."""
    enc = tok(sentences, add_special_tokens=False)
    chunks, ids, wids, w0 = [], [], [], 0
    for k in range(len(sentences)):
        t = enc["input_ids"][k][: max_len - 2]
        wi = [w0 + (w if w is not None else 0) for w in enc.word_ids(k)][: max_len - 2]
        if ids and len(ids) + len(t) > max_len - 2:
            chunks.append({"input_ids": [tok.cls_token_id] + ids + [tok.sep_token_id], "word_ids": [-1] + wids + [-1]})
            ids, wids = [], []
        ids += t
        wids += wi
        w0 = (max(wi) + 1) if wi else w0
    if ids:
        chunks.append({"input_ids": [tok.cls_token_id] + ids + [tok.sep_token_id], "word_ids": [-1] + wids + [-1]})
    return chunks


class WholeWordCollator:
    def __init__(self, tok, p: float, seed: int):
        self.tok, self.p = tok, p
        self.rng = np.random.default_rng(seed)

    def __call__(self, batch: list[dict]) -> dict:
        L = max(len(b["input_ids"]) for b in batch)
        x = torch.full((len(batch), L), self.tok.pad_token_id)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        y = torch.full((len(batch), L), -100)
        for i, b in enumerate(batch):
            ids = torch.tensor(b["input_ids"])
            n = len(ids)
            x[i, :n], att[i, :n] = ids, 1
            words = sorted({w for w in b["word_ids"] if w >= 0})
            chosen = {w for w in words if self.rng.random() < self.p} or ({words[self.rng.integers(len(words))]} if words else set())
            for j, w in enumerate(b["word_ids"]):
                if w in chosen:
                    y[i, j] = ids[j]
                    r = self.rng.random()
                    if r < 0.8:
                        x[i, j] = self.tok.mask_token_id
                    elif r < 0.9:
                        x[i, j] = int(self.rng.integers(5, len(self.tok)))
        return {"input_ids": x, "attention_mask": att, "labels": y}


def masked_logits(model, h: torch.Tensor) -> torch.Tensor:
    """XLM-R's output head on selected hidden states; works with a bfloat16 (frozen) embedding matrix."""
    head = model.lm_head
    t = head.layer_norm(F.gelu(head.dense(h)))
    w = model.roberta.embeddings.word_embeddings.weight
    bias = head.decoder.bias if getattr(head.decoder, "bias", None) is not None else head.bias
    return F.linear(t.to(w.dtype), w).float() + bias


class MaskedOnlyTrainer(Trainer):
    """Masked-LM loss computed from the masked positions only (same loss, a fraction of the memory)."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        h = model.roberta(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state
        sel = labels != -100
        loss = F.cross_entropy(masked_logits(model, h[sel]), labels[sel], reduction="sum")
        loss = loss / (num_items_in_batch if num_items_in_batch is not None else sel.sum().clamp_min(1))
        return (loss, {}) if return_outputs else loss


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    require_gate("step3_tesseract", allow_user_accepted=True)
    c = _cfg["mlm"]
    out = REPO_ROOT / c["out"]
    tok = AutoTokenizer.from_pretrained(REPO_ROOT / _cfg["model"])
    rng = np.random.default_rng(SEED)
    domain = read("legal_train") + read("pos_train")
    train_s = read("wiki_train") + domain * c["domain_oversample"]
    train_s = [train_s[i] for i in rng.permutation(len(train_s))]
    val_s = read("legal_val") + read("pos_val") + read("wiki_val")
    train, val = pack(tok, train_s, c["max_len"]), pack(tok, val_s, c["max_len"])
    print(f"train chunks {len(train)} (sentences {len(train_s)}), val chunks {len(val)}")
    model = AutoModelForMaskedLM.from_pretrained(REPO_ROOT / _cfg["model"])
    if c["freeze_embeddings"]:
        for p in model.roberta.embeddings.parameters():
            p.requires_grad = False
        model.roberta.embeddings.LayerNorm.weight.requires_grad = True
        model.roberta.embeddings.LayerNorm.bias.requires_grad = True
    emb = model.roberta.embeddings.word_embeddings
    fp32_embeddings = emb.weight.detach().clone()
    if c["freeze_embeddings"]:
        emb.weight.data = emb.weight.data.to(torch.bfloat16)
    for layer in model.roberta.encoder.layer[: c.get("freeze_layers", 0)]:
        for p in layer.parameters():
            p.requires_grad = False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters {n_train / 1e6:.1f}M of {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    targs = TrainingArguments(
        output_dir=str(out), per_device_train_batch_size=c["batch_size"], per_device_eval_batch_size=4,
        gradient_accumulation_steps=c["grad_accum"], learning_rate=c["lr"], warmup_steps=c["warmup_steps"],
        max_steps=args.max_steps or c["max_steps"], eval_strategy="steps", eval_steps=c["eval_steps"],
        save_strategy="steps", save_steps=c["save_steps"], save_total_limit=3, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False, logging_steps=50, report_to=[],
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        prediction_loss_only=True, dataloader_num_workers=0, seed=SEED, remove_unused_columns=False)
    trainer = MaskedOnlyTrainer(model=model, args=targs, train_dataset=ListDataset(train), eval_dataset=ListDataset(val),
                      data_collator=WholeWordCollator(tok, c["whole_word_mask"], SEED))
    base_eval = trainer.evaluate()
    print(f"val loss before training: {base_eval['eval_loss']:.4f}")
    trainer.train(resume_from_checkpoint=args.resume or None)
    final_eval = trainer.evaluate()
    emb.weight.data = fp32_embeddings.to(emb.weight.device)   # exact original (frozen) weights
    trainer.save_model(str(out / "best"))
    tok.save_pretrained(str(out / "best"))
    log = {"val_loss_before": base_eval["eval_loss"], "val_loss_best": final_eval["eval_loss"],
           "best_checkpoint": trainer.state.best_model_checkpoint, "history": trainer.state.log_history,
           "train_chunks": len(train), "val_chunks": len(val), "trainable_params_M": round(n_train / 1e6, 1)}
    (REPO_ROOT / "results" / "step4").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "results" / "step4" / "mlm_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"val loss {base_eval['eval_loss']:.4f} -> {final_eval['eval_loss']:.4f}; best {trainer.state.best_model_checkpoint}")


if __name__ == "__main__":
    main()
