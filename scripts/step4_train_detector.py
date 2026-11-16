"""Step 4: "is this word wrong?" detector (token classification on top of the Sindhi-trained XLM-R; Trainer).

Training data: Step 4 TRAIN sentences (legal + POS three times with different corruption seeds, plus Wikipedia
up to `train_sentences`), corrupted with the recognizer error model (xlmr/errors.py) at 20% of words. Label per
input word: 1 if it is not exactly a correct word (changed, merged or split), else 0; only a word's first piece
carries the label. Validation: the frozen corrupted domain val set; the lowest validation loss is kept.
Embeddings frozen and held in bfloat16, lower `freeze_layers` layers frozen (as for the masked-LM run, to fit the
8 GB M2); the upper layers and the 2-class head train.

Outputs: checkpoints/step4_xlmr_detector/best, results/step4/detector_log.json
Usage: python scripts/step4_train_detector.py [--max-steps N]
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
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments  # noqa: E402

from common.gates import require_gate  # noqa: E402
from tesseract_ft.pages import label_norm_words  # noqa: E402
from xlmr.correct import clean_tokens  # noqa: E402
from xlmr.errors import ErrorModel  # noqa: E402

SEED = 42


def read(name: str) -> list[list[str]]:
    lines = (REPO_ROOT / "data" / "splits" / "step4" / f"{name}.txt").read_text(encoding="utf-8").splitlines()
    return [w for w in (label_norm_words(l) for l in lines) if w]


def encode(tok, items: list[tuple[list[str], list[int]]], max_len: int = 128) -> list[dict]:
    out = []
    for words, labels in items:
        enc = tok(words, is_split_into_words=True, truncation=True, max_length=max_len)
        lab, seen = [], set()
        for w in enc.word_ids():
            if w is None or w in seen:
                lab.append(-100)
            else:
                lab.append(labels[w])
                seen.add(w)
        out.append({"input_ids": enc["input_ids"], "labels": lab})
    return out


def collate(tok):
    def f(batch):
        L = max(len(b["input_ids"]) for b in batch)
        x = torch.full((len(batch), L), tok.pad_token_id)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        y = torch.full((len(batch), L), -100)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            x[i, :n], att[i, :n], y[i, :n] = torch.tensor(b["input_ids"]), 1, torch.tensor(b["labels"])
        return {"input_ids": x, "attention_mask": att, "labels": y}
    return f


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()
    require_gate("step3_tesseract", allow_user_accepted=True)
    c = _cfg["detector"]
    init = REPO_ROOT / c["init"]
    tok = AutoTokenizer.from_pretrained(init)
    em = ErrorModel.load(REPO_ROOT / "results" / "step4" / "error_model.json")
    rng = np.random.default_rng(SEED)
    domain = read("legal_train") + read("pos_train")
    wiki = read("wiki_train")
    n_wiki = max(0, c["train_sentences"] - 3 * len(domain))
    wiki = [wiki[i] for i in rng.choice(len(wiki), n_wiki, replace=False)]
    train_items = []
    for k, sents in enumerate([domain, domain, domain, wiki]):
        toks, origin = em.corrupt(sents, _cfg["gate"]["corruption_rate"], seed=1000 + k)
        for ref, t, o in zip(sents, toks, origin):
            train_items.append((t, [0 if ok else 1 for ok in clean_tokens(ref, t, o)]))
    val_items = []
    for l in open(REPO_ROOT / "data" / "splits" / "step4" / "corrupted_domain_val.jsonl", encoding="utf-8"):
        it = json.loads(l)
        val_items.append((it["input"], [0 if ok else 1 for ok in clean_tokens(it["ref"], it["input"], it["origin"])]))
    train, val = encode(tok, train_items), encode(tok, val_items)
    print(f"train sentences {len(train)} (wrong-word share "
          f"{np.mean([y for _, ys in train_items for y in ys]):.3f}), val {len(val)}")
    model = AutoModelForTokenClassification.from_pretrained(init, num_labels=2)
    for p in model.roberta.embeddings.parameters():
        p.requires_grad = False
    emb = model.roberta.embeddings.word_embeddings
    fp32_embeddings = emb.weight.detach().clone()
    emb.weight.data = emb.weight.data.to(torch.bfloat16)   # frozen: halves its memory during training
    for layer in model.roberta.encoder.layer[: c.get("freeze_layers", 0)]:
        for p in layer.parameters():
            p.requires_grad = False
    out = REPO_ROOT / c["out"]
    targs = TrainingArguments(
        output_dir=str(out), per_device_train_batch_size=c["batch_size"], per_device_eval_batch_size=16,
        gradient_accumulation_steps=c["grad_accum"], learning_rate=c["lr"], warmup_steps=0.06,
        num_train_epochs=c["epochs"], max_steps=args.max_steps, eval_strategy="steps", eval_steps=200,
        save_strategy="steps", save_steps=200, save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False, logging_steps=50, report_to=[],
        dataloader_num_workers=0, seed=SEED, remove_unused_columns=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    trainer = Trainer(model=model, args=targs, train_dataset=ListDataset(train), eval_dataset=ListDataset(val),
                      data_collator=collate(tok))
    trainer.train()
    ev = trainer.evaluate()
    emb.weight.data = fp32_embeddings.to(emb.weight.device)   # exact original (frozen) weights
    trainer.save_model(str(out / "best"))
    tok.save_pretrained(str(out / "best"))
    (REPO_ROOT / "results" / "step4" / "detector_log.json").write_text(json.dumps(
        {"val_loss_best": ev["eval_loss"], "best_checkpoint": trainer.state.best_model_checkpoint,
         "train_sentences": len(train), "history": trainer.state.log_history}, indent=1), encoding="utf-8")
    print(f"val loss {ev['eval_loss']:.4f}; best {trainer.state.best_model_checkpoint}")


if __name__ == "__main__":
    main()
