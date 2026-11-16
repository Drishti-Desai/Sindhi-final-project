"""Step 2b: train and test the word recognizer (letter-CNN encoder + Transformer decoder).

Requires the step2_cnn gate, or the user's recorded acceptance of it (option A, 2026-10-18: the letter
CNN plateaued at 0.8225 and is used as the backbone; Step 2 is gated on word-level 1-CER instead).

Training mix (configs/step2_cnn.yaml -> recognizer.mix; finetune.mix with --finetune):
  printed   text2image renders of TRAIN words (training fonts)
  composed  TRAIN words assembled from TRAIN-split handwritten letter crops
  real      verified real word crops: TESS_TRAIN pages, plus batch 2 TRAIN crops with --finetune
Validation (checkpoint selection): printed VAL + composed VAL words, plus batch 2 VAL real crops (--finetune).
Test (reported, never used for selection):
  composed_test       held-out words built from TEST-split letter crops (sheets never seen)  <- gate
  printed_test        held-out words in held-out fonts (Tahoma, Tahoma Bold)
  real_test           56 verified real word crops from TESS_TEST pages H1, J2, J3 (writers H and J: seen in
                      training once batch 2 is used)
  real_test_unseen    batch 2 purple-pen writer (144 crops), never used for training or validation
Every set reports CER, CER without punctuation (fair when a model cannot write a mark such as ؟),
word accuracy, and per-writer results for the real sets.

--finetune: continue from checkpoints/step2_recognizer/best.pt with the finetune: settings, into
checkpoints/step2_recognizer_ft/. The vocabulary is extended by any characters the new labels need
(existing weights copied unchanged; new output rows start neutral).

Usage:
  python scripts/step2_train_recognizer.py                      # original training (resumes) + test + gate
  python scripts/step2_train_recognizer.py --finetune           # fine-tune with batch 2 (resumes) + test + gate
  python scripts/step2_train_recognizer.py --eval [--finetune]  # evaluate best.pt only (no gate unless --gate)
  python scripts/step2_train_recognizer.py --small              # tiny run, no gate
"""
from __future__ import annotations

import os
from pathlib import Path as _Path

import yaml

# The Apple-GPU memory cap must be set before torch is imported. On this 8 GB Mac an unbounded
# run silently swapped for over an hour; with a cap PyTorch raises an out-of-memory error instead.
_rc = yaml.safe_load(open(_Path(__file__).resolve().parents[1] / "configs" / "step2_cnn.yaml"))["recognizer"]
_ratio = float(_rc.get("mps_memory_ratio", 0.6))
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", str(_ratio))     # hard cap
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", str(_ratio * 0.8))  # must be <= the cap

import argparse
import csv
import math
import sys
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cnn_transformer.models import WordRecognizer, load_trunk_from_letter_cnn  # noqa: E402
from cnn_transformer.train_utils import (Checkpointer, get_device, get_logger,  # noqa: E402
                                         load_step2_config, save_json, set_seed)
from cnn_transformer.words import (EOS, BOS, PAD, LetterBank, MixedWords, Vocab, collate,  # noqa: E402
                                   fixed_set, label_norm, read_printed, read_real)
from common.gates import read_gate, require_gate, write_gate  # noqa: E402
from common.metrics import cer, exact_match  # noqa: E402
from common.paths import REPO_ROOT, load_paths, load_yaml  # noqa: E402
from common.text import strip_punct  # noqa: E402


def read_words(path: Path) -> list[str]:
    return [label_norm(w) for w in path.read_text(encoding="utf-8").split()]


def read_batch2(split: str) -> tuple[list[tuple[Path, str]], list[str]]:
    """Kept batch 2 crops of one split -> ([(path, label)], [writer])."""
    root = REPO_ROOT / "data" / "preprocessed" / "handwritten_batch2_words"
    rows = [r for r in csv.DictReader(open(REPO_ROOT / "data" / "splits" / "handwritten_batch2" / "words.csv",
                                           encoding="utf-8")) if r["split"] == split and r["verified"] == "1"]
    return [(root / r["file"], label_norm(r["label"])) for r in rows], [r["writer"] for r in rows]


def old_real_writers(split: str) -> list[str]:
    """Writer (page prefix letter) for each verified crop of the original real-word set, in read_real order."""
    rows = csv.DictReader(open(REPO_ROOT / "data" / "splits" / "step2_real_words" / "verified.csv", encoding="utf-8"))
    return [r["page"][0] for r in rows if r["split"] == split and r["verified"] == "1"]


def predict(model, items, vocab, device, bs=64, max_len=26):
    model.eval()
    preds, confs = [], []
    for i in range(0, len(items), bs):
        x, widths, _, _ = collate(items[i:i + bs], vocab)
        seq, probs = model.greedy(x.to(device), widths.to(device), BOS, EOS, max_len)
        for s, p in zip(seq.cpu(), probs.cpu()):
            text = vocab.decode(s)
            preds.append(text)
            n = min(len(text) + 1, len(p))
            confs.append(float(p[:n].log().mean().exp()) if n else 0.0)
    return preds, confs


def score(items, preds, writers: list[str] | None = None) -> dict:
    refs = [l for _, l in items]
    np_refs, np_preds = [strip_punct(r) for r in refs], [strip_punct(p) for p in preds]
    out = {"n": len(refs), "cer": round(cer(refs, preds), 4), "one_minus_cer": round(1 - cer(refs, preds), 4),
           "cer_no_punct": round(cer(np_refs, np_preds), 4), "word_accuracy": round(exact_match(refs, preds), 4)}
    if writers:
        by = defaultdict(list)
        for r, p, w in zip(refs, preds, writers):
            by[w].append((r, p))
        out["per_writer"] = {w: {"n": len(v), "cer": round(cer([a for a, _ in v], [b for _, b in v]), 4),
                                 "word_accuracy": round(exact_match([a for a, _ in v], [b for _, b in v]), 4)}
                             for w, v in sorted(by.items())}
    return out


def extend_vocab(model: WordRecognizer, old: Vocab, new_chars: list[str], d_model: int) -> Vocab:
    """Append characters to the vocabulary: existing embedding/output rows are copied unchanged; new output
    rows get zero weights and a bias equal to the lowest existing bias (they start as unlikely, not random)."""
    if not new_chars:
        return old
    vocab = Vocab("")
    vocab.itos = list(old.itos) + new_chars
    vocab.stoi = {c: i for i, c in enumerate(vocab.itos)}
    n_old, n_new = len(old.itos), len(vocab.itos)
    emb = nn.Embedding(n_new, d_model).to(model.embed.weight.device)
    out = nn.Linear(d_model, n_new).to(model.out.weight.device)
    with torch.no_grad():
        emb.weight[:n_old] = model.embed.weight
        emb.weight[n_old:] = model.embed.weight.mean(0, keepdim=True)
        out.weight[:n_old], out.bias[:n_old] = model.out.weight, model.out.bias
        out.weight[n_old:] = 0.0
        out.bias[n_old:] = model.out.bias.min()
    model.embed, model.out = emb, out
    return vocab


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--finetune", action="store_true", help="fine-tune from the trained recognizer with batch 2")
    ap.add_argument("--gate", action="store_true", help="with --eval: also write the gate")
    ap.add_argument("--workers", type=int, help="data-loading processes (default: recognizer.num_workers)")
    ap.add_argument("--cnn-checkpoint", help="letter CNN to initialise from (default: the gated checkpoint)")
    args = ap.parse_args()

    if not args.small:
        require_gate("step2_cnn", allow_user_accepted=True)
    cfg = load_step2_config(args.small)
    rc = dict(cfg["recognizer"])
    if args.finetune:
        rc.update({k: v for k, v in cfg["finetune"].items() if k != "init"})
        if args.small:  # quick check: small sizes win over the fine-tune schedule
            rc.update(cfg["small"]["recognizer"])
    paths = load_paths()
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    tag = "_small" if args.small else ""
    run = ("recognizer_ft" if args.finetune else "recognizer") + tag
    log = get_logger(run, REPO_ROOT / "logs" / "step2")
    res_dir = REPO_ROOT / "results" / "step2"
    ck = Checkpointer(REPO_ROOT / "checkpoints" / f"{'step2_recognizer_ft' if args.finetune else 'step2_recognizer'}{tag}")

    # ---------------- data
    words_dir = REPO_ROOT / "data" / "splits" / f"step2_words{tag}"
    words = {s: read_words(words_dir / f"{s}.txt") for s in ("train", "val", "test")}
    syn = paths["synthetic"] / f"step2{tag}" / "printed"
    printed = {s: read_printed(syn / s) for s in ("train", "val", "test")}
    verified = REPO_ROOT / "data" / "splits" / "step2_real_words" / "verified.csv"
    crops_root = REPO_ROOT / "data" / "preprocessed" / "handwritten_words"
    real_train, real_test = read_real(verified, crops_root, "TESS_TRAIN"), read_real(verified, crops_root, "TESS_TEST")
    b2_train, _ = read_batch2("TRAIN")
    b2_val, b2_val_writers = read_batch2("VAL")
    b2_test, b2_test_writers = read_batch2("TEST_UNSEEN_WRITER")
    if args.finetune:
        real_train = real_train + b2_train

    lroot = paths["preprocessed"]["letters52_gray"]  # grayscale crops, like the letter CNN's input
    idx = list(csv.DictReader(open(paths["preprocessed"]["letters52"] / "index.csv", encoding="utf-8")))
    man = list(csv.DictReader(open(paths["splits"]["letters52"] / "manifest.csv", encoding="utf-8")))
    for m in man:
        m["h"] = cv2.imread(str(paths["raw"]["letters52"] / m["relpath"]), cv2.IMREAD_GRAYSCALE).shape[0]
    lmap = {int(k): v for k, v in load_yaml(paths["configs"]["letter_map"])["classes"].items()}
    banks = {s: LetterBank(lroot, idx, lmap, man, s, gray=True) for s in ("train", "val", "test")}
    composable = {s: [w for w in words[s] if banks[s].tokenize(w)] for s in words}

    # ---------------- model + vocabulary
    model_args = (cfg["letters"]["widths"], rc["d_model"], rc["enc_layers"], rc["dec_layers"], rc["heads"],
                  rc["ff"], rc["dropout"], rc["max_label_len"])
    init_state = None
    if args.finetune:
        init_state = torch.load(REPO_ROOT / cfg["finetune"]["init"], map_location="cpu", weights_only=False)
        vocab = Vocab("")
        vocab.itos = init_state["extra"]["vocab"]
        vocab.stoi = {c: i for i, c in enumerate(vocab.itos)}
    else:
        vocab = Vocab.build(words["train"] + [l for _, l in printed["train"]] + [l for _, l in real_train])
    model = WordRecognizer(len(vocab), *model_args).to(device)
    if init_state is not None:
        model.load_state_dict(init_state["model"])
        needed = sorted({ch for _, l in real_train + b2_val + b2_test for ch in l} - set(vocab.itos))
        vocab = extend_vocab(model, vocab, needed, rc["d_model"])
        log.info(f"fine-tuning from {cfg['finetune']['init']} (epoch {init_state['epoch'] + 1}); "
                 f"vocabulary extended by {needed} -> {len(vocab)}")
    model.checkpointing = rc.get("gradient_checkpointing", True)

    log.info(f"batch={rc['batch_size']} (= {rc.get('grad_accum', 1)} x {rc['batch_size'] // max(1, int(rc.get('grad_accum', 1)))}) "
             f"steps/epoch={rc['steps_per_epoch']} epochs={rc['epochs']} lr={rc['lr']} mix={rc['mix']} "
             f"workers={0 if args.small else (args.workers if args.workers is not None else rc.get('num_workers', 2))}")
    log.info(f"device={device} vocab={len(vocab)} printed train/val/test={[len(printed[s]) for s in printed]} "
             f"real train={len(real_train)} (batch2 {len(b2_train) if args.finetune else 0}) real_test={len(real_test)} "
             f"batch2 val/test_unseen={len(b2_val)}/{len(b2_test)}")
    log.info(f"composable words train/val/test = {[len(composable[s]) for s in composable]}")

    val_sets = {"printed_val": fixed_set("printed", printed["val"], None, rc, rc["val_size"], 1),
                "composed_val": fixed_set("composed", composable["val"], banks["val"], rc, rc["val_size"], 2)}
    if args.finetune:  # select on real handwriting too
        val_sets["real_val"] = fixed_set("real", b2_val, None, rc, 0, 6)
    stream = MixedWords(printed["train"], real_train, composable["train"], banks["train"], rc["mix"], rc, cfg["seed"])
    workers = args.workers if args.workers is not None else rc.get("num_workers", 2)
    accum = max(1, int(rc.get("grad_accum", 1)))
    loader = DataLoader(stream, batch_size=rc["batch_size"] // accum, num_workers=0 if args.small else workers,
                        collate_fn=partial(collate, vocab=vocab), persistent_workers=not args.small)

    trunk_ids = {id(p) for p in model.trunk.parameters()}
    opt = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) in trunk_ids], "lr": rc["lr"] * rc["backbone_lr_mult"]},
        {"params": [p for p in model.parameters() if id(p) not in trunk_ids], "lr": rc["lr"]}], weight_decay=0.01)
    total = rc["epochs"] * rc["steps_per_epoch"]
    warm = max(1, min(rc["warmup_steps"], total // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=rc["label_smoothing"])

    start, best, since = 0, -1.0, 0
    if not args.eval:
        st = None if args.fresh else ck.load("last", model, opt, sched, restore_rng=True)
        if st:
            start, best, since = st["epoch"] + 1, st["best"], st["extra"].get("since_best", 0)
            log.info(f"resumed from epoch {st['epoch'] + 1} (best val {best:.4f})")
        elif not args.finetune:
            cnn_ckpt = Path(args.cnn_checkpoint) if args.cnn_checkpoint else REPO_ROOT / "checkpoints" / "step2_cnn" / "best.pt"
            letter_state = torch.load(cnn_ckpt, map_location="cpu", weights_only=False)["model"]
            log.info(f"initialised encoder from letter CNN: {load_trunk_from_letter_cnn(model, letter_state)} tensors")
        if args.finetune and start == 0:
            # The starting model is the score to beat: save it as best.pt (epoch -1 = "no fine-tuning"), so if
            # no epoch improves on it, the final evaluation honestly uses the original model.
            vs = {k: score(v, predict(model, v, vocab, device)[0]) for k, v in val_sets.items()}
            best = float(np.mean([v["one_minus_cer"] for v in vs.values()]))
            ck.save("best", model, None, None, -1, best, {"vocab": vocab.itos})
            log.info("before fine-tuning: " + " ".join(f"{k} 1-CER {v['one_minus_cer']:.3f}" for k, v in vs.items())
                     + f" (mean {best:.4f} = score to beat)")

        it = iter(loader)
        for epoch in range(start, rc["epochs"]):
            model.train()
            t0, loss_sum = time.time(), 0.0
            for step in range(1, rc["steps_per_epoch"] + 1):
                opt.zero_grad(set_to_none=True)
                step_loss = 0.0
                for _ in range(accum):  # micro-batches: gradients add up to one effective batch
                    x, widths, y, _ = next(it)
                    x, widths, y = x.to(device), widths.to(device), y.to(device)
                    logits = model(x, widths, y[:, :-1])
                    loss = loss_fn(logits.reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1)) / accum
                    loss.backward()
                    step_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                loss_sum += step_loss
                if step % rc.get("log_every", 100) == 0 and device.type == "mps":
                    torch.mps.empty_cache()  # hand cached blocks back so memory cannot creep up
                if step % rc.get("log_every", 100) == 0:
                    el = time.time() - t0
                    log.info(f"  epoch {epoch + 1} step {step}/{rc['steps_per_epoch']} loss {loss_sum / step:.3f} "
                             f"({el / step:.2f}s/step, epoch ETA {el / step * (rc['steps_per_epoch'] - step) / 60:.0f} min)")
            vs = {k: score(v, predict(model, v, vocab, device)[0]) for k, v in val_sets.items()}
            val = np.mean([v["one_minus_cer"] for v in vs.values()])
            improved = val > best
            if improved:
                best, since = val, 0
                ck.save("best", model, None, None, epoch, best, {"vocab": vocab.itos})
            else:
                since += 1
            ck.save("last", model, opt, sched, epoch, best, {"since_best": since, "vocab": vocab.itos})
            log.info(f"epoch {epoch + 1:2d}/{rc['epochs']} loss {loss_sum / rc['steps_per_epoch']:.3f} "
                     + " ".join(f"{k} 1-CER {v['one_minus_cer']:.3f} acc {v['word_accuracy']:.3f}" for k, v in vs.items())
                     + f"{' *' if improved else ''} ({time.time() - t0:.0f}s)")
            if since >= rc["patience"]:
                log.info("early stop")
                break

    # ---------------- test
    st = torch.load(ck.dir / "best.pt", map_location="cpu", weights_only=False) if (ck.dir / "best.pt").exists() else None
    if st is None:
        sys.exit("No best.pt; train first.")
    vocab = Vocab("")
    vocab.itos = st["extra"]["vocab"]
    vocab.stoi = {c: i for i, c in enumerate(vocab.itos)}
    model = WordRecognizer(len(vocab), *model_args).to(device)
    model.load_state_dict(st["model"])
    n_test = 200 if args.small else 2000
    tests = {"composed_test": (fixed_set("composed", composable["test"], banks["test"], rc, n_test, 3), None),
             "printed_test": (fixed_set("printed", printed["test"], None, rc, n_test, 4), None),
             "real_test": (fixed_set("real", real_test, None, rc, 0, 5), old_real_writers("TESS_TEST")),
             "real_test_unseen": (fixed_set("real", b2_test, None, rc, 0, 7), b2_test_writers)}
    metrics = {"model": str(ck.dir.relative_to(REPO_ROOT) / "best.pt"), "best_epoch": st["epoch"] + 1,
               "best_val_mean_1_minus_cer": round(st["best"], 4), "device": str(device), "vocab_size": len(vocab),
               "labels": {"real_test": "writers H, J: seen in training" if args.finetune else "writer H seen, J unseen",
                          "real_test_unseen": "unseen writer (purple pen)"},
               "sets": {}}
    for name, (items, writers) in tests.items():
        preds, confs = predict(model, items, vocab, device)
        metrics["sets"][name] = score(items, preds, writers)
        out_name = f"{run}{'_eval' if args.eval else ''}_predictions_{name}.csv"
        with open(res_dir / out_name, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["reference", "prediction", "confidence", "correct", "writer"])
            for k, ((_, ref), p, c) in enumerate(zip(items, preds, confs)):
                w.writerow([ref, p, round(c, 3), int(ref == p), writers[k] if writers else ""])
        log.info(f"TEST {name}: " + ", ".join(f"{k}={v}" for k, v in metrics["sets"][name].items() if k != "per_writer")
                 + (f" per writer {metrics['sets'][name]['per_writer']}" if writers else ""))
    save_json(metrics, res_dir / f"{run}{'_eval' if args.eval else ''}_metrics.json")
    if args.small or (args.eval and not args.gate):
        log.info("gate not written (small mode, or --eval without --gate)")
        return
    cnn_gate = read_gate("step2_cnn")
    previous = read_gate("step2_cnn_transformer")
    g = write_gate("step2_cnn_transformer", {
        "composed_test_1_minus_cer": {"value": metrics["sets"]["composed_test"]["one_minus_cer"],
                                      "threshold": cfg["gate"]["min_sequence_1_minus_cer"]},
    }, extra={"model": metrics["model"], "sets": metrics["sets"], "labels": metrics["labels"],
              "letter_cnn": {"test_accuracy": cnn_gate["checks"]["letter_test_accuracy"]["value"],
                             "threshold": cfg["gate"]["min_letter_accuracy"], "status": cnn_gate["status"],
                             "user_decision": cnn_gate.get("user_decision")},
              "previous_gate": {k: previous[k] for k in ("status", "checks", "timestamp", "git_commit")} if previous else None,
              "note": "Gated on word level per the user's option A. real_test / real_test_unseen are reported, "
                      "not gated."})
    log.info(f"gate step2_cnn_transformer: {g['status']}")


if __name__ == "__main__":
    main()
