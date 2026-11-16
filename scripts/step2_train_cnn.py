"""Step 2a: train and test the 52-letter CNN.

Data: data/preprocessed/letters52/index.csv (Step 1 output; split by sheet, seed 42).
  train -> training, val -> checkpoint selection / early stopping, test -> reported once at the end.

Outputs:
  checkpoints/step2_cnn/{last,best}.pt     (resumable; rerun the script to continue)
  logs/step2/letter_cnn.log
  results/step2/cnn_metrics.json, cnn_per_class.csv, cnn_confusions.csv,
  results/step2/cnn_errors.png (misclassified test crops), cnn_augment_samples.png
  results/gates/step2_cnn.json             (accuracy >= 0.90 on the test split)

Usage:
  python scripts/step2_train_cnn.py            # train (resumes if a checkpoint exists) + test + gate
  python scripts/step2_train_cnn.py --fresh    # ignore existing checkpoints
  python scripts/step2_train_cnn.py --small    # 3 epochs, no gate (pipeline check)
  python scripts/step2_train_cnn.py --eval     # only evaluate best.pt
  python scripts/step2_train_cnn.py --run aux --aux-weight 0.3 --synthetic-p 0   # named experiment
  python scripts/step2_train_cnn.py --run aux --eval --gate   # promote a run (chosen by VAL accuracy)

Runs are compared on the validation split only; --gate copies the run to checkpoints/step2_cnn/best.pt
and writes the gate from its test accuracy.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cnn_transformer.letters import LetterDataset, augment, read_index, render_font_letters  # noqa: E402
from cnn_transformer.models import LetterCNN  # noqa: E402
from cnn_transformer.train_utils import (Checkpointer, get_device, get_logger,  # noqa: E402
                                         load_step2_config, save_json, set_seed)
from common.gates import require_gate, write_gate  # noqa: E402
from common.paths import REPO_ROOT, load_paths, load_yaml  # noqa: E402


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).argmax(1).cpu())
            labels.append(y[:, 0] if y.ndim == 2 else y)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


def macro_f1(pred, true, n):
    f1s = []
    for c in range(n):
        tp = np.sum((pred == c) & (true == c))
        p = tp / max(1, np.sum(pred == c))
        r = tp / max(1, np.sum(true == c))
        f1s.append(0 if p + r == 0 else 2 * p * r / (p + r))
    return float(np.mean(f1s))


def save_error_grid(ds, pred, true, letters, path: Path, n=48):
    wrong = np.where(pred != true)[0][:n]
    tiles = []
    for i in wrong:
        ink = ds.items[i][0]
        t = cv2.cvtColor(255 - ink, cv2.COLOR_GRAY2BGR)
        t = cv2.copyMakeBorder(cv2.resize(t, (96, 96)), 0, 22, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(t, f"{true[i] + 1}->{pred[i] + 1}", (4, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        tiles.append(cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(180, 180, 180)))
    if not tiles:
        return
    while len(tiles) % 8:
        tiles.append(np.full_like(tiles[0], 255))
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--run", default="main", help="experiment name (separate checkpoints/results)")
    ap.add_argument("--aux-weight", type=float, help="override letters.aux_weight")
    ap.add_argument("--synthetic-p", type=float, help="override letters.synthetic_p")
    ap.add_argument("--gate", action="store_true", help="promote this run and write the gate")
    ap.add_argument("--input", choices=["binary", "gray"], default="binary",
                    help="binary = Step-1 output; gray = Step-1 cleaned grayscale before thresholding")
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="learning curve: keep this fraction of training SHEETS (seed 42)")
    args = ap.parse_args()

    require_gate("step1_preprocessing")
    cfg = load_step2_config(args.small)
    lc = cfg["letters"]
    if args.aux_weight is not None:
        lc["aux_weight"] = args.aux_weight
    if args.synthetic_p is not None:
        lc["synthetic_p"] = args.synthetic_p
    paths = load_paths()
    lmap_file = load_yaml(paths["configs"]["letter_map"])
    letters = {int(k): v for k, v in lmap_file["classes"].items()}
    structure = {int(k): v for k, v in lmap_file["structure"].items()} if lc["aux_weight"] > 0 else None
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    run = args.run + ("_small" if args.small else "")
    ck_dir = REPO_ROOT / "checkpoints" / "step2_cnn_runs" / run
    res_dir = REPO_ROOT / "results" / "step2"
    res_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger(f"letter_cnn_{run}", REPO_ROOT / "logs" / "step2")
    log.info(f"run={run} aux_weight={lc['aux_weight']} synthetic_p={lc['synthetic_p']} input={args.input} "
             f"train_frac={args.train_frac}")

    root = paths["preprocessed"]["letters52_gray" if args.input == "gray" else "letters52"]
    rows = read_index(paths["preprocessed"]["letters52"] / "index.csv")
    gray = args.input == "gray"
    by = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    if args.train_frac < 1.0:  # learning-curve runs: subsample whole sheets, never single crops
        import random as _r
        sheets = sorted({r["sheet"] for r in by["train"]})
        keep = set(_r.Random(42).sample(sheets, max(1, round(len(sheets) * args.train_frac))))
        by["train"] = [r for r in by["train"] if r["sheet"] in keep]
    log.info(f"device={device} train={len(by['train'])} val={len(by['val'])} test={len(by['test'])}")
    synthetic = render_font_letters(letters, lc["synthetic_fonts"]) if lc["synthetic_p"] > 0 else None
    if synthetic:
        log.info(f"font-rendered letters: {len(synthetic)}")
    train_ds = LetterDataset(root, by["train"], lc["image_size"], lc["augment"], cfg["seed"],
                             structure, synthetic, lc["synthetic_p"], gray=gray)
    val_ds, test_ds = (LetterDataset(root, by[s], lc["image_size"], gray=gray) for s in ("val", "test"))

    # sanity image: what the network actually sees after augmentation
    rng = np.random.default_rng(0)
    samp = [255 - augment(train_ds.items[i][0], rng, lc["augment"]) for i in rng.choice(len(train_ds), 32)]
    cv2.imwrite(str(res_dir / "cnn_augment_samples.png"),
                np.vstack([np.hstack(samp[i:i + 8]) for i in range(0, 32, 8)]))

    train_dl = DataLoader(train_ds, lc["batch_size"], shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, 256)
    test_dl = DataLoader(test_ds, 256)

    aux_sizes = train_ds.aux_sizes if structure else ()
    model = LetterCNN(52, lc["widths"], lc["dropout"], aux_sizes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lc["lr"], weight_decay=lc["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lc["lr"], epochs=lc["epochs"],
                                                steps_per_epoch=len(train_dl), pct_start=0.1)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=lc["label_smoothing"])
    ck = Checkpointer(ck_dir)

    start, best, since_best = 0, -1.0, 0
    if not args.fresh and not args.eval:
        st = ck.load("last", model, opt, sched, restore_rng=True)
        if st:
            start, best = st["epoch"] + 1, st["best"]
            since_best = st["extra"].get("since_best", 0)
            log.info(f"resumed from epoch {st['epoch']} (best val acc {best:.4f})")

    if not args.eval:
        for epoch in range(start, lc["epochs"]):
            t0 = time.time()
            model.train()
            tot, correct, loss_sum = 0, 0, 0.0
            for x, y in train_dl:
                x, y = x.to(device), y.to(device)
                if structure:
                    out, aux_out = model.forward_all(x)
                    loss = loss_fn(out, y[:, 0]) + lc["aux_weight"] * sum(
                        nn.functional.cross_entropy(a, y[:, k + 1]) for k, a in enumerate(aux_out))
                    y = y[:, 0]
                else:
                    out = model(x)
                    loss = loss_fn(out, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                loss_sum += loss.item() * len(y)
                correct += (out.argmax(1) == y).sum().item()
                tot += len(y)
            vp, vt = evaluate(model, val_dl, device)
            val_acc = float((vp == vt).mean())
            improved = val_acc > best
            if improved:
                best, since_best = val_acc, 0
                ck.save("best", model, None, None, epoch, best)
            else:
                since_best += 1
            ck.save("last", model, opt, sched, epoch, best, {"since_best": since_best})
            log.info(f"epoch {epoch + 1:3d}/{lc['epochs']} loss {loss_sum / tot:.3f} train_acc {correct / tot:.3f} "
                     f"val_acc {val_acc:.3f}{' *' if improved else ''} ({time.time() - t0:.1f}s)")
            if since_best >= lc["patience"]:
                log.info(f"early stop: no val improvement for {lc['patience']} epochs")
                break

    st = ck.load("best", model)
    if st is None:
        sys.exit("No best.pt found; train first.")
    pred, true = evaluate(model, test_dl, device)
    acc = float((pred == true).mean())
    vp, vt = evaluate(model, val_dl, device)
    per_class = [{"class_id": c + 1, "letter": letters[c + 1], "n": int((true == c).sum()),
                  "accuracy": round(float((pred[true == c] == c).mean()), 3) if (true == c).any() else None}
                 for c in range(52)]
    conf = Counter((int(t) + 1, int(p) + 1) for t, p in zip(true, pred) if t != p)
    top = [{"true": t, "true_letter": letters[t], "pred": p, "pred_letter": letters[p], "count": n}
           for (t, p), n in conf.most_common(25)]
    metrics = {
        "test_accuracy": round(acc, 4), "test_macro_f1": round(macro_f1(pred, true, 52), 4),
        "val_accuracy": round(float((vp == vt).mean()), 4), "best_epoch": st["epoch"] + 1,
        "n_test": int(len(true)), "n_test_sheets": len({r["sheet"] for r in by["test"]}),
        "device": str(device), "small_mode": args.small, "top_confusions": top,
    }
    metrics["run"] = run
    metrics["config"] = {"aux_weight": lc["aux_weight"], "synthetic_p": lc["synthetic_p"], "input": args.input,
                         "train_frac": args.train_frac}
    save_json(metrics, res_dir / "cnn_runs" / f"{run}.json")
    if not args.gate:
        log.info(f"run {run} done (val {metrics['val_accuracy']:.4f}); promote with --run {args.run} --eval --gate")
        return
    import shutil
    canon = REPO_ROOT / "checkpoints" / "step2_cnn"
    canon.mkdir(parents=True, exist_ok=True)
    shutil.copy(ck_dir / "best.pt", canon / "best.pt")
    save_json({**metrics, "aux_sizes": list(aux_sizes)}, res_dir / "cnn_metrics.json")
    with open(res_dir / "cnn_per_class.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_class[0].keys()))
        w.writeheader()
        w.writerows(per_class)
    with open(res_dir / "cnn_confusions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(top[0].keys()) if top else ["true"])
        w.writeheader()
        w.writerows(top)
    save_error_grid(test_ds, pred, true, letters, res_dir / "cnn_errors.png")
    log.info(f"TEST accuracy {acc:.4f}, macro-F1 {metrics['test_macro_f1']:.4f} "
             f"(best epoch {metrics['best_epoch']}, val {metrics['val_accuracy']:.4f})")
    log.info("top confusions: " + ", ".join(f"{c['true_letter']}->{c['pred_letter']}({c['count']})" for c in top[:10]))

    if args.small:
        log.info("small mode: gate not written")
        return
    g = write_gate("step2_cnn", {"letter_test_accuracy": {"value": round(acc, 4),
                                                          "threshold": cfg["gate"]["min_letter_accuracy"]}},
                   extra={"macro_f1": metrics["test_macro_f1"], "n_test": metrics["n_test"],
                          "test_definition": "top-1 accuracy, 52 classes, test sheets never seen in training"})
    log.info(f"gate step2_cnn: {g['status']}")


if __name__ == "__main__":
    main()
