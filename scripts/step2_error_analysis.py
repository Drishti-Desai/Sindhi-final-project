"""Step 2 error analysis for a letter-CNN run (evaluation only, no training).

Reports, per split (val, test):
  accuracy, top-3 accuracy, body-shape accuracy (letter body right, dots possibly wrong),
  dot accuracy given the right body, and accuracy per body family (beh, jeem, dal, ...).
Writes results/step2/cnn_error_analysis_<run>.json

Usage: python scripts/step2_error_analysis.py --run aux
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cnn_transformer.letters import LetterDataset, read_index  # noqa: E402
from cnn_transformer.models import LetterCNN  # noqa: E402
from cnn_transformer.train_utils import get_device, load_step2_config, save_json  # noqa: E402
from common.paths import REPO_ROOT, load_paths, load_yaml  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    cfg = load_step2_config()
    paths = load_paths()
    lm = load_yaml(paths["configs"]["letter_map"])
    structure = {int(k) - 1: v for k, v in lm["structure"].items()}
    device = get_device(cfg["device"])
    state = torch.load(REPO_ROOT / "checkpoints" / "step2_cnn_runs" / args.run / "best.pt",
                       map_location="cpu", weights_only=False)["model"]
    aux = tuple(state[f"aux_heads.{i}.weight"].shape[0] for i in range(3) if f"aux_heads.{i}.weight" in state)
    model = LetterCNN(52, cfg["letters"]["widths"], cfg["letters"]["dropout"], aux).to(device)
    model.load_state_dict(state)
    model.eval()
    import json
    meta = json.loads((REPO_ROOT / "results" / "step2" / "cnn_runs" / f"{args.run}.json").read_text())
    gray = meta.get("config", {}).get("input", "binary") == "gray"
    root = paths["preprocessed"]["letters52_gray" if gray else "letters52"]
    rows = read_index(paths["preprocessed"]["letters52"] / "index.csv")
    out = {"run": args.run}
    for split in ("val", "test"):
        ds = LetterDataset(root, [r for r in rows if r["split"] == split], cfg["letters"]["image_size"], gray=gray)
        logits, labels = [], []
        with torch.no_grad():
            for x, y in DataLoader(ds, 256):
                logits.append(model(x.to(device)).cpu())
                labels.append(y)
        logits, labels = torch.cat(logits), torch.cat(labels).numpy()
        pred = logits.argmax(1).numpy()
        top3 = logits.topk(3, 1).indices.numpy()
        body_ok = np.array([structure[p][0] == structure[t][0] for p, t in zip(pred, labels)])
        fam = defaultdict(list)
        for p, t in zip(pred, labels):
            fam[structure[t][0]].append(p == t)
        out[split] = {
            "n": int(len(labels)),
            "accuracy": round(float((pred == labels).mean()), 4),
            "top3_accuracy": round(float(np.mean([t in k for t, k in zip(labels, top3)])), 4),
            "body_accuracy": round(float(body_ok.mean()), 4),
            "dots_accuracy_given_body": round(float((pred == labels)[body_ok].mean()), 4),
            "per_family": {f: {"n": len(v), "accuracy": round(float(np.mean(v)), 3)}
                           for f, v in sorted(fam.items(), key=lambda kv: np.mean(kv[1]))},
        }
        print(split, {k: v for k, v in out[split].items() if k != "per_family"})
    print("test per family:", {f: v["accuracy"] for f, v in out["test"]["per_family"].items()})
    save_json(out, REPO_ROOT / "results" / "step2" / f"cnn_error_analysis_{args.run}.json")


if __name__ == "__main__":
    main()
