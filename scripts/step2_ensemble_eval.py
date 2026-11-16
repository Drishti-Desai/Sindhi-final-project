"""Step 2: evaluate ensembles of letter-CNN runs, with and without test-time augmentation (TTA).

The combination is chosen on the VALIDATION split only; test accuracy is then reported for that
single choice (and, for transparency, for every candidate in the JSON).

TTA = average softmax over the original crop and a few fixed small rotations/scalings.
Writes results/step2/cnn_ensemble.json

Usage: python scripts/step2_ensemble_eval.py --runs baseline aux aux_synth
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cnn_transformer.letters import LetterDataset, read_index  # noqa: E402
from cnn_transformer.models import LetterCNN  # noqa: E402
from cnn_transformer.train_utils import get_device, load_step2_config, save_json  # noqa: E402
from common.paths import REPO_ROOT, load_paths  # noqa: E402

TTA = [(0, 1.0), (-6, 1.0), (6, 1.0), (0, 0.9), (0, 1.1)]  # (degrees, scale)


def load_run(run: str, cfg: dict, device):
    state = torch.load(REPO_ROOT / "checkpoints" / "step2_cnn_runs" / run / "best.pt",
                       map_location="cpu", weights_only=False)["model"]
    aux = tuple(state[f"aux_heads.{i}.weight"].shape[0] for i in range(3) if f"aux_heads.{i}.weight" in state)
    m = LetterCNN(52, cfg["letters"]["widths"], cfg["letters"]["dropout"], aux).to(device)
    m.load_state_dict(state)
    return m.eval()


def warp(x: torch.Tensor, deg: float, scale: float) -> torch.Tensor:
    if deg == 0 and scale == 1.0:
        return x
    out = []
    for img in x[:, 0].numpy():
        m = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), deg, scale)
        out.append(cv2.warpAffine(img, m, img.shape[::-1], flags=cv2.INTER_LINEAR, borderValue=0))
    return torch.from_numpy(np.stack(out)).unsqueeze(1)


def probs(model, ds, device, tta: bool) -> tuple[np.ndarray, np.ndarray]:
    ps, ys = [], []
    with torch.no_grad():
        for x, y in DataLoader(ds, 256):
            views = TTA if tta else TTA[:1]
            p = sum(model(warp(x, d, s).to(device)).softmax(1).cpu() for d, s in views) / len(views)
            ps.append(p)
            ys.append(y)
    return torch.cat(ps).numpy(), torch.cat(ys).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args()
    cfg = load_step2_config()
    paths = load_paths()
    device = get_device(cfg["device"])
    rows = read_index(paths["preprocessed"]["letters52"] / "index.csv")
    datasets = {}
    for kind in ("binary", "gray"):  # each run is evaluated on the input type it was trained on
        root = paths["preprocessed"]["letters52_gray" if kind == "gray" else "letters52"]
        datasets[kind] = {s: LetterDataset(root, [r for r in rows if r["split"] == s], cfg["letters"]["image_size"],
                                           gray=kind == "gray") for s in ("val", "test")}
    cache = {}
    for run in args.runs:
        meta = json.loads((REPO_ROOT / "results" / "step2" / "cnn_runs" / f"{run}.json").read_text())
        kind = meta.get("config", {}).get("input", "binary")
        model = load_run(run, cfg, device)
        for tta in (False, True):
            for s in ("val", "test"):
                cache[(run, tta, s)] = probs(model, datasets[kind][s], device, tta)
    results = []
    for k in range(1, len(args.runs) + 1):
        for combo in itertools.combinations(args.runs, k):
            for tta in (False, True):
                acc = {}
                for s in ("val", "test"):
                    p = sum(cache[(r, tta, s)][0] for r in combo)
                    acc[s] = float((p.argmax(1) == cache[(combo[0], tta, s)][1]).mean())
                results.append({"runs": list(combo), "tta": tta, "val_accuracy": round(acc["val"], 4),
                                "test_accuracy": round(acc["test"], 4)})
    results.sort(key=lambda r: (-r["val_accuracy"], len(r["runs"])))
    for r in results:
        print(f"{'+'.join(r['runs']):32s} tta={str(r['tta']):5s} val {r['val_accuracy']:.4f}  test {r['test_accuracy']:.4f}")
    best = results[0]
    print(f"\nchosen on VAL: {best}")
    save_json({"chosen_on_val": best, "all": results}, REPO_ROOT / "results" / "step2" / "cnn_ensemble.json")


if __name__ == "__main__":
    main()
