"""Step 5.1: choose the reading-order rule on the TRAIN + VAL pages (boxes only, no recognizer, no test page).

For each page the gap-rule pieces lying mostly on verified ink (as in Step 4) are put in reading order; each piece
stands for the verified word owning most of its ink. Metric: the share of piece pairs (with different words) whose
order is the reverse of the true order. Compared: the Step 3/4 order (lines by mean word centre) and
pipeline.layout.order_lines over a small grid of (min_v, max_x).

Output: results/step5/order_choice.json
Usage: python scripts/step5_order.py
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import REPO_ROOT, load_config  # noqa: E402


def load_module(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


diag = load_module("step4_page_diagnosis", "scripts/step4_page_diagnosis.py")

from pipeline.layout import line_bbox, order_lines  # noqa: E402
from tesseract_ft.boxes import group_box, groups_from_cuts, line_structures  # noqa: E402
from tesseract_ft.pages import load_batch2_pages  # noqa: E402

GRID = {"min_v": [0.3, 0.4, 0.5, 0.6, 0.7], "max_x": [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]}


def page_lines(p, box_cfg) -> list[dict]:
    """Per line piece: its box, and the owners (verified word index) of its kept gap-rule pieces in reading order."""
    ink = p.binary < 128
    gt_ink = np.zeros_like(ink)
    for x0, y0, x1, y1 in p.gt_words:
        gt_ink[y0:y1, x0:x1] = True
    owner = diag.ink_owner(p)
    lines, _ = line_structures(p.binary, box_cfg["line_finder"])
    out = []
    for ln in lines:
        gaps = ln["gaps"]
        thr = max(box_cfg["gap_h"] * ln["h"], box_cfg["gap_med"] * float(np.median(gaps)) if gaps else 0.0)
        owners, centres = [], []
        for i0, i1 in groups_from_cuts(len(ln["spans"]), {k for k, g in enumerate(gaps) if g > thr}):
            x0, y0, x1, y1 = group_box(ln, i0, i1)
            m = ink[y0:y1, x0:x1]
            if not m.any() or gt_ink[y0:y1, x0:x1][m].mean() <= 0.5:
                continue
            o = owner[y0:y1, x0:x1][m]
            o = o[o >= 0]
            if len(o):
                owners.append(int(np.bincount(o).argmax()))
                centres.append((y0 + y1) / 2)
        out.append({"box": line_bbox(ln), "owners": owners, "mean_centre": np.mean(centres) if centres else 0.0})
    return out


def out_of_order(pages: list[list[dict]], order_fn) -> tuple[int, int]:
    bad = tot = 0
    for lines in pages:
        seq = [o for i in order_fn(lines) for o in lines[i]["owners"]]
        for a, b in itertools.combinations(seq, 2):
            if a != b:
                tot += 1
                bad += a > b
    return bad, tot


def main() -> None:
    box_cfg = load_config("step3")["box_finder"]
    pages = [page_lines(p, box_cfg) for p in load_batch2_pages(("TRAIN", "VAL")) if p.gt_words]
    res = {}
    # Step 3/4 order: lines by the mean centre of their (kept) word boxes
    b, t = out_of_order(pages, lambda ls: sorted(range(len(ls)), key=lambda i: ls[i]["mean_centre"]))
    res["step3_4_mean_centre"] = {"out_of_order": b, "pairs": t, "share": round(b / t, 5)}
    for v, x in itertools.product(GRID["min_v"], GRID["max_x"]):
        b, t = out_of_order(pages, lambda ls: order_lines([l["box"] for l in ls], v, x))
        res[f"rows min_v={v} max_x={x}"] = {"min_v": v, "max_x": x, "out_of_order": b, "pairs": t, "share": round(b / t, 5)}
    grid = [r for k, r in res.items() if k.startswith("rows")]
    # lowest share; ties -> the stricter rule (larger min_v, smaller max_x)
    best = min(grid, key=lambda r: (r["out_of_order"], -r["min_v"], r["max_x"]))
    out = {"pages": len(pages), "chosen": {"min_v": best["min_v"], "max_x": best["max_x"]}, "results": res}
    d = REPO_ROOT / "results" / "step5"
    d.mkdir(parents=True, exist_ok=True)
    (d / "order_choice.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for k, r in res.items():
        print(f"{k:32s} {r['out_of_order']:5d} / {r['pairs']}  {r['share']:.4f}")
    print("chosen:", out["chosen"])


if __name__ == "__main__":
    main()
