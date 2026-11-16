"""Step 2 data: cut real handwritten WORD crops from TESS_TRAIN / TESS_TEST pages.

FINAL_TEST pages are never read here (they are reserved for Step 6).

How: the draft ground truth (text_gt) says how many lines and words each page has, so
segmentation is a constrained alignment rather than blind detection:
  lines  cluster the larger connected components by vertical centre into exactly n lines;
         small components (dots, diacritics) join the nearest line
  words  inside a line, merge components that overlap horizontally, then cut at the
         (n_words - 1) widest gaps; words are assigned right-to-left (Sindhi reading order)
Each word gets a confidence = smallest chosen gap / largest unchosen gap (>1 means the cut is
clearly separated). A review image per page is saved so the crops can be checked by eye.

Outputs: data/preprocessed/handwritten_words/{TESS_TRAIN,TESS_TEST}/*.png + labels.csv
         results/step2/real_words_review/<page>.png
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import REPO_ROOT, load_paths  # noqa: E402
from common.text import normalize  # noqa: E402
from cnn_transformer.segment import assign_lines, components, split_words  # noqa: E402


def main() -> None:
    paths = load_paths()
    pairs = list(csv.DictReader(open(paths["splits"]["handwritten"] / "pairs.csv", encoding="utf-8")))
    out_root = REPO_ROOT / "data" / "preprocessed" / "handwritten_words"
    review_dir = REPO_ROOT / "results" / "step2" / "real_words_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in pairs:
        if p["split"] == "FINAL_TEST":
            continue  # reserved for Step 6
        stem = p["image"][:-4]
        binary = cv2.imread(str(paths["preprocessed"]["handwritten"] / p["image"]), cv2.IMREAD_GRAYSCALE)
        gt = normalize((paths["raw"]["handwritten_text_gt"] / p["text_gt"]).read_text(encoding="utf-8"),
                       diacritics=False).splitlines()
        comps, _ = components(binary)
        H, W = binary.shape
        # drop specks (< 6 px) and anything touching the page edge (photo-edge remnants)
        comps = [c for c in comps if c["a"] >= 6 and c["x"] > 1 and c["y"] > 1
                 and c["x"] + c["w"] < W - 1 and c["y"] + c["h"] < H - 1]
        lines = assign_lines(comps, len(gt))
        review = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        (out_root / p["split"]).mkdir(parents=True, exist_ok=True)
        for li, (line, text) in enumerate(zip(lines, gt)):
            words = text.split()
            groups, conf = split_words(line, len(words), words)
            if len(groups) != len(words):
                print(f"  {stem} line {li + 1}: could not split into {len(words)} words, skipped")
                continue
            for wi, (g, label) in enumerate(zip(groups, words)):
                x0, y0 = min(c["x"] for c in g), min(c["y"] for c in g)
                x1, y1 = max(c["x"] + c["w"] for c in g), max(c["y"] + c["h"] for c in g)
                mask = np.full(binary.shape, 255, np.uint8)
                _, lab = components(binary)
                ids = [c["i"] for c in g]
                mask[np.isin(lab, ids)] = 0
                crop = mask[max(0, y0 - 4):y1 + 4, max(0, x0 - 4):x1 + 4]
                fn = f"{stem}_l{li + 1}_w{wi + 1}.png"
                cv2.imwrite(str(out_root / p["split"] / fn), crop)
                rows.append({"split": p["split"], "file": f"{p['split']}/{fn}", "label": label,
                             "page": p["image"], "line": li + 1, "word": wi + 1,
                             "line_confidence": round(conf, 2)})
                color = (0, 160, 0) if conf > 1.2 else (0, 0, 255)
                cv2.rectangle(review, (x0, y0), (x1, y1), color, 2)
                cv2.putText(review, str(wi + 1), (x0, max(12, y0 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imwrite(str(review_dir / f"{stem}.png"), review)
    with open(out_root / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for s in ("TESS_TRAIN", "TESS_TEST"):
        rs = [r for r in rows if r["split"] == s]
        print(f"{s}: {len(rs)} word crops, {sum(r['line_confidence'] > 1.2 for r in rs)} from clearly separated lines")


if __name__ == "__main__":
    main()
