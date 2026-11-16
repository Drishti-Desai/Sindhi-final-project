"""Step 2 experiment input: Step-1 cleaned GRAYSCALE letter crops (before thresholding).

Same pipeline and parameters as Step 1 (glyph profile); only the last thresholding is skipped,
so faint dots stay as soft grey marks instead of vanishing or fusing.
Output: data/preprocessed/letters52_gray/<class_id>/<stem>.png (same relpaths as letters52/index.csv)
"""
from __future__ import annotations

import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import load_config, load_paths  # noqa: E402
from preprocess import preprocess_image  # noqa: E402

CFG = load_config("step1")


def work(job):
    src, dst = job
    r = preprocess_image(src, CFG, profile="glyph")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(dst, r.gray)


def main() -> None:
    paths = load_paths()
    rows = list(csv.DictReader(open(paths["preprocessed"]["letters52"] / "index.csv", encoding="utf-8")))
    out = paths["preprocessed"]["letters52_gray"]
    jobs = [(str(paths["raw"]["letters52"] / r["raw_relpath"]), str(out / r["relpath"])) for r in rows]
    with Pool(6) as p:
        p.map(work, jobs, chunksize=32)
    print(f"{len(jobs)} grayscale letter crops -> {out}")


if __name__ == "__main__":
    main()
