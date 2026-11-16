"""Build the data manifests and the fixed train/val/test splits (run once, before Step 1).

Outputs (paths from configs/paths.yaml):
  data/splits/letters52/manifest.csv   one row per UNIQUE letter image (byte-identical copies removed)
  data/splits/letters52/split.csv      80/20 split grouped by source sheet (+ val slice of train)
  data/splits/letters52/summary.json   per-class / per-split counts
  data/splits/handwritten/pairs.csv    image/text pairs, story group, and split (PLAN.md section 2.2)

Why group by sheet: crops from one photographed sheet share writer, pen, paper and lighting.
Splitting by image would put near-identical conditions in train and test and inflate accuracy.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import load_paths  # noqa: E402

SEED = 42
TEST_FRAC = 0.20
VAL_FRAC = 0.10  # of the training sheets
# Roboflow export names look like "<photo>_jpg.rf.<hash>_<n>.jpg"; <photo> identifies the sheet.
SHEET_RE = re.compile(r"^(?P<sheet>.+?)_(?:jpg|jpeg|png)\.rf\.")

# Story groups and splits approved in PLAN.md section 2.2. Images that share text stay together.
# D4 was a duplicate of D3 at Step 0; the user replaced D4.png on 2026-10-17 so it now matches
# D4.txt (a separate story), and it joins TESS_TRAIN, where its text was already going to be used.
HANDWRITTEN_GROUPS = {
    "G1_match_fixing": (["A1", "A2", "A3", "A4", "A5", "D1", "D1.", "H2"], "TESS_TRAIN"),
    "G2_supreme_court": (["D6.", "J1"], "FINAL_TEST"),
    "G3_dharamshala": (["J2", "J3"], "TESS_TEST"),
    "G4_sanghar": (["J4", "J5"], "FINAL_TEST"),
    "G5_u19_world_cup": (["D2"], "FINAL_TEST"),
    "G6_nz_odi": (["D3"], "TESS_TRAIN"),
    "G7_web_form_label": (["D5"], "TESS_TRAIN"),
    "G8_lakyari": (["H1"], "TESS_TEST"),
    "G9_malir": (["H3"], "FINAL_TEST"),
    "G10_democracy": (["H4"], "TESS_TRAIN"),
    "G11_ig_sindh": (["H5"], "FINAL_TEST"),
    "G12_sarfraz": (["D4"], "TESS_TRAIN"),
}


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def build_letter_manifest(raw_dir: Path) -> list[dict]:
    """One row per unique image. The representative copy is the shortest filename (no _1/(1))."""
    by_hash: dict[str, list[tuple[str, str, Path]]] = defaultdict(list)
    for img in sorted(raw_dir.glob("*/*/*.jpg")):
        old_split, cls = img.parts[-3], img.parts[-2]
        by_hash[md5(img)].append((old_split, cls, img))
    rows = []
    for h, copies in by_hash.items():
        labels = {c for _, c, _ in copies}
        if len(labels) != 1:  # never seen in this data, but refuse to guess a label
            raise ValueError(f"Byte-identical images with different labels {labels}: {copies}")
        old_split, cls, rep = min(copies, key=lambda c: (len(c[2].name), c[2].name))
        m = SHEET_RE.match(rep.name)
        sheet = m.group("sheet") if m else rep.stem
        rows.append({
            "hash": h,
            "class_id": int(cls),
            "relpath": rep.relative_to(raw_dir).as_posix(),
            "sheet": sheet,
            "batch": "whatsapp_2025_08" if sheet.startswith("WhatsApp") else "img_2025_04",
            "old_split": old_split,
            "n_copies": len(copies),
        })
    rows.sort(key=lambda r: (r["class_id"], r["relpath"]))
    return rows


def pick_sheets(sheets: list[str], sizes: dict[str, int], frac: float, rng: random.Random) -> set[str]:
    """Shuffle sheets and take them until ~frac of the images are covered."""
    order = sheets[:]
    rng.shuffle(order)
    target = frac * sum(sizes[s] for s in order)
    chosen, total = set(), 0
    for s in order:
        if total >= target:
            break
        chosen.add(s)
        total += sizes[s]
    return chosen


def split_letters(rows: list[dict]) -> None:
    """Assign train/val/test in place, grouped by sheet and stratified by photo batch."""
    rng = random.Random(SEED)
    sizes = Counter(r["sheet"] for r in rows)
    for batch in sorted({r["batch"] for r in rows}):
        sheets = sorted({r["sheet"] for r in rows if r["batch"] == batch})
        test = pick_sheets(sheets, sizes, TEST_FRAC, rng)
        remaining = [s for s in sheets if s not in test]
        val = pick_sheets(remaining, sizes, VAL_FRAC, rng)
        for r in rows:
            if r["batch"] == batch:
                r["split"] = "test" if r["sheet"] in test else "val" if r["sheet"] in val else "train"


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def letters(paths: dict) -> dict:
    raw = paths["raw"]["letters52"]
    out = paths["splits"]["letters52"]
    rows = build_letter_manifest(raw)
    split_letters(rows)
    write_csv(out / "manifest.csv", rows,
              ["hash", "class_id", "relpath", "sheet", "batch", "old_split", "n_copies"])
    write_csv(out / "split.csv", rows, ["relpath", "class_id", "sheet", "batch", "split"])

    per_class = Counter(r["class_id"] for r in rows)
    per_split_class = {s: Counter(r["class_id"] for r in rows if r["split"] == s)
                       for s in ("train", "val", "test")}
    files_on_disk = sum(1 for _ in raw.glob("*/*/*.jpg"))
    summary = {
        "seed": SEED,
        "files_on_disk": files_on_disk,
        "unique_images": len(rows),
        "duplicate_files_removed": files_on_disk - len(rows),
        "classes": len(per_class),
        "per_class_unique": {str(k): per_class[k] for k in sorted(per_class)},
        "per_class_min_max_mean": [min(per_class.values()), max(per_class.values()),
                                   round(len(rows) / len(per_class), 1)],
        "split_images": {s: sum(c.values()) for s, c in per_split_class.items()},
        "split_sheets": {s: len({r["sheet"] for r in rows if r["split"] == s})
                         for s in ("train", "val", "test")},
        "split_per_class_min_max": {s: [min(c.get(k, 0) for k in per_class),
                                        max(c.get(k, 0) for k in per_class)]
                                    for s, c in per_split_class.items()},
        "sheet_overlap_between_splits": 0,  # guaranteed by construction; asserted below
    }
    sheet_split = defaultdict(set)
    for r in rows:
        sheet_split[r["sheet"]].add(r["split"])
    assert all(len(v) == 1 for v in sheet_split.values()), "a sheet landed in two splits"
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def handwritten(paths: dict) -> list[dict]:
    img_dir = paths["raw"]["handwritten_images"]
    txt_dir = paths["raw"]["handwritten_text_original"]
    stem_to_group = {s: (g, split) for g, (stems, split) in HANDWRITTEN_GROUPS.items() for s in stems}
    rows = []
    for img in sorted(img_dir.glob("*.png")):
        stem = img.name[:-len(".png")]  # keep "D1." for "D1..png"
        if stem not in stem_to_group:
            raise ValueError(f"{img.name} has no story group; update HANDWRITTEN_GROUPS")
        group, split = stem_to_group[stem]
        txt = txt_dir / f"{stem}.txt"
        rows.append({
            "image": img.name,
            "text_original": txt.name if txt.exists() else "",
            "text_gt": f"{stem}.txt",
            "writer": stem[0],
            "story_group": group,
            "split": split,
            "image_md5": md5(img),
        })
    dup = [h for h, n in Counter(r["image_md5"] for r in rows).items() if n > 1]
    if dup:
        raise ValueError(f"Duplicate handwritten images found: {dup}")
    write_csv(paths["splits"]["handwritten"] / "pairs.csv", rows, list(rows[0].keys()))
    return rows


def main() -> None:
    paths = load_paths()
    s = letters(paths)
    print(f"letters52: {s['files_on_disk']} files -> {s['unique_images']} unique, "
          f"{s['classes']} classes, per-class min/max/mean {s['per_class_min_max_mean']}")
    print(f"  split images {s['split_images']}, sheets {s['split_sheets']}")
    print(f"  per-class min/max by split {s['split_per_class_min_max']}")
    rows = handwritten(paths)
    print(f"handwritten: {len(rows)} pairs, {dict(Counter(r['split'] for r in rows))}")


if __name__ == "__main__":
    main()
