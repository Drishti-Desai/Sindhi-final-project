"""Manifest for the second handwritten batch (copied 2026-11-13 from ~/Desktop/new data).

Writes data/splits/handwritten_batch2/manifest.csv: one row per image or text file with its group,
status and notes. Statuses:
  ok                    image + matching text
  text_mismatch         the text file belongs to other CSV rows; a draft correction exists
  ok_text_corrected     image + text edited by hand to match the page (text_corrected/<name>.txt)
  duplicate_no_text     a copy of one of our existing D pages (incl. FINAL_TEST D2/D6..); never used

Splits (decided with the user 2026-11-13, see DECISIONS.md):
  TEST_UNSEEN_WRITER    the purple-pen writer: never used for training or validation
  VAL                   ~10% of each other writer's files (whole files, duplicate questions together),
                        used only to pick the best checkpoint
  TRAIN                 everything else, including writer J
  EXCLUDED              duplicate_no_text files

Also drafts corrected texts for the mismatched pages from the legal Q&A CSV the pages were copied
from (row = circled number on the page - 1, checked against the images on 2026-11-13) into
data/raw/handwritten_batch2/text_corrected/<name>.txt. Drafts need the user's confirmation.

Usage: python scripts/prepare_batch2.py [--legal-csv PATH]
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import load_paths  # noqa: E402

SUB_IMG = "Drishti-Desai SINDHI-PROJECT main input"
SUB_TXT = "Drishti-Desai SINDHI-PROJECT main textextract"
# image name -> legal-CSV data rows actually shown on the page (verified visually 2026-11-13)
MISMATCHED = {"350-353": range(349, 353), "354-357": range(353, 357), "358-361": range(357, 361),
              "362-365": range(361, 365), "370-366": range(365, 370)}
# pages photographed sideways or upside down: degrees to turn COUNTER-clockwise to make them upright
# (checked visually on 2026-11-13); applied at segmentation time
ORIENTATION = {"221-223": "180", "358-361": "90", "362-365": "90", "370-366": "90"}
SEED = 42
VAL_FRAC = 0.10
DUPLICATE_GROUPS = [{"A7", "A11"}, {"A15", "A24"}, {"H102", "H110"}, {"J214", "J215"}]  # same question twice
TEST_WRITER = "purple_pen"

# writers: A/H/J confirmed by the user (2026-11-13) to be the same people as A1-A5 / H1-H5 / J1-J5.
# Top-level groups are guesses from pen/paper, except purple_pen (confirmed: one person, nothing else).
WRITER_GUESS = {"A": "A (confirmed = A1-A5)", "H": "H (confirmed = H1-H5)", "J": "J (confirmed = J1-J5)"}
TOP_GROUPS = {"notebook_black_pen": ["221-223", "224-226", "227-229"],
              "spaced_letters_pencil": ["305-306", "307-309", "310", "311-312", "313", "314-316"],
              "neat_pen_rotated": ["350-353", "354-357", "358-361", "362-365", "370-366"],
              "purple_pen": ["383", "384", "385", "386", "387", "dd381", "dd382"],
              "poem": ["poem1"]}


def image_for(folder: Path, stem: str) -> Path | None:
    return next((folder / f"{stem}{e}" for e in (".png", ".PNG") if (folder / f"{stem}{e}").exists()), None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legal-csv", default=str(Path.home() / "Downloads" / "Sindhi_Legal_Dataset.csv"))
    args = ap.parse_args()
    paths = load_paths()
    root = paths["raw"]["handwritten_batch2"]
    top_of = {n: g for g, ns in TOP_GROUPS.items() for n in ns}
    rows = []
    for txt in sorted((root / SUB_TXT).glob("*.txt")):
        stem = txt.stem
        img = image_for(root / SUB_IMG, stem)
        rows.append({"name": stem, "group": stem[0], "writer_guess": WRITER_GUESS[stem[0]],
                     "image": img.relative_to(root).as_posix() if img else "", "text": txt.relative_to(root).as_posix(),
                     "status": "ok" if img else "text_without_image", "orientation": "", "notes": ""})
    for img in sorted((root / SUB_IMG).glob("D*")):
        rows.append({"name": img.stem, "group": "D", "writer_guess": "D (existing pages)",
                     "image": img.relative_to(root).as_posix(), "text": "", "status": "duplicate_no_text",
                     "orientation": "", "notes": "copy of an existing NEW page (D2 = the old faint D2; D4 = old copy of D3); "
                                               "D2 and D6.. are FINAL_TEST - never use"})
    for txt in sorted(root.glob("*.txt")):
        stem = txt.stem
        img = image_for(root, stem)
        status = "text_mismatch" if stem in MISMATCHED else ("ok" if img else "text_without_image")
        grp = top_of.get(stem, "?")
        rows.append({"name": stem, "group": "top",
                     "writer_guess": f"top:{grp}" + (" (confirmed)" if grp == "purple_pen" else " (guess)"),
                     "image": img.relative_to(root).as_posix() if img else "", "text": txt.relative_to(root).as_posix(),
                     "status": status, "orientation": ORIENTATION.get(stem, ""), "notes": ""})  # notes filled below
    for r in rows:
        r["words"] = len((root / r["text"]).read_text(encoding="utf-8").split()) if r["text"] else 0

    # texts trimmed/edited by hand to match what is visible on the page (override the original text)
    HAND_CORRECTED = {"387": "photo is cut off after 'پولنگ اسٽيشن تي وڃڻ،'; text trimmed to the visible words "
                             "(43 -> 32 words); the half-visible 'سڃ…' is left out"}
    for r in rows:
        if r["name"] in HAND_CORRECTED:
            r["status"] = "ok_text_corrected"
            r["notes"] = HAND_CORRECTED[r["name"]] + " -> text_corrected/387.txt"
            r["words"] = len((paths["raw"]["handwritten_batch2_text_corrected"] / f"{r['name']}.txt")
                             .read_text(encoding="utf-8").split())

    # draft corrected texts for the mismatched pages, from the legal Q&A CSV
    legal = Path(args.legal_csv)
    out = paths["raw"]["handwritten_batch2_text_corrected"]
    if legal.exists():
        data = list(csv.reader(open(legal, encoding="utf-8-sig")))[1:]
        out.mkdir(parents=True, exist_ok=True)
        for name, rng in MISMATCHED.items():
            blocks = [f"{data[i][0].strip()}\n{data[i][1].strip()}" for i in rng]
            (out / f"{name}.txt").write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
            wrong = (root / f"{name}.txt").read_text(encoding="utf-8").split("\n")[0].strip()
            src = next((i for i, d in enumerate(data) if wrong and wrong in d[0]), -1)
            for r in rows:
                if r["name"] == name:
                    r["notes"] = (f"text file = CSV rows {src}-{src + len(rng) - 1}; the page shows rows "
                                  f"{rng.start}-{rng.stop - 1} -> draft text_corrected/{name}.txt (needs confirmation)")
    else:
        print(f"legal CSV not found at {legal}: corrected drafts not written")

    assign_splits(rows, top_of)
    dst = paths["splits"]["handwritten_batch2"]
    dst.mkdir(parents=True, exist_ok=True)
    for r in rows:
        r["rotate_ccw"] = r.pop("orientation")
    fields = ["name", "group", "writer", "writer_guess", "split", "status", "rotate_ccw", "words", "image", "text", "notes"]
    with open(dst / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print("status:", dict(Counter(r["status"] for r in rows)))
    print("split:", dict(Counter(r["split"] for r in rows)),
          "| words:", {s: sum(r["words"] for r in rows if r["split"] == s) for s in ("TRAIN", "VAL", "TEST_UNSEEN_WRITER")})
    print("usable words (status ok):", sum(r["words"] for r in rows if r["status"] == "ok"),
          "| in mismatched pages:", sum(r["words"] for r in rows if r["status"] == "text_mismatch"))


def assign_splits(rows: list[dict], top_of: dict) -> None:
    """Purple pen -> unseen-writer test; ~10% of every other writer's files -> VAL; rest -> TRAIN."""
    rng = random.Random(SEED)
    for r in rows:
        r["writer"] = r["group"] if r["group"] in "AHJD" else top_of.get(r["name"], "?")
    usable = [r for r in rows if r["status"] in ("ok", "ok_text_corrected", "text_mismatch")]
    for r in rows:
        r["split"] = "EXCLUDED" if r["status"] == "duplicate_no_text" else ""
    for r in usable:
        if r["writer"] == TEST_WRITER:
            r["split"] = "TEST_UNSEEN_WRITER"
    # units = files, except duplicate questions which move together
    unit_of = {n: frozenset(g) for g in DUPLICATE_GROUPS for n in g}
    by_writer: dict[str, list[frozenset]] = {}
    for r in usable:
        if r["split"]:
            continue
        u = unit_of.get(r["name"], frozenset([r["name"]]))
        by_writer.setdefault(r["writer"], [])
        if u not in by_writer[r["writer"]]:
            by_writer[r["writer"]].append(u)
    val = set()
    for writer, units in sorted(by_writer.items()):
        units = sorted(units, key=lambda u: sorted(u))
        rng.shuffle(units)
        k = round(len(units) * VAL_FRAC) if len(units) >= 5 else 0  # tiny groups stay in TRAIN
        for u in units[:k]:
            val |= u
    for r in usable:
        if not r["split"]:
            r["split"] = "VAL" if r["name"] in val else "TRAIN"


if __name__ == "__main__":
    main()
