"""Text sources and the leakage filter shared by Step 3 (word list for lever 2) and later Step 4.

A sentence "overlaps the tests" if it shares 4 consecutive words with, or is contained in, the text of any test
page: the unseen writer, TESS_TEST, FINAL_TEST, and the batch 2 EXCLUDED copies of FINAL_TEST pages.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

from tesseract_ft.pages import label_norm_words

LEGAL_CSV = Path.home() / "Downloads" / "Sindhi_Legal_Dataset.csv"   # read only, never modified
SENT_SPLIT = re.compile(r"[.۔؟?!\n]+")


def ngrams(words: list[str], n: int = 4) -> set:
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def test_texts(paths) -> list[str]:
    texts = []
    for p in csv.DictReader(open(paths["splits"]["handwritten"] / "pairs.csv", encoding="utf-8")):
        if p["split"] in ("TESS_TEST", "FINAL_TEST"):
            texts.append((paths["raw"]["handwritten_text_gt"] / p["text_gt"]).read_text(encoding="utf-8"))
    root = paths["raw"]["handwritten_batch2"]
    for m in csv.DictReader(open(paths["splits"]["handwritten_batch2"] / "manifest.csv", encoding="utf-8")):
        if m["split"] in ("TEST_UNSEEN_WRITER", "EXCLUDED") and m["text"]:
            corr = paths["raw"]["handwritten_batch2_text_corrected"] / f"{m['name']}.txt"
            src = corr if corr.exists() else root / m["text"]
            if src.exists():
                texts.append(src.read_text(encoding="utf-8"))
    return [" ".join(label_norm_words(t)) for t in texts]


class LeakFilter:
    def __init__(self, paths):
        self.tests = test_texts(paths)
        self.grams = set().union(*(ngrams(t.split()) for t in self.tests))
        self.joined = "\n".join(self.tests)

    def overlaps(self, text: str) -> bool:
        words = label_norm_words(text)
        if not words:
            return False
        return bool(ngrams(words) & self.grams) or (len(words) >= 2 and " ".join(words) in self.joined)


def training_texts(paths) -> dict[str, list[str]]:
    """Texts we may learn from: batch 2 TRAIN + VAL pages, TESS_TRAIN pages, and the legal CSV."""
    out: dict[str, list[str]] = {"batch2_train_val": [], "tess_train": [], "legal_csv": []}
    root = paths["raw"]["handwritten_batch2"]
    for m in csv.DictReader(open(paths["splits"]["handwritten_batch2"] / "manifest.csv", encoding="utf-8")):
        if m["split"] in ("TRAIN", "VAL") and m["text"]:
            corr = paths["raw"]["handwritten_batch2_text_corrected"] / f"{m['name']}.txt"
            src = corr if corr.exists() and m["status"] in ("text_mismatch", "ok_text_corrected") else root / m["text"]
            if src.exists():
                out["batch2_train_val"].append(src.read_text(encoding="utf-8"))
    for p in csv.DictReader(open(paths["splits"]["handwritten"] / "pairs.csv", encoding="utf-8")):
        if p["split"] == "TESS_TRAIN":
            out["tess_train"].append((paths["raw"]["handwritten_text_gt"] / p["text_gt"]).read_text(encoding="utf-8"))
    for r in csv.DictReader(open(LEGAL_CSV, encoding="utf-8-sig")):
        out["legal_csv"] += [v for v in r.values() if v and v.strip()]
    return out
