"""Sindhi word list for Step 3 lever 2 (and Step 4): known words with counts. Nothing is trained.

Sources, split into sentences; any sentence that overlaps a test or FINAL_TEST text (tesseract_ft/texts.py) is
dropped before its words are counted:
  Wikipedia-sd (words seen at least WIKI_MIN_COUNT times), the POS-tagging sentences (tags stripped, see
  scripts/inspect_pos_data.py), batch 2 TRAIN + VAL texts, TESS_TRAIN texts, Sindhi_Legal_Dataset.csv.
Words are normalised like recognizer labels (no harakat), punctuation removed.

Output: data/preprocessed/lexicon/words.tsv (word, count, sources), data/preprocessed/lexicon/report.json
Usage: python scripts/build_lexicon.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import REPO_ROOT, load_paths  # noqa: E402
from common.text import strip_punct  # noqa: E402
from tesseract_ft.pages import label_norm_words  # noqa: E402
from tesseract_ft.texts import SENT_SPLIT, LeakFilter, training_texts  # noqa: E402

WIKI_MIN_COUNT = 3   # rarer Wikipedia tokens are mostly typos, names and foreign words


def main() -> None:
    paths = load_paths()
    leak = LeakFilter(paths)
    sources: dict[str, list[str]] = training_texts(paths)
    sources["pos_sentences"] = (REPO_ROOT / "data" / "preprocessed" / "pos_text" / "sentences_clean.txt"
                                ).read_text(encoding="utf-8").splitlines()
    wiki = pq.read_table(paths["raw"]["corpus_wikipedia_sd"] / "train-00000-of-00001.parquet", columns=["text"])
    sources["wikipedia"] = wiki.column("text").to_pylist()
    counts: dict[str, collections.Counter] = {}
    report = {}
    for name, docs in sources.items():
        c, kept, dropped = collections.Counter(), 0, 0
        for doc in docs:
            for sent in SENT_SPLIT.split(doc):
                if not sent.strip():
                    continue
                if leak.overlaps(sent):
                    dropped += 1
                    continue
                kept += 1
                c.update(w for w in (strip_punct(x) for x in label_norm_words(sent)) if w and " " not in w)
        if name == "wikipedia":
            c = collections.Counter({w: n for w, n in c.items() if n >= WIKI_MIN_COUNT})
        counts[name] = c
        report[name] = {"sentences_kept": kept, "sentences_dropped_test_overlap": dropped, "unique_words": len(c)}
        print(f"{name:18s} sentences kept {kept:7d}, dropped (test overlap) {dropped:4d}, unique words {len(c):6d}")
    total = collections.Counter()
    for c in counts.values():
        total.update(c)
    out = REPO_ROOT / "data" / "preprocessed" / "lexicon"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "words.tsv", "w", encoding="utf-8") as fh:
        for w, n in total.most_common():
            fh.write(f"{w}\t{n}\t{','.join(k for k, c in counts.items() if w in c)}\n")
    report["total_unique_words"] = len(total)
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"total unique words {len(total)} -> {out / 'words.tsv'}")


if __name__ == "__main__":
    main()
