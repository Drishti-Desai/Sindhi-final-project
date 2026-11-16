"""Step 4 text splits. Every sentence that overlaps a test or FINAL_TEST page text is removed first
(tesseract_ft/texts.py: LeakFilter, the same filter as the Step 3 word list).

  legal CSV   80/10/10 by question-answer row (rows sharing a question or an answer stay together). Rows whose text
              appears on a batch 2 VAL page go to val, rows on a batch 2 TRAIN page go to train, so the real
              validation pages never have their sentences in training.
  POS         518 cleaned sentences, 80/10/10
  Wikipedia   split by ARTICLE: ~1,000 sentences of 4-40 words each for val and test, the rest train
Units are sentences (answers are split at . ؟ ! ۔).
"""
from __future__ import annotations

import csv
import re

import numpy as np
import pyarrow.parquet as pq

from common.paths import REPO_ROOT
from tesseract_ft.pages import label_norm_words
from tesseract_ft.texts import LEGAL_CSV, LeakFilter, ngrams

SPLITS = ("train", "val", "test")
KEEP_PUNCT_SPLIT = re.compile(r"(?<=[.۔؟?!])\s+")
SEED = 42


def sentences_of(text: str) -> list[str]:
    return [s.strip() for s in KEEP_PUNCT_SPLIT.split(" ".join(text.split())) if len(s.split()) >= 2]


def page_texts(paths, split: str) -> list[str]:
    root = paths["raw"]["handwritten_batch2"]
    out = []
    for m in csv.DictReader(open(paths["splits"]["handwritten_batch2"] / "manifest.csv", encoding="utf-8")):
        if m["split"] == split and m["text"]:
            corr = paths["raw"]["handwritten_batch2_text_corrected"] / f"{m['name']}.txt"
            src = corr if corr.exists() and m["status"] in ("text_mismatch", "ok_text_corrected") else root / m["text"]
            if src.exists():
                out.append(" ".join(label_norm_words(src.read_text(encoding="utf-8"))))
    return out


def legal_splits(paths, leak: LeakFilter, rng) -> tuple[dict, dict]:
    rows = [[v for v in r.values() if v and v.strip()] for r in csv.DictReader(open(LEGAL_CSV, encoding="utf-8-sig"))]
    # group rows that share a question or an answer
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    first = {}
    for i, r in enumerate(rows):
        for v in r:
            key = " ".join(label_norm_words(v))
            if key in first:
                parent[find(i)] = find(first[key])
            first.setdefault(key, i)
    groups: dict[int, list[int]] = {}
    for i in range(len(rows)):
        groups.setdefault(find(i), []).append(i)
    val_pages = [set(ngrams(t.split())) for t in page_texts(paths, "VAL")]
    train_pages = [set(ngrams(t.split())) for t in page_texts(paths, "TRAIN")]
    out = {s: [] for s in SPLITS}
    stats = {"rows": len(rows), "groups": len(groups), "groups_with_test_overlap": 0, "forced_val": 0, "forced_train": 0}
    order = list(groups.values())
    rng.shuffle(order)
    n = len(order)
    for gi, members in enumerate(order):
        texts = [v for i in members for v in rows[i]]
        if any(leak.overlaps(t) for t in texts):
            stats["groups_with_test_overlap"] += 1
            continue
        grams = set().union(*(ngrams(label_norm_words(t)) for t in texts))
        if any(grams & p for p in val_pages):
            split = "val"
            stats["forced_val"] += 1
        elif any(grams & p for p in train_pages):
            split = "train"
            stats["forced_train"] += 1
        else:
            split = "train" if gi < 0.8 * n else ("val" if gi < 0.9 * n else "test")
        out[split] += [s for t in texts for s in sentences_of(t)]
    return out, stats


def pos_splits(leak: LeakFilter, rng) -> dict:
    sents = (REPO_ROOT / "data" / "preprocessed" / "pos_text" / "sentences_clean.txt").read_text(encoding="utf-8").splitlines()
    sents = [s for s in sents if s.strip() and not leak.overlaps(s)]
    idx = rng.permutation(len(sents))
    n = len(sents)
    return {"train": [sents[i] for i in idx[:int(0.8 * n)]], "val": [sents[i] for i in idx[int(0.8 * n):int(0.9 * n)]],
            "test": [sents[i] for i in idx[int(0.9 * n):]]}


def wiki_splits(paths, leak: LeakFilter, rng, n_eval: int = 1000) -> tuple[dict, dict]:
    docs = pq.read_table(paths["raw"]["corpus_wikipedia_sd"] / "train-00000-of-00001.parquet",
                         columns=["text"]).column("text").to_pylist()
    order = rng.permutation(len(docs))
    out = {s: [] for s in SPLITS}
    dropped = 0
    for i in order:
        sents = []
        for para in docs[i].split("\n"):
            for s in sentences_of(para):
                if leak.overlaps(s):
                    dropped += 1
                else:
                    sents.append(s)
        if len(out["val"]) < n_eval:
            out["val"] += [s for s in sents if 4 <= len(s.split()) <= 40]
        elif len(out["test"]) < n_eval:
            out["test"] += [s for s in sents if 4 <= len(s.split()) <= 40]
        else:
            out["train"] += sents
    return out, {"articles": len(docs), "sentences_dropped_test_overlap": dropped}


def build_splits(paths) -> tuple[dict, dict]:
    """{source: {split: [sentences]}}, stats."""
    leak = LeakFilter(paths)
    rng = np.random.default_rng(SEED)
    legal, s_legal = legal_splits(paths, leak, rng)
    pos = pos_splits(leak, rng)
    wiki, s_wiki = wiki_splits(paths, leak, rng)
    return {"legal": legal, "pos": pos, "wiki": wiki}, {"legal": s_legal, "wiki": s_wiki}


def lexicon_from(sentences: list[str], min_count: int = 1) -> dict[str, int]:
    """Word -> count; words normalised like recognizer labels, punctuation removed."""
    from common.text import strip_punct
    counts: dict[str, int] = {}
    for s in sentences:
        for w in label_norm_words(s):
            w = strip_punct(w)
            if w and " " not in w:
                counts[w] = counts.get(w, 0) + 1
    return {w: c for w, c in counts.items() if c >= min_count}

