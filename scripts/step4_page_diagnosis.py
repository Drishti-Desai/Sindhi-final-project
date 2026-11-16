"""Why do whole pages read through the Step 5 gap-rule boxes have a high CER? (diagnosis for Step 5; no choice is
made here, and no test page is read)

On the batch-2 VAL pages (verified part of each page, as in step4_prepare.page_item) the recognizer's CER is
measured four ways, each removing one source of error:
  a. saved crops        the verified word crops (only the word's own ink), page order        -> recognizer alone
  b. true boxes         the verified word boxes cut as rectangles from the page, page order  -> + other ink in the box
  c. gap-rule, re-ordered  the gap-rule boxes, sorted by the true word they cover most      -> + splits / merges
  d. gap-rule as is     the gap-rule boxes in our reading order (what the pipeline does)      -> + reading order
On the TRAIN + VAL pages (box finding only, no recognizer): how many gap-rule boxes split or merge words, mix ink of
two text lines, or put two true lines into one found line, and how often the reading order is wrong.

Output: results/step4/page_diagnosis.json
Usage: python scripts/step4_page_diagnosis.py
"""
from __future__ import annotations

import csv
import importlib.util
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


prep = load_module("step4_prepare", "scripts/step4_prepare.py")
step3 = prep.step3

from cnn_transformer.train_utils import get_device  # noqa: E402
from cnn_transformer.words import fixed_set, label_norm  # noqa: E402
from common.metrics import cer  # noqa: E402
from tesseract_ft.boxes import own_layout  # noqa: E402
from tesseract_ft.pages import load_batch2_pages  # noqa: E402


def found_boxes(p, box_cfg):
    """Gap-rule lines (reading order) restricted to boxes lying mostly on verified ink, as in page_item."""
    ink = p.binary < 128
    gt_ink = np.zeros_like(ink)
    for x0, y0, x1, y1 in p.gt_words:
        gt_ink[y0:y1, x0:x1] = True
    lines = step3.reading_order(own_layout(p.binary, box_cfg["line_finder"], box_cfg["gap_h"], box_cfg["gap_med"]))
    return [[x for x in ln if step3.tighten(x, ink) is not None
             and gt_ink[x[1]:x[3], x[0]:x[2]][ink[x[1]:x[3], x[0]:x[2]]].mean() > 0.5] for ln in lines]


def ink_owner(p) -> np.ndarray:
    """Pixel -> index of the verified word whose box holds it (-1: none)."""
    own = np.full(p.binary.shape, -1, int)
    ink = p.binary < 128
    for i, (x0, y0, x1, y1) in enumerate(p.gt_words):
        sub = own[y0:y1, x0:x1]
        sub[ink[y0:y1, x0:x1] & (sub < 0)] = i
    return own


def box_stats(pages, rows_by_page, box_cfg) -> dict:
    s = {"pages": len(pages), "true_words": 0, "found_boxes": 0, "boxes_one_word": 0, "boxes_part_of_a_word": 0,
         "boxes_2plus_words_same_line": 0, "boxes_mixing_two_lines": 0, "found_lines": 0,
         "found_lines_with_2plus_true_lines": 0, "pairs_out_of_order": 0, "pairs": 0}
    for p in pages:
        line_of = [int(r["line"]) for r in rows_by_page[p.name]]
        own = ink_owner(p)
        lines = found_boxes(p, box_cfg)
        s["true_words"] += len(p.gt_words)
        seq = []
        for ln in lines:
            if not ln:
                continue
            s["found_lines"] += 1
            tl = set()
            for x0, y0, x1, y1 in ln:
                ids, n = np.unique(own[y0:y1, x0:x1], return_counts=True)
                keep = {int(i): int(c) for i, c in zip(ids, n) if i >= 0}
                tot = sum(keep.values())
                words = [i for i, c in keep.items() if c >= 0.15 * tot]   # words with a real share of the box ink
                s["found_boxes"] += 1
                if len({line_of[i] for i in words}) > 1:
                    s["boxes_mixing_two_lines"] += 1
                elif len(words) > 1:
                    s["boxes_2plus_words_same_line"] += 1
                elif words:
                    w = words[0]
                    full = (own == w).sum()
                    s["boxes_one_word" if keep[w] >= 0.8 * full else "boxes_part_of_a_word"] += 1
                if words:
                    main = max(words, key=lambda i: keep[i])
                    seq.append(main)
                    tl.add(line_of[main])
            s["found_lines_with_2plus_true_lines"] += len(tl) > 1
        for i in range(len(seq)):
            for j in range(i + 1, len(seq)):
                if seq[i] != seq[j]:
                    s["pairs"] += 1
                    s["pairs_out_of_order"] += seq[i] > seq[j]
    s["share_pairs_out_of_order"] = round(s["pairs_out_of_order"] / max(1, s["pairs"]), 4)
    return s


def main() -> None:
    box_cfg = load_config("step3")["box_finder"]
    rows = [r for r in csv.DictReader(open(REPO_ROOT / "data" / "splits" / "handwritten_batch2" / "words.csv",
                                           encoding="utf-8")) if r["verified"] == "1"]
    rows_by_page: dict[str, list] = {}
    for r in rows:
        rows_by_page.setdefault(r["name"], []).append(r)
    device = get_device("auto")
    rec = step3.load_recognizer(device)
    model, vocab, rc = rec

    val = [p for p in load_batch2_pages(("VAL",)) if p.gt_words]
    refs, hyp = [], {k: [] for k in ("a_saved_crops", "b_true_boxes", "c_gap_rule_reordered", "d_gap_rule_as_is")}
    crops_root = REPO_ROOT / "data" / "preprocessed" / "handwritten_batch2_words"
    for p in val:
        ref = [label_norm(l) for l in p.gt_labels]
        refs.append(" ".join(ref))
        items = fixed_set("real", [(crops_root / r["file"], label_norm(r["label"])) for r in rows_by_page[p.name]],
                          None, rc, 0, 6)
        hyp["a_saved_crops"].append(" ".join(prep.step2.predict(model, items, vocab, device)[0]))
        hyp["b_true_boxes"].append(" ".join(step3.read_boxes(rec, p, list(p.gt_words), device)))
        flat = [b for ln in found_boxes(p, box_cfg) for b in ln]
        texts = step3.read_boxes(rec, p, flat, device) if flat else []
        own = ink_owner(p)
        key = []
        for x0, y0, x1, y1 in flat:
            ids, n = np.unique(own[y0:y1, x0:x1], return_counts=True)
            m = [(c, i) for i, c in zip(ids, n) if i >= 0]
            key.append(max(m)[1] if m else 10 ** 6)
        hyp["c_gap_rule_reordered"].append(" ".join(t for _, t in sorted(zip(key, texts), key=lambda x: x[0]) if t))
        hyp["d_gap_rule_as_is"].append(" ".join(t for t in texts if t))
    out = {"val_pages": [p.name for p in val], "val_words": sum(len(r.split()) for r in refs),
           "cer": {k: round(cer(refs, v), 4) for k, v in hyp.items()},
           "per_page_cer": {k: {p.name: round(cer([r], [h]), 3) for p, r, h in zip(val, refs, v)} for k, v in hyp.items()},
           "box_stats_val": box_stats(val, rows_by_page, box_cfg),
           "box_stats_train_val": box_stats([p for p in load_batch2_pages(("TRAIN", "VAL")) if p.gt_words],
                                            rows_by_page, box_cfg)}
    (REPO_ROOT / "results" / "step4" / "page_diagnosis.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                                                        encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("val_words", "cer", "box_stats_val", "box_stats_train_val")}, indent=1))
    print(json.dumps(out["per_page_cer"], indent=0))


if __name__ == "__main__":
    main()
