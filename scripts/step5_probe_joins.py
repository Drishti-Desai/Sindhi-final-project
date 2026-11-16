"""Step 5 planning probe: how much can JOINING neighbouring gap-rule pieces help? (VAL pages only; nothing is trained,
no setting is tuned, no test page is read)

On the batch-2 VAL pages (verified part of each page, as in scripts/step4_page_diagnosis.py) the recognizer reads
each gap-rule piece and every run of 2-3 neighbouring pieces (only their own ink). Readings compared by CER:
  pieces           the gap-rule pieces as they are (the Step 4 page reading)
  oracle joins     neighbouring pieces joined exactly when they belong to the same true word: the best any
                   join-only method can do (it cannot undo the gap rule's merges)
  rule: word list  join so that as many characters as possible are read as known words (Step 4 word list,
                   training text only); ties keep the pieces. No weights.
  rule: recognizer join when the recognizer's total log-probability of the joined reading beats the pieces'
                   readings. No weights.
Each also with the lines in the true order of the words, to separate reading order.

Output: results/step5/probe_joins_<units>.json
--units spans: the units are single ink spans instead of gap-rule pieces (groups of up to 6), so a join method can
also undo the gap rule's merges.

Usage: python scripts/step5_probe_joins.py [--units pieces|spans]
"""
from __future__ import annotations

import argparse
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


diag = load_module("step4_page_diagnosis", "scripts/step4_page_diagnosis.py")

from cnn_transformer.train_utils import get_device  # noqa: E402
from cnn_transformer.words import label_norm  # noqa: E402
from common.metrics import cer  # noqa: E402
from tesseract_ft.boxes import group_box, group_comps, groups_from_cuts, line_structures  # noqa: E402
from tesseract_ft.cuts import read_groups  # noqa: E402
from tesseract_ft.pages import load_batch2_pages  # noqa: E402
from xlmr.correct import split_tail  # noqa: E402

MAX_JOIN = {"pieces": 3, "spans": 6}


def dp(n: int, score, max_join: int) -> list[tuple[int, int]]:
    """Best split of units 0..n-1 into runs of <= max_join; score(i, j) -> comparable tuple, summed."""
    best = [None] * (n + 1)
    best[0] = ((0.0,) * 3, [])
    for j in range(1, n + 1):
        for i in range(max(0, j - max_join), j):
            s = score(i, j - 1)
            tot = tuple(a + b for a, b in zip(best[i][0], s))
            if best[j] is None or tot > best[j][0]:
                best[j] = (tot, best[i][1] + [(i, j - 1)])
    return best[n][1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", choices=("pieces", "spans"), default="pieces")
    units = ap.parse_args().units
    mj = MAX_JOIN[units]
    box_cfg = load_config("step3")["box_finder"]
    lexicon = {l.split("\t")[0] for l in open(REPO_ROOT / "data" / "preprocessed" / "step4" / "lexicon.tsv",
                                              encoding="utf-8")}
    device = get_device("auto")
    rec = diag.step3.load_recognizer(device)
    methods = ("pieces", "oracle_joins", "rule_word_list", "rule_recognizer")
    hyp = {m: [] for m in methods}
    hyp_true_order = {m: [] for m in methods}
    refs, stats = [], {"pieces": 0, "true_words": 0, "joins": {m: 0 for m in methods}}
    for p in [p for p in load_batch2_pages(("VAL",)) if p.gt_words]:
        ink = p.binary < 128
        gt_ink = np.zeros_like(ink)
        for x0, y0, x1, y1 in p.gt_words:
            gt_ink[y0:y1, x0:x1] = True
        owner = diag.ink_owner(p)
        refs.append(" ".join(label_norm(l) for l in p.gt_labels))
        stats["true_words"] += len(p.gt_words)
        lines, labels = line_structures(p.binary, box_cfg["line_finder"])
        page_out = {m: [] for m in methods}          # (line y, [(owner word, text)])
        for ln in lines:
            gaps = ln["gaps"]
            thr = max(box_cfg["gap_h"] * ln["h"], box_cfg["gap_med"] * float(np.median(gaps)) if gaps else 0.0)
            pieces = groups_from_cuts(len(ln["spans"]), {k for k, g in enumerate(gaps) if g > thr} if units == "pieces"
                                      else set(range(len(gaps))))
            keep = []
            for i0, i1 in pieces:
                x0, y0, x1, y1 = group_box(ln, i0, i1)
                m = ink[y0:y1, x0:x1]
                keep.append(m.any() and gt_ink[y0:y1, x0:x1][m].mean() > 0.5)
            runs, cur = [], []
            for pc, k in zip(pieces, keep):
                if k:
                    cur.append(pc)
                elif cur:
                    runs.append(cur)
                    cur = []
            if cur:
                runs.append(cur)
            for run in runs:
                n = len(run)
                stats["pieces"] += n
                groups = [(i, j) for i in range(n) for j in range(i, min(n, i + mj))]
                spans = [(run[i][0], run[j][1]) for i, j in groups]
                reads = dict(zip(groups, read_groups(rec, labels, ln, spans, device)))

                def own(i0, i1):
                    ids = [c["i"] for c in group_comps(ln, i0, i1)]
                    x0, y0, x1, y1 = group_box(ln, i0, i1)
                    o = owner[y0:y1, x0:x1][np.isin(labels[y0:y1, x0:x1], ids)]
                    o = o[o >= 0]
                    return int(np.bincount(o).argmax()) if len(o) else -1
                owners = [own(*pc) for pc in run]
                oracle, i = [], 0
                while i < n:
                    j = i
                    while j + 1 < n and owners[j + 1] == owners[i] and owners[i] >= 0:
                        j += 1
                    oracle.append((i, j))
                    i = j + 1

                def known(i, j):
                    t = split_tail(reads[(i, j)][0])[0]
                    return len(t) if t in lexicon else 0
                chosen = {
                    "pieces": [(i, i) for i in range(n)],
                    "oracle_joins": oracle,
                    "rule_word_list": dp(n, lambda i, j: (known(i, j), -(j - i), 0.0), mj),
                    "rule_recognizer": dp(n, lambda i, j: (reads[(i, j)][1] * (len(reads[(i, j)][0]) + 1), 0.0, 0.0), mj),
                }
                y = float(np.mean([(group_box(ln, *pc)[1] + group_box(ln, *pc)[3]) / 2 for pc in run]))
                for m, gs in chosen.items():
                    stats["joins"][m] += n - len(gs)
                    out = []
                    for i, j in gs:
                        if (i, j) not in reads:      # an oracle run longer than the join limit
                            reads[(i, j)] = read_groups(rec, labels, ln, [(run[i][0], run[j][1])], device)[0]
                        out.append((own(run[i][0], run[j][1]), reads[(i, j)][0]))
                    page_out[m].append((y, out))
        for m in methods:
            seq = [w for _, ws in sorted(page_out[m], key=lambda x: x[0]) for w in ws]
            hyp[m].append(" ".join(t for _, t in seq if t))
            hyp_true_order[m].append(" ".join(t for _, t in sorted(seq, key=lambda x: (x[0] < 0, x[0])) if t))
    out = {"units": units, "pages": len(refs), "true_words": stats["true_words"], "pieces": stats["pieces"], "joins": stats["joins"],
           "cer_our_order": {m: round(cer(refs, hyp[m]), 4) for m in methods},
           "cer_true_order": {m: round(cer(refs, hyp_true_order[m]), 4) for m in methods},
           "reference_cer_true_word_crops": 0.1216}
    d = REPO_ROOT / "results" / "step5"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"probe_joins_{units}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
