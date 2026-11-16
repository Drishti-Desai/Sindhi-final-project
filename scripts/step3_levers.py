"""Step 3, levers 1 and 2 (no download, no training). Rules agreed with the user on 2026-11-14:
everything is chosen on the TUNING pages (batch 2 TRAIN + VAL); the unseen-writer test is measured ONCE, with the
final chosen method only (it was already looked at in the baseline); TESS_TEST (H1, J2, J3) likewise.

Lines are always our own (slope-tracking finder, chosen in the baseline). Methods:
  baseline_own        our gap rule (baseline settings)
  baseline_tess_lines Tesseract's boxes inside our lines (baseline settings)
  lever1              Tesseract's cuts snapped to the ink; Tesseract setting re-chosen among the TOP_TESS best
                      line settings of the baseline grid; lines where Tesseract finds nothing use the gap rule
  lever2              the recognizer scores alternative cuts (tesseract_ft/cuts.py); weights by random search,
                      Tesseract agreement taken from the lever 1 setting; lever2_no_tess = the same with e = 0
The final method is the one with the best word-box F1 on the tuning pages.

Reported for every method on the tuning pages, per writer (in-sample: these pages chose the settings, and
the recognizer was fine-tuned on the TRAIN ones), with VAL pages also shown alone. For the final method only:
the unseen writer and TESS_TEST. Word boxes as in the baseline (IoU >= 0.5), and the recognizer's page CER/WER
reading the found boxes.

Outputs: results/step3/levers_{metrics.json,search.csv,per_page.csv}, results/step3/debug/final/, the gate.
Usage: python scripts/step3_levers.py [--dry]   (--dry: 8 tuning pages, VAL pages stand in for the tests,
       nothing written; checks the code paths without touching the test pages)
"""
from __future__ import annotations

import argparse
import collections
import csv
import importlib.util
import json
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cnn_transformer.train_utils import get_device  # noqa: E402
from common.gates import read_gate, require_gate, write_gate  # noqa: E402
from common.metrics import cer, wer  # noqa: E402
from common.paths import REPO_ROOT, load_config, load_paths  # noqa: E402
from common.text import strip_punct  # noqa: E402
from tesseract_ft.boxes import group_box, groups_from_cuts, line_structures, tesseract_in_lines  # noqa: E402
from tesseract_ft.cuts import (best_cuts, candidate_groups, read_groups, snap_cuts, tess_cut_flags,  # noqa: E402
                               word_features)
from tesseract_ft.pages import label_norm_words, load_batch2_pages, load_tess_test_pages  # noqa: E402

_spec = importlib.util.spec_from_file_location("step3_baseline", REPO_ROOT / "scripts" / "step3_baseline.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

TOP_TESS = 12        # lever 1 re-chooses among the 12 best line settings of the baseline grid
N_RANDOM = 300       # lever 2: random weight settings, then coordinate refinement around the best
SEED = 42


def gap_cuts(line: dict, gap_h: float, gap_med: float) -> set[int]:
    gaps = line["gaps"]
    thr = max(gap_h * line["h"], gap_med * float(np.median(gaps)) if gaps else 0.0)
    return {k for k, g in enumerate(gaps) if g > thr}


def boxes_from(lines: list[dict], cuts: list[set[int]]) -> list[list[tuple]]:
    return [[group_box(ln, i0, i1) for i0, i1 in groups_from_cuts(len(ln["spans"]), c)] for ln, c in zip(lines, cuts)]


class Page:
    """An evaluation page plus its line structures and cached Tesseract / recognizer results."""

    def __init__(self, page, finder: str):
        self.p = page
        self.lines, self.labels = line_structures(page.binary, finder)
        self.tess: dict[tuple, list] = {}
        self.groups: list[list[tuple]] = []
        self.feats: list[list[tuple]] = []

    def run_tess(self, s: dict) -> list[list[tuple]]:
        key = tuple(sorted(s.items()))
        if key not in self.tess:
            boxes, _ = tesseract_in_lines(self.p.binary, self.p.gray, [ln["ids"] for ln in self.lines], self.labels,
                                          s["lang"], s["input"] == "gray", s["scale"], int(s["psm"]))
            self.tess[key] = boxes
        return self.tess[key]


def cuts_for(pg: Page, method: str, prm: dict) -> list[set[int]]:
    own = prm["own"]
    if method == "baseline_own":
        return [gap_cuts(ln, own["gap_h"], own["gap_med"]) for ln in pg.lines]
    if method == "baseline_tess_lines":
        raise ValueError("boxes, not cuts")
    if method == "lever1":
        out = []
        for ln, tb in zip(pg.lines, pg.run_tess(prm["tess"])):
            c = snap_cuts(ln, tb)
            out.append(c if c is not None else gap_cuts(ln, own["gap_h"], own["gap_med"]))
        return out
    w = prm["w"]
    tess = pg.run_tess(prm["tess"])
    return [best_cuts(ln, g, f, tess_cut_flags(ln, tb) if w["e"] else [0] * len(ln["gaps"]), **w)
            for ln, g, f, tb in zip(pg.lines, pg.groups, pg.feats, tess)]


def line_boxes(pg: Page, method: str, prm: dict) -> list[list[tuple]]:
    if method == "baseline_tess_lines":
        return pg.run_tess(prm["tess_baseline"])
    return boxes_from(pg.lines, cuts_for(pg, method, prm))


def evaluate(pages: list[Page], method: str, prm: dict) -> tuple[dict, list[dict]]:
    evs = [base.evaluate(pg.p, base.reading_order(line_boxes(pg, method, prm))) for pg in pages]
    return base.total(evs, "w"), evs


def text_scores(pages: list[Page], method: str, prm: dict, rec, device) -> dict:
    """The recognizer reading the found boxes (box crops, as the pipeline will do):
    found_words  CER/word accuracy over the true words whose box was found (IoU >= 0.5), each found box read and
                 compared with that word's label (comparable to the Step 2 reference, which read the true crops);
    page         page-level CER/WER over all found boxes in reading order against the whole page text (missed,
                 merged and extra words all count; pages holding untranscribed writing inflate it)."""
    refs, hyps, w_ref, w_hyp = [], [], [], []
    for pg in pages:
        lines = base.reading_order(line_boxes(pg, method, prm))
        ev = base.evaluate(pg.p, lines)
        flat = [t for t in (base.tighten(x, pg.p.binary < 128) for ln in lines for x in ln) if t is not None]
        refs.append(" ".join(pg.p.words))
        hyps.append(" ".join(t for t in base.read_boxes(rec, pg.p, flat, device) if t))
        found = ev["w"]["matched"]
        if found:
            reads = base.read_boxes(rec, pg.p, [ev["w"]["pred"][i] for i, _ in found], device)
            w_ref += [label_norm_words(pg.p.gt_labels[j])[0] if label_norm_words(pg.p.gt_labels[j]) else ""
                      for _, j in found]
            w_hyp += reads
    return {"found_words": len(w_ref), "found_words_cer": round(cer(w_ref, w_hyp), 4),
            "found_words_accuracy": round(float(np.mean([a == b for a, b in zip(w_ref, w_hyp)])), 4) if w_ref else 0.0,
            "page_cer": round(cer(refs, hyps), 4), "page_wer": round(wer(refs, hyps), 4)}


def report(pages: list[Page], method: str, prm: dict, rec, device) -> dict:
    tot, evs = evaluate(pages, method, prm)
    return {"pages": len(pages), "words": sum(len(pg.p.gt_words) for pg in pages), "word_boxes": tot,
            "split": sum(e["sm"]["gt_words_split"] for e in evs), "merged": sum(e["sm"]["pred_boxes_merging_2plus"] for e in evs),
            "recognizer_on_found_boxes": text_scores(pages, method, prm, rec, device)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    global N_RANDOM
    require_gate("step2_cnn_transformer", allow_user_accepted=True)
    paths = load_paths()
    out = paths["results"]["step3"]
    if args.dry:
        N_RANDOM = 4
        out = Path(str(out) + "_dry")
        out.mkdir(exist_ok=True)
        for f in ("baseline_metrics.json", "baseline_tuning_grid.csv"):
            (out / f).write_bytes((paths["results"]["step3"] / f).read_bytes())
    cache_dir = out / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    bm = json.loads((out / "baseline_metrics.json").read_text(encoding="utf-8"))["chosen_on_tuning_pages"]
    own = {k: bm["own"][k] for k in ("gap_h", "gap_med")}
    finder = bm["own"]["finder"]
    tess_baseline = {k: bm["tesseract_in_own_lines"][k] for k in ("lang", "psm", "input", "scale")}
    grid = [r for r in csv.DictReader(open(out / "baseline_tuning_grid.csv", encoding="utf-8"))
            if r["method"] == "tesseract_in_own_lines"]
    grid.sort(key=lambda r: (-float(r["word_f1"]), -float(r["word_recall"])))
    conv = lambda v: float(v) if v.replace(".", "", 1).isdigit() else v  # noqa: E731
    tess_cands = [{"lang": r["lang"], "psm": int(r["psm"]), "input": r["input"], "scale": conv(r["scale"])}
                  for r in grid[:TOP_TESS]]

    tuning = [Page(p, finder) for p in load_batch2_pages(("TRAIN", "VAL"))]
    if args.dry:
        tuning = tuning[:4] + [pg for pg in tuning if pg.p.split == "VAL"][:4]
    print(f"tuning: {len(tuning)} pages, {sum(len(pg.p.gt_words) for pg in tuning)} word boxes, "
          f"{sum(len(pg.lines) for pg in tuning)} lines")
    cache = cache_dir / "tess_lines_tuning.pkl"
    if cache.exists():
        saved = pickle.loads(cache.read_bytes())
        for pg in tuning:
            pg.tess = saved.get(pg.p.name, {})
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda j: tuning[j[1]].run_tess(tess_cands[j[0]]),
                    [(si, pi) for si in range(len(tess_cands)) for pi in range(len(tuning))]))
    cache.write_bytes(pickle.dumps({pg.p.name: pg.tess for pg in tuning}))

    prm = {"own": own, "tess_baseline": tess_baseline}
    search = []
    # ---------------- lever 1
    best1 = None
    for s in tess_cands:
        tot, _ = evaluate(tuning, "lever1", {**prm, "tess": s})
        search.append({"method": "lever1", **s, **{f"word_{k}": v for k, v in tot.items()}})
        if best1 is None or (tot["f1"], tot["recall"]) > (best1[1]["f1"], best1[1]["recall"]):
            best1 = (s, tot)
    prm["tess"] = best1[0]
    print(f"lever 1 chosen: {best1[0]} tuning word F1 {best1[1]['f1']}")

    # ---------------- lever 2: read every candidate group once
    words_spans = []
    for pg in tuning:   # how many spans a true word covers -> the longest candidate group needed
        for gb in pg.p.gt_words:
            for ln in pg.lines:
                n = sum(1 for s in ln["spans"] if gb[0] <= (s[0] + s[1]) / 2 <= gb[2]
                        and gb[1] <= np.mean([c["cy"] for c in s[2]]) <= gb[3])
                if n:
                    words_spans.append(n)
                    break
    max_spans = int(min(10, np.percentile(words_spans, 99)))
    print(f"spans per true word: median {np.median(words_spans):.0f}, 99th percentile {max_spans}")
    lexicon = {l.split("\t")[0] for l in (REPO_ROOT / "data" / "preprocessed" / "lexicon" / "words.tsv")
               .read_text(encoding="utf-8").splitlines()}
    device = get_device("auto")
    rec = base.load_recognizer(device)
    cache = cache_dir / f"groups_tuning_m{max_spans}.pkl"
    saved = pickle.loads(cache.read_bytes()) if cache.exists() else {}
    for k, pg in enumerate(tuning, 1):
        if pg.p.name in saved:
            pg.groups, readings = saved[pg.p.name]
        else:
            pg.groups = [candidate_groups(ln, max_spans) for ln in pg.lines]
            readings = [read_groups(rec, pg.labels, ln, g, device) for ln, g in zip(pg.lines, pg.groups)]
            saved[pg.p.name] = (pg.groups, readings)
        pg.feats = [word_features(r, lexicon) for r in readings]
        if k % 20 == 0:
            print(f"  read candidate groups: {k}/{len(tuning)} pages")
    cache.write_bytes(pickle.dumps(saved))

    rng = np.random.default_rng(SEED)
    space = {"a": (0.0, 3.0), "b": (0.0, 3.0), "c": (-2.0, 4.0), "d": (0.0, 6.0), "e": (0.0, 2.0)}

    def run(w: dict) -> dict:
        tot, _ = evaluate(tuning, "lever2", {**prm, "w": w})
        search.append({"method": "lever2", **w, **{f"word_{k}": v for k, v in tot.items()}})
        return tot

    def search_weights(fix_e0: bool):
        best = None
        cands = [{k: float(rng.uniform(*r)) for k, r in space.items()} for _ in range(N_RANDOM)]
        for w in cands:
            if fix_e0:
                w["e"] = 0.0
            tot = run(w)
            if best is None or tot["f1"] > best[1]["f1"]:
                best = (w, tot)
        for _ in range(2):   # coordinate refinement
            for k, (lo, hi) in space.items():
                if fix_e0 and k == "e":
                    continue
                for v in np.linspace(lo, hi, 13):
                    w = {**best[0], k: float(v)}
                    tot = run(w)
                    if tot["f1"] > best[1]["f1"]:
                        best = (w, tot)
        return best

    best2 = search_weights(False)
    best2n = search_weights(True)
    print(f"lever 2 chosen: {best2[0]} tuning word F1 {best2[1]['f1']}; without Tesseract {best2n[1]['f1']}")
    keys = list(dict.fromkeys(k for r in search for k in r))
    with open(out / "levers_search.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys, restval="")
        wr.writeheader()
        wr.writerows(search)

    # ---------------- choose on tuning pages
    methods = {"baseline_own": {}, "baseline_tess_lines": {}, "lever1": {}, "lever2": {"w": best2[0]},
               "lever2_no_tess": {"w": best2n[0]}}
    tuning_f1 = {m: evaluate(tuning, m.replace("_no_tess", ""), {**prm, **x})[0] for m, x in methods.items()}
    final = max(tuning_f1, key=lambda m: (tuning_f1[m]["f1"], tuning_f1[m]["recall"]))
    print("tuning word F1:", {m: t["f1"] for m, t in tuning_f1.items()}, "-> final:", final)

    # ---------------- reports
    metrics = {"rules": "chosen on TRAIN+VAL; unseen writer and TESS_TEST measured once, final method only",
               "settings": {"line_finder": finder, "own": own, "tess_baseline": tess_baseline, "lever1_tess": prm["tess"],
                            "lever2_weights": best2[0], "lever2_no_tess_weights": best2n[0], "max_spans": max_spans,
                            "lexicon_words": len(lexicon)},
               "tuning_word_boxes": tuning_f1, "final_method": final, "in_sample": {}}
    groups = collections.defaultdict(list)
    for pg in tuning:
        groups[pg.p.writer].append(pg)
    val = [pg for pg in tuning if pg.p.split == "VAL"]
    for m, x in methods.items():
        mm, px = m.replace("_no_tess", ""), {**prm, **x}
        metrics["in_sample"][m] = {"all_tuning": report(tuning, mm, px, rec, device),
                                   "val_pages_only": report(val, mm, px, rec, device),
                                   "per_writer": {wtr: report(pgs, mm, px, rec, device) for wtr, pgs in sorted(groups.items())}}
    # final method, once, on the tests
    fm, fx = final.replace("_no_tess", ""), {**prm, **methods[final]}
    tests = ({"TEST_UNSEEN_WRITER": load_batch2_pages(("TEST_UNSEEN_WRITER",)), "TESS_TEST": load_tess_test_pages()}
             if not args.dry else {"TEST_UNSEEN_WRITER": load_batch2_pages(("VAL",))[:3], "TESS_TEST": load_batch2_pages(("VAL",))[3:5]})
    per_page = []
    for tname, pgs in tests.items():
        tp = [Page(p, finder) for p in pgs]
        if fm == "lever2":
            for pg in tp:
                pg.groups = [candidate_groups(ln, max_spans) for ln in pg.lines]
                pg.feats = [word_features(read_groups(rec, pg.labels, ln, g, device), lexicon)
                            for ln, g in zip(pg.lines, pg.groups)]
        metrics[tname] = report(tp, fm, fx, rec, device)
        _, evs = evaluate(tp, fm, fx)
        for pg, ev in zip(tp, evs):
            base.draw(pg.p, ev, out / "debug" / "final" / f"{tname}_{pg.p.name}.png")
            per_page.append({"test": tname, "page": pg.p.name, "writer": pg.p.writer,
                             **{f"word_{k}": v for k, v in base.prf(ev["w"]["tp"], ev["w"]["fp"], ev["w"]["fn"]).items()}})
        if tname == "TEST_UNSEEN_WRITER":   # lexicon coverage of the test words (diagnostic, no choice made)
            words = [strip_punct(w) for pg in tp for w in pg.p.words]
            metrics[tname]["lexicon_coverage_of_text_words"] = round(np.mean([w in lexicon for w in words if w]), 3)
    tw = [strip_punct(w) for pg in tuning for w in pg.p.words]
    metrics["tuning_lexicon_coverage_of_text_words"] = round(np.mean([w in lexicon for w in tw if w]), 3)
    (out / "levers_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "levers_per_page.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(per_page[0].keys()))
        wr.writeheader()
        wr.writerows(per_page)
    print(json.dumps({k: metrics[k] for k in ("final_method", "TEST_UNSEEN_WRITER", "TESS_TEST")}, indent=1,
                     ensure_ascii=False))

    if args.dry:
        print("dry run: no gate written")
        return
    # ---------------- gate
    gate_cfg = load_config("step3")["gate"]
    prev = read_gate("step3_tesseract")
    uses_tess = final in ("baseline_tess_lines", "lever1", "lever2")
    gate = write_gate("step3_tesseract", {"unseen_writer_word_box_f1": {
        "value": metrics["TEST_UNSEEN_WRITER"]["word_boxes"]["f1"], "threshold": gate_cfg["min_word_box_f1"], "op": ">="}},
        extra={"stage": "levers 1-2 (no download, no fine-tuning); final method chosen on TRAIN+VAL, test measured once",
               "final_method": final, "uses_tesseract": uses_tess, "iou": gate_cfg["iou"],
               "metrics": {k: metrics[k] for k in ("settings", "tuning_word_boxes", "TEST_UNSEEN_WRITER", "TESS_TEST")},
               "previous_gate": {k: prev[k] for k in ("status", "checks", "timestamp", "git_commit")} if prev else None})
    print(f"[gate] step3_tesseract: {gate['status']} ({final}, word-box F1 "
          f"{gate['checks']['unseen_writer_word_box_f1']['value']} vs {gate_cfg['min_word_box_f1']})")


if __name__ == "__main__":
    main()
