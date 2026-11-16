"""Step 3, sub-steps 1-2: zero-shot Tesseract vs our own text-free box finder. Nothing is installed or trained.

1. Tesseract (installed models only) over langs x page-segmentation modes x input (Step 1 binary / gray) x
   scale, scored on the TUNING pages (batch-2 TRAIN + VAL, `tuning_splits` in the config); the setting with
   the best word-box F1 there is chosen. Nothing is trained: only these settings are picked.
2. Our own finder (line finder + word-gap rule, no text given), its settings chosen the same way.
2b. Tesseract inside our lines: each line found by our finder is given to Tesseract alone (--psm 7). This
   isolates Tesseract's word splitting (the part fine-tuning could change) from its layout analysis (which
   fine-tuning does not touch). Language / input / scale chosen on the tuning pages.
3. Both chosen settings are measured ONCE on the tests:
     TEST_UNSEEN_WRITER  7 purple-pen pages, 144 word boxes checked by eye   (the gate)
     TESS_TEST           H1, J2, J3, 56 verified word boxes                  (writers seen in training)
   Reported: word boxes and line boxes (precision / recall / F1 at IoU >= 0.5), split/merge counts,
   Tesseract's own text CER/WER, and the CER/WER when the Step 2 recognizer reads the words inside each
   method's boxes (and inside the ground-truth boxes, for reference).
4. Writes results/gates/step3_tesseract.json (stage: baseline, no fine-tuning). The gated value is the
   Tesseract variant (whole page or inside our lines) with the better word-box F1 on the tuning pages.

Outputs: results/step3/baseline_{metrics.json,tuning_grid.csv,per_page.csv,text.csv}, results/step3/debug/<method>/
Usage: python scripts/step3_baseline.py
"""
from __future__ import annotations

import csv
import itertools
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cnn_transformer.models import WordRecognizer  # noqa: E402
from cnn_transformer.train_utils import get_device, load_step2_config  # noqa: E402
from cnn_transformer.words import BOS, EOS, Vocab, collate, ink_crop, label_norm, to_ink, to_tensor_image  # noqa: E402
from common.gates import require_gate, write_gate  # noqa: E402
from common.metrics import cer, wer  # noqa: E402
from common.paths import REPO_ROOT, load_config, load_paths  # noqa: E402
from tesseract_ft.boxes import (own_layout, prf, resolve_scale, run_tesseract, score_boxes, split_merge,  # noqa: E402
                                tesseract_in_lines, tesseract_layout, tighten)
from tesseract_ft.pages import load_batch2_pages, load_tess_test_pages, union  # noqa: E402

RECOGNIZER = "checkpoints/step2_recognizer_ft/best.pt"


def reading_order(lines: list[list[tuple]]) -> list[list[tuple]]:
    """Lines top to bottom (by mean centre), words right to left."""
    lines = [sorted(ln, key=lambda b: -(b[0] + b[2])) for ln in lines if ln]
    return sorted(lines, key=lambda ln: np.mean([(b[1] + b[3]) / 2 for b in ln]))


def tess_lines(page, s: dict):
    words = run_tesseract(page.binary if s["input"] == "binary" else page.gray, s["lang"], s["psm"],
                          resolve_scale(page.binary, s["scale"]))
    boxes, texts = tesseract_layout(words)
    return boxes, texts


def tess_in_own_lines(page, s_o: dict, s: dict):
    _, line_ids, labels = own_layout(page.binary, **s_o, with_comps=True)
    return tesseract_in_lines(page.binary, page.gray, line_ids, labels, s["lang"], s["input"] == "gray", s["scale"],
                              s["psm"])


def evaluate(page, line_boxes: list[list[tuple]]) -> dict:
    words = [b for ln in line_boxes for b in ln]
    lines = [union(ln) for ln in line_boxes if ln]
    w = score_boxes(words, page.gt_words, page.dc_words, page.binary)
    ln = score_boxes(lines, page.gt_lines, page.dc_lines, page.binary)
    return {"w": w, "l": ln, "sm": split_merge(w["pred"], page.gt_words)}


def total(evals: list[dict], key: str) -> dict:
    return prf(*(sum(e[key][k] for e in evals) for k in ("tp", "fp", "fn")))


def load_recognizer(device):
    st = torch.load(REPO_ROOT / RECOGNIZER, map_location="cpu", weights_only=False)
    cfg = load_step2_config()
    rc = cfg["recognizer"]
    vocab = Vocab("")
    vocab.itos = st["extra"]["vocab"]
    vocab.stoi = {c: i for i, c in enumerate(vocab.itos)}
    m = WordRecognizer(len(vocab), cfg["letters"]["widths"], rc["d_model"], rc["enc_layers"], rc["dec_layers"],
                       rc["heads"], rc["ff"], rc["dropout"], rc["max_label_len"]).to(device)
    m.load_state_dict(st["model"])
    return m.eval(), vocab, rc


@torch.no_grad()
def read_boxes(rec, page, boxes: list[tuple], device) -> list[str]:
    """The Step 2 recognizer reads the ink inside each box (box crop, as the pipeline will do)."""
    model, vocab, rc = rec
    ink = page.binary < 128
    items = []
    for b in boxes:
        x0, y0, x1, y1 = tighten(b, ink)
        crop = page.binary[max(0, y0 - 4):y1 + 4, max(0, x0 - 4):x1 + 4]
        items.append((to_tensor_image(ink_crop(to_ink(crop)), rc["height"], rc["max_width"]), ""))
    out = []
    for i in range(0, len(items), 32):
        x, w, _, _ = collate(items[i:i + 32], vocab)
        seq, _ = model.greedy(x.to(device), w.to(device), BOS, EOS, 26)
        out += [vocab.decode(s) for s in seq.cpu()]
    return out


def draw(page, ev: dict, path: Path) -> None:
    img = cv2.cvtColor(page.binary, cv2.COLOR_GRAY2BGR)
    for b in page.gt_words:
        cv2.rectangle(img, b[:2], (b[2] - 1, b[3] - 1), (0, 170, 0), 1)
    used = {i for i, _ in ev["w"]["matched"]}
    for i, b in enumerate(ev["w"]["pred"]):
        cv2.rectangle(img, (b[0] - 2, b[1] - 2), (b[2] + 1, b[3] + 1), (255, 120, 0) if i in used else (0, 0, 230), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def main() -> None:
    require_gate("step2_cnn_transformer", allow_user_accepted=True)
    cfg, paths = load_config("step3"), load_paths()
    b = cfg["baseline"]
    out = paths["results"]["step3"]
    out.mkdir(parents=True, exist_ok=True)
    val = load_batch2_pages(tuple(b["tuning_splits"]))   # tuning pages
    tests = {"TEST_UNSEEN_WRITER": load_batch2_pages(("TEST_UNSEEN_WRITER",)), "TESS_TEST": load_tess_test_pages()}
    print(f"tuning {len(val)} pages / {sum(len(p.gt_words) for p in val)} word boxes; "
          + "; ".join(f"{k} {len(v)} pages / {sum(len(p.gt_words) for p in v)} word boxes" for k, v in tests.items()))

    # ---------------- 1. Tesseract grid on the tuning pages
    settings = [dict(lang=l, psm=m, input=i, scale=s) for l, m, i, s in
                itertools.product(b["langs"], b["psms"], b["inputs"], b["scales"])]
    jobs = [(si, pi) for si in range(len(settings)) for pi in range(len(val))]
    with ThreadPoolExecutor(b["workers"]) as ex:
        res = list(ex.map(lambda j: tess_lines(val[j[1]], settings[j[0]])[0], jobs))
    grid = []
    for si, s in enumerate(settings):
        evs = [evaluate(val[pi], res[k]) for k, (sj, pi) in enumerate(jobs) if sj == si]
        grid.append({"method": "tesseract", **s, **{f"word_{k}": v for k, v in total(evs, "w").items()},
                     **{f"line_{k}": v for k, v in total(evs, "l").items() if k in ("precision", "recall", "f1")}})
    # ---------------- 2. own finder grid on the tuning pages
    for f, gh, gm in itertools.product(b["own_finders"], b["own_gap_h"], b["own_gap_med"]):
        evs = [evaluate(p, own_layout(p.binary, f, gh, gm)) for p in val]
        grid.append({"method": "own", "finder": f, "gap_h": gh, "gap_med": gm,
                     **{f"word_{k}": v for k, v in total(evs, "w").items()},
                     **{f"line_{k}": v for k, v in total(evs, "l").items() if k in ("precision", "recall", "f1")}})
    pick = lambda m: max((g for g in grid if g["method"] == m), key=lambda g: (g["word_f1"], g["word_recall"]))  # noqa: E731
    best_t, best_o = pick("tesseract"), pick("own")
    s_t = {k: best_t[k] for k in ("lang", "psm", "input", "scale")}
    s_o = {k: best_o[k] for k in ("finder", "gap_h", "gap_med")}
    print(f"chosen on tuning pages: tesseract {s_t} word F1 {best_t['word_f1']}; own {s_o} word F1 {best_o['word_f1']}")
    # ---------------- 2b. Tesseract inside our own lines, on the tuning pages
    settings_l = [dict(lang=l, psm=m, input=i, scale=s) for l, m, i, s in
                  itertools.product(b["langs"], b["line_psms"], b["inputs"], b["scales"])]
    jobs = [(si, pi) for si in range(len(settings_l)) for pi in range(len(val))]
    with ThreadPoolExecutor(b["workers"]) as ex:
        res = list(ex.map(lambda j: tess_in_own_lines(val[j[1]], s_o, settings_l[j[0]])[0], jobs))
    for si, s in enumerate(settings_l):
        evs = [evaluate(val[pi], res[k]) for k, (sj, pi) in enumerate(jobs) if sj == si]
        grid.append({"method": "tesseract_in_own_lines", **s, **{f"word_{k}": v for k, v in total(evs, "w").items()},
                     **{f"line_{k}": v for k, v in total(evs, "l").items() if k in ("precision", "recall", "f1")}})
    best_l = pick("tesseract_in_own_lines")
    s_l = {k: best_l[k] for k in ("lang", "psm", "input", "scale")}
    print(f"chosen on tuning pages: tesseract inside own lines {s_l} word F1 {best_l['word_f1']}")
    keys = list(dict.fromkeys(k for g in grid for k in g))
    with open(out / "baseline_tuning_grid.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(grid)
    gated = max((("tesseract", best_t), ("tesseract_in_own_lines", best_l)), key=lambda x: x[1]["word_f1"])[0]
    for lang in b["langs"]:  # best per language, for the report
        g = max((g for g in grid if g.get("lang") == lang and g["method"] == "tesseract"), key=lambda g: (g["word_f1"], g["word_recall"]))
        print(f"  tuning best {lang:14s} psm {g['psm']:2d} {g['input']:6s} {g['scale']}: word F1 {g['word_f1']:.3f} "
              f"(P {g['word_precision']:.3f} R {g['word_recall']:.3f}), line F1 {g['line_f1']:.3f}")

    # ---------------- 3. tests, measured once
    device = get_device("auto")
    rec = load_recognizer(device)
    metrics, per_page, text_rows = {"chosen_on_tuning_pages": {
        "tuning_splits": b["tuning_splits"], "pages": len(val), "word_boxes": sum(len(p.gt_words) for p in val),
        "tesseract": {**s_t, "tuning_word_f1": best_t["word_f1"]}, "own": {**s_o, "tuning_word_f1": best_o["word_f1"]},
        "tesseract_in_own_lines": {**s_l, "tuning_word_f1": best_l["word_f1"]}, "gated_tesseract_variant": gated}}, [], []
    for tname, pages in tests.items():
        m = {}
        refs = [" ".join(p.words) for p in pages]
        for method in ("tesseract", "tesseract_in_own_lines", "own", "ground_truth_boxes"):
            evs, hyps_rec, hyps_tess = [], [], []
            for p in pages:
                if method.startswith("tesseract"):
                    boxes, texts = tess_lines(p, s_t) if method == "tesseract" else tess_in_own_lines(p, s_o, s_l)
                    order = sorted(range(len(boxes)), key=lambda i: np.mean([(x[1] + x[3]) / 2 for x in boxes[i]]))
                    hyps_tess.append(" ".join(t for i in order for t in (label_norm(x) for x in texts[i]) if t))
                    lines = reading_order(boxes)
                elif method == "own":
                    lines = reading_order(own_layout(p.binary, **s_o))
                else:
                    lines = [p.all_words]   # already in reading order
                if method != "ground_truth_boxes":
                    ev = evaluate(p, lines)
                    evs.append(ev)
                    draw(p, ev, out / "debug" / method / f"{tname}_{p.name}.png")
                    per_page.append({"test": tname, "page": p.name, "method": method,
                                     **{f"word_{k}": v for k, v in prf(ev["w"]["tp"], ev["w"]["fp"], ev["w"]["fn"]).items()},
                                     "ignored": ev["w"]["ignored"], **ev["sm"]})
                flat = [x for ln in lines for x in ln if tighten(x, p.binary < 128) is not None]
                hyps_rec.append(" ".join(t for t in read_boxes(rec, p, flat, device) if t))
            entry = {}
            if evs:
                entry["word_boxes"] = total(evs, "w")
                entry["line_boxes"] = total(evs, "l")
                entry["ignored_boxes"] = sum(e["w"]["ignored"] for e in evs)
                entry["gt_words_split"] = sum(e["sm"]["gt_words_split"] for e in evs)
                entry["pred_boxes_merging_2plus"] = sum(e["sm"]["pred_boxes_merging_2plus"] for e in evs)
                entry["gt_words_not_found"] = sum(e["sm"]["gt_words_not_found"] for e in evs)
            entry["recognizer_on_these_boxes"] = {"cer": round(cer(refs, hyps_rec), 4), "wer": round(wer(refs, hyps_rec), 4)}
            if hyps_tess:
                entry["tesseract_text"] = {"cer": round(cer(refs, hyps_tess), 4), "wer": round(wer(refs, hyps_tess), 4)}
            for p, r, h in zip(pages, refs, hyps_rec):
                text_rows.append({"test": tname, "page": p.name, "method": method, "reader": "step2_recognizer",
                                  "ref": r, "hyp": h})
            for p, r, h in zip(pages, refs, hyps_tess):
                text_rows.append({"test": tname, "page": p.name, "method": method, "reader": "tesseract",
                                  "ref": r, "hyp": h})
            m[method] = entry
        metrics[tname] = m
    (out / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, rows in (("baseline_per_page.csv", per_page), ("baseline_text.csv", text_rows)):
        with open(out / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(json.dumps({k: v for k, v in metrics.items() if k != "chosen_on_tuning_pages"}, indent=1, ensure_ascii=False))

    # ---------------- 4. gate (Tesseract's word boxes on the unseen writer)
    g = cfg["gate"]
    gate = write_gate("step3_tesseract", {"unseen_writer_word_box_f1": {
        "value": metrics["TEST_UNSEEN_WRITER"][gated]["word_boxes"]["f1"], "threshold": g["min_word_box_f1"],
        "op": ">="}}, extra={"stage": "baseline: installed Tesseract models, no fine-tuning", "iou": g["iou"],
                             "gated_variant": gated, "tesseract_setting": s_t if gated == "tesseract" else
                             {"own_lines": s_o, **s_l}, "metrics": metrics,
                             "run1_val_only": "results/step3/run1_val_only/ (settings chosen on VAL alone; superseded)",
                             "note": "own finder reported for comparison; it is not Tesseract and does not count for this gate"})
    print(f"[gate] step3_tesseract: {gate['status']} (word-box F1 {gate['checks']['unseen_writer_word_box_f1']['value']} "
          f"vs {g['min_word_box_f1']})")


if __name__ == "__main__":
    main()
