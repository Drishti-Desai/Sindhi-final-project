"""Step 5 evaluation of run_pipeline.py on pages (the verified part of each page, as in Steps 3-4).

  python scripts/step5_eval.py            VAL pages (development; every choice is made here; also --words gaps)
  python scripts/step5_eval.py --final    the single measurement: unseen writer's pages (gate) and H/J TESS_TEST
                                          pages (reported). Refuses to run twice. FINAL_TEST is never read.

The pages are the same cleaned binaries as in Steps 3-4 (tesseract_ft/pages.py: upright, Step 1, circled numbers
and rulings removed) and go through run_pipeline.run, the function the command line uses. Metrics per set (as in
Step 4, xlmr.correct.evaluate): CER and word accuracy (1 - WER) before and after the correction, and how many
correct words the correction changed. Step 4's numbers for the same pages (gap-rule boxes) are shown alongside.

Outputs: results/step5/val_metrics.json, results/step5/test_metrics.json (+ readings), results/gates/step5_pipeline.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_pipeline  # noqa: E402  (sets the MPS memory limits)
from cnn_transformer.train_utils import get_device  # noqa: E402
from cnn_transformer.words import label_norm  # noqa: E402
from common.gates import read_gate, require_gate, write_gate  # noqa: E402
from common.metrics import cer  # noqa: E402
from common.paths import REPO_ROOT  # noqa: E402
from pipeline.layout import order_by_centre  # noqa: E402
from tesseract_ft.pages import load_batch2_pages, load_tess_test_pages  # noqa: E402
from xlmr.correct import evaluate  # noqa: E402

GATE_MIN = 0.90
PREDICTED = "CER 0.12-0.15 (1 - CER 0.85-0.88), likely FAIL (docs/STEP5_pipeline.md, recorded before the run)"


def region(p) -> np.ndarray:
    m = np.zeros(p.binary.shape, bool)
    for x0, y0, x1, y1 in p.gt_words:
        m[y0:y1, x0:x1] = True
    return m


def measure(pages, device, order=None, words: str = "joins") -> tuple[dict, list[dict]]:
    pages = [p for p in pages if p.gt_words]
    out = run_pipeline.run([p.binary for p in pages], device, regions=[region(p) for p in pages], order=order,
                           words=words)
    items = [{"ref": [w for l in p.gt_labels for w in label_norm(l).split()], "input": o["tokens"]}
             for p, o in zip(pages, out)]
    m = evaluate(items, [o["corrected"] for o in out], [o["actions"] for o in out])
    m["one_minus_cer"] = round(1 - m["cer"], 4)
    m["pages"], m["words"] = len(pages), sum(len(it["ref"]) for it in items)
    m["per_page_cer"] = {p.name: {"before": round(cer([" ".join(it["ref"])], [" ".join(it["input"])]), 3),
                                  "after": round(cer([" ".join(it["ref"])], [" ".join(o["corrected"])]), 3)}
                         for p, it, o in zip(pages, items, out)}
    readings = [{"page": p.name, "ref": " ".join(it["ref"]), "read": " ".join(it["input"]),
                 "corrected": " ".join(o["corrected"])} for p, it, o in zip(pages, items, out)]
    return m, readings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    for g in ("step2_cnn_transformer", "step3_tesseract", "step4_xlmroberta"):
        require_gate(g, allow_user_accepted=True)
    device = get_device("auto")
    res = REPO_ROOT / "results" / "step5"
    res.mkdir(parents=True, exist_ok=True)
    s4_val = json.loads((REPO_ROOT / "results" / "step4" / "val_comparison.json").read_text(encoding="utf-8"))
    s4_test = json.loads((REPO_ROOT / "results" / "step4" / "test_metrics.json").read_text(encoding="utf-8"))
    final_v = s4_val["final_variant"]

    if not args.final:
        val = load_batch2_pages(("VAL",))
        m, readings = measure(val, device)
        m_old_order, _ = measure(val, device, order=order_by_centre)
        m_gaps, _ = measure(val, device, words="gaps")
        out = {"set": "batch-2 VAL pages (verified part)", "pipeline": m,
               "ablation_old_reading_order": m_old_order, "option_words_gaps": m_gaps,
               "step4_gap_rule_pages": s4_val["validation"][final_v]["real_pages"], "readings": readings}
        (res / "val_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        for k in ("step4_gap_rule_pages", "option_words_gaps", "ablation_old_reading_order", "pipeline"):
            r = out[k]
            print(f"{k:28s} CER {r['cer_before']:.3f} -> {r['cer']:.3f}  words {r['word_accuracy_before']:.3f} -> "
                  f"{r['word_accuracy']:.3f}  correct words changed {r['clean_changed']}/{r['clean_tokens']}")
        return

    if read_gate("step5_pipeline") is not None:
        sys.exit("[step5] the final measurement has already been made (results/gates/step5_pipeline.json)")
    sets = {"unseen_writer_pages": load_batch2_pages(("TEST_UNSEEN_WRITER",)), "hj_pages": load_tess_test_pages()}
    table, readings = {}, {}
    for name, pages in sets.items():
        table[name], readings[name] = measure(pages, device)
    step4 = {"unseen_writer_pages": s4_test["test"][final_v]["unseen_pages"],
             "hj_pages": s4_test["test"][final_v]["hj_pages"]}
    out = {"rules": "run_pipeline.py as chosen on VAL; measured once", "predicted_before_run": PREDICTED,
           "test": table, "step4_gap_rule_pages": step4, "readings": readings}
    (res / "test_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    u = table["unseen_writer_pages"]
    gate = write_gate("step5_pipeline", {"unseen_writer_pages_one_minus_cer": {
        "value": u["one_minus_cer"], "threshold": GATE_MIN, "op": ">="}},
        extra={"test_set": "TEST_UNSEEN_WRITER pages, verified part (144 words)", "predicted_before_run": PREDICTED,
               "metrics": {k: {kk: vv for kk, vv in v.items() if kk != "per_page_cer"} for k, v in table.items()},
               "step4_gap_rule_pages": step4})
    for name in table:
        for label, r in (("step4 gap rule", step4[name]), ("step5 pipeline", table[name])):
            print(f"{name:20s} {label:15s} CER {r['cer_before']:.3f} -> {r['cer']:.3f}  words "
                  f"{r['word_accuracy_before']:.3f} -> {r['word_accuracy']:.3f}  changed {r['clean_changed']}/{r['clean_tokens']}")
    print(f"[gate] step5_pipeline: {gate['status']} (1 - CER {u['one_minus_cer']} vs {GATE_MIN})")


if __name__ == "__main__":
    main()
