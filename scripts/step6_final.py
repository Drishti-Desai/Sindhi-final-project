"""Step 6: the final test. run_pipeline.py defaults on the FINAL_TEST pages, whole page, run once.

Each raw image goes through exactly what the command line does (pipeline.page.load_rgb -> clean_page with no
options -> run_pipeline.run with --words joins and the Step 4 correction). Reference: the page's text file,
normalised as for TESS_TEST (tesseract_ft/pages.py). The 3 TESS_TEST pages go through the same procedure as a
reference (whole page; Steps 3-5 scored only their verified part). Protocol: docs/STEP6_testing.md.

  python scripts/step6_final.py --dry   TESS_TEST reference pages only (checks the script; FINAL_TEST not read)
  python scripts/step6_final.py         the single final measurement (refuses to run twice)

Outputs: results/step6/final_metrics.json, results/step6/pages/<page>.txt|.json, results/gates/step6_final.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_pipeline  # noqa: E402  (sets the MPS memory limits)
from cnn_transformer.train_utils import get_device  # noqa: E402
from common.gates import read_gate, require_gate, write_gate  # noqa: E402
from common.metrics import cer  # noqa: E402
from common.paths import REPO_ROOT, load_paths  # noqa: E402
from common.text import normalize  # noqa: E402
from pipeline.page import clean_page, load_rgb  # noqa: E402
from tesseract_ft.pages import label_norm_words  # noqa: E402
from xlmr.correct import evaluate  # noqa: E402

GATE_MIN = 0.90
PREDICTED = "CER 0.28-0.40 (1 - CER 0.60-0.72), FAIL (docs/STEP6_testing.md, recorded before the run)"
N_BOOT = 2000


def load_set(split: str) -> list[dict]:
    paths = load_paths()
    rows = [r for r in csv.DictReader(open(paths["splits"]["handwritten"] / "pairs.csv", encoding="utf-8"))
            if r["split"] == split]
    out = []
    for r in rows:
        lines = [l for l in normalize((paths["raw"]["handwritten_text_gt"] / r["text_gt"]).read_text(encoding="utf-8"),
                                      diacritics=False).splitlines() if l.strip()]
        out.append({"page": r["image"][:-4], "writer": r["writer"], "image": paths["raw"]["handwritten_images"] / r["image"],
                    "ref": label_norm_words(" ".join(lines)), "ref_lines": len(lines)})
    return out


def scores(pages: list[dict], outs: list[dict]) -> dict:
    items = [{"ref": p["ref"], "input": o["tokens"]} for p, o in zip(pages, outs)]
    m = evaluate(items, [o["corrected"] for o in outs], [o["actions"] for o in outs])
    m["one_minus_cer"] = round(1 - m["cer"], 4)
    m["pages"], m["words"] = len(pages), sum(len(p["ref"]) for p in pages)
    m["words_read"] = sum(len(o["tokens"]) for o in outs)
    m["lines_reference"], m["lines_found"] = sum(p["ref_lines"] for p in pages), sum(len(o["lines"]) for o in outs)
    return m


def bootstrap(pages: list[dict], outs: list[dict], seed: int = 0) -> list[float]:
    """95% interval of the pooled CER after correction, resampling pages."""
    rng = np.random.default_rng(seed)
    refs = [" ".join(p["ref"]) for p in pages]
    hyps = [" ".join(o["corrected"]) for o in outs]
    vals = []
    for _ in range(N_BOOT):
        k = rng.integers(0, len(pages), len(pages))
        vals.append(cer([refs[i] for i in k], [hyps[i] for i in k]))
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def measure(pages: list[dict], device, out_dir: Path) -> dict:
    binaries = [clean_page(load_rgb(p["image"])) for p in pages]
    outs = run_pipeline.run(binaries, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p, o in zip(pages, outs):
        (out_dir / f"{p['page']}.txt").write_text(o["text"] + "\n", encoding="utf-8")
        (out_dir / f"{p['page']}.json").write_text(json.dumps({"image": str(p["image"].relative_to(REPO_ROOT)), **o},
                                                              indent=1, ensure_ascii=False), encoding="utf-8")
    m = scores(pages, outs)
    m["cer_95ci_over_pages"] = bootstrap(pages, outs)
    m["per_writer"] = {w: scores([p for p in pages if p["writer"] == w], [o for p, o in zip(pages, outs) if p["writer"] == w])
                       for w in sorted({p["writer"] for p in pages})}
    m["per_page"] = {p["page"]: {k: v for k, v in scores([p], [o]).items() if k != "pages"} for p, o in zip(pages, outs)}
    m["readings"] = [{"page": p["page"], "ref": " ".join(p["ref"]), "read": " ".join(o["tokens"]),
                      "corrected": " ".join(o["corrected"])} for p, o in zip(pages, outs)]
    return m


def line(name: str, r: dict) -> str:
    return (f"{name:12s} CER {r['cer_before']:.3f} -> {r['cer']:.3f}  words {r['word_accuracy_before']:.3f} -> "
            f"{r['word_accuracy']:.3f}  changed {r['clean_changed']}/{r['clean_tokens']}  "
            f"words read {r['words_read']}/{r['words']}  lines {r['lines_found']}/{r['lines_reference']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    for g in ("step1_preprocessing", "step2_cnn_transformer", "step3_tesseract", "step4_xlmroberta", "step5_pipeline"):
        require_gate(g, allow_user_accepted=True)
    device = get_device("auto")
    res = REPO_ROOT / "results" / "step6"
    if args.dry:
        m = measure(load_set("TESS_TEST"), device, res / "dry")
        print(line("TESS_TEST", m))
        return
    if read_gate("step6_final") is not None:
        sys.exit("[step6] the final measurement has already been made (results/gates/step6_final.json)")
    final = measure(load_set("FINAL_TEST"), device, res / "pages")
    ref = measure(load_set("TESS_TEST"), device, res / "pages_tess_test_reference")
    out = {"rules": "run_pipeline.py defaults (--words joins, correction on), whole page, measured once",
           "predicted_before_run": PREDICTED, "FINAL_TEST": final, "reference_TESS_TEST_whole_page": ref}
    (res / "final_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    gate = write_gate("step6_final", {"final_test_one_minus_cer": {"value": final["one_minus_cer"], "threshold": GATE_MIN,
                                                                   "op": ">="}},
                      extra={"test_set": "FINAL_TEST, 7 pages, whole page", "predicted_before_run": PREDICTED,
                             "metrics": {k: v for k, v in final.items() if k not in ("readings", "per_page")},
                             "reference_TESS_TEST_whole_page": {k: v for k, v in ref.items()
                                                                if k not in ("readings", "per_page", "per_writer")}})
    print(line("FINAL_TEST", final))
    for w, r in final["per_writer"].items():
        print(line(f"  writer {w}", r))
    for p, r in final["per_page"].items():
        print(line(f"  {p}", r))
    print(line("TESS_TEST", ref))
    print(f"[gate] step6_final: {gate['status']} (1 - CER {final['one_minus_cer']}, 95% CI of CER "
          f"{final['cer_95ci_over_pages']})")


if __name__ == "__main__":
    main()
