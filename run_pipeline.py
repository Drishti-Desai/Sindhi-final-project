"""Sindhi handwriting OCR: page image(s) -> text.

  Step 1 preprocessing -> lines (our tracked line finder) -> reading order (line pieces joined into rows) ->
  words: recognizer joins of neighbouring ink spans (Step 2 recognizer, no tuned settings; the default) or, with
  --words gaps, the Step 3 gap rule (better for writers who leave clear spaces between words) -> Step 4 correction
  (Sindhi-trained XLM-R + detector). Details: docs/STEP5_pipeline.md.

Pages must be upright. Writes, per image, <out>/<name>.txt (one line of text per found line) and <out>/<name>.json
(line and word boxes, the recognizer's readings, the corrected words and the action taken on each).
Correction (--correct): "auto" (default) changes words; "off" keeps the recognizer's output; "suggest" changes
nothing and writes, in the JSON's "suggestions", every word the corrector would change with its top-3 candidates.
A plausibility warning is printed (and stored in the JSON) when far more lines or words are found than a page of
text can hold: a usability warning only, designed after the Step 6 test (docs/STEP6_testing.md), not a measured
improvement.

Usage: python run_pipeline.py IMAGE [IMAGE ...] [--out DIR] [--words joins|gaps] [--correct off|auto|suggest]
       [--no-correct] [--erase-rules] [--remove-circled]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.paths import REPO_ROOT, load_config  # noqa: E402

_cfg = load_config("step4")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", str(_cfg["mps_memory_ratio"]))
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", str(_cfg["mps_memory_ratio"] * 0.8))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from cnn_transformer.train_utils import get_device  # noqa: E402
from common.gates import require_gate  # noqa: E402


def load_recognizer(device):
    import importlib.util
    spec = importlib.util.spec_from_file_location("step3_baseline", REPO_ROOT / "scripts" / "step3_baseline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_recognizer(device)


def run(binaries: list, device, correct: bool | str = "auto", regions: list | None = None, order=None,
        words: str = "joins", progress=None) -> list[dict]:
    """Cleaned page binaries -> per page {'lines': [...], 'tokens', 'corrected', 'actions', 'text', 'warnings',
    'correct_mode', 'suggestions'}. `correct`: "off" | "auto" | "suggest" (True/False mean auto/off).
    `progress` (optional, the app's status line): called with "read" before the recognizer starts and "correct"
    before the correction; it does not change any result.
    `regions` (evaluation only) restricts each page to a mask (see pipeline.page.read_page); `order` replaces the
    reading-order rule (evaluation only, to measure its effect)."""
    from pipeline.page import (Corrector, add_suggestion_boxes, correct_mode, finish_page, free, plausibility,
                               read_page)
    mode = correct_mode(correct)
    if progress:
        progress("read")
    rec = load_recognizer(device)
    kw = {"words": words} if order is None else {"order": order, "words": words}
    pages, layout = [], []
    for i, b in enumerate(binaries):
        st: dict = {}
        pages.append({"lines": read_page(b, rec, device, None if regions is None else regions[i], stats=st, **kw)})
        layout.append(st)
    del rec
    free(device)
    for p, st in zip(pages, layout):
        p["tokens"] = [w["text"] for ln in p["lines"] for w in ln["words"]]
        p["warnings"] = plausibility(st["line_boxes"], p["tokens"])
        p["correct_mode"], p["words_mode"], p["suggestions"] = mode, words, []
    if mode == "off":
        for p in pages:
            finish_page(p, list(p["tokens"]), ["keep"] * len(p["tokens"]))
        return pages
    if progress:
        progress("correct")
    for p, (out, act, sugg) in zip(pages, Corrector(device).correct([p["tokens"] for p in pages], mode)):
        p["suggestions"] = sugg
        add_suggestion_boxes(p)
        finish_page(p, out, act)
    return pages


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("pipeline_out"))
    ap.add_argument("--words", choices=("joins", "gaps"), default="joins",
                    help="joins (default, chosen on VAL) or gaps: the Step 3 gap rule, for writers with clear spacing")
    ap.add_argument("--correct", choices=("off", "auto", "suggest"), default="auto",
                    help="auto (default): correct words; off: recognizer only; suggest: no changes, candidates in the JSON")
    ap.add_argument("--no-correct", action="store_true", help="same as --correct off")
    ap.add_argument("--erase-rules", action="store_true", help="remove curved ruled lines (notebook pages)")
    ap.add_argument("--remove-circled", action="store_true", help="remove circled question numbers")
    args = ap.parse_args()
    for g in ("step1_preprocessing", "step2_cnn_transformer", "step3_tesseract", "step4_xlmroberta"):
        require_gate(g, allow_user_accepted=True)
    from pipeline.page import clean_page, load_rgb
    device = get_device("auto")
    binaries = [clean_page(load_rgb(p), args.erase_rules, args.remove_circled) for p in args.images]
    pages = run(binaries, device, correct="off" if args.no_correct else args.correct, words=args.words)
    args.out.mkdir(parents=True, exist_ok=True)
    for path, p in zip(args.images, pages):
        (args.out / f"{path.stem}.txt").write_text(p["text"] + "\n", encoding="utf-8")
        (args.out / f"{path.stem}.json").write_text(json.dumps({"image": str(path), **p}, indent=1, ensure_ascii=False),
                                                    encoding="utf-8")
        print(f"{path.name}: {len(p['lines'])} lines, {len(p['tokens'])} words -> {args.out / (path.stem + '.txt')}"
              + (f" ({len(p['suggestions'])} suggestions in the JSON)" if p["correct_mode"] == "suggest" else ""))
        if p["warnings"]:
            print(f"  WARNING ({path.name}): " + "\n  WARNING: ".join(p["warnings"]), file=sys.stderr)


if __name__ == "__main__":
    main()
