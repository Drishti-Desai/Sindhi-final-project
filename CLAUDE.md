# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Sindhi handwriting OCR: page photo → Unicode Sindhi text. Built in gated steps (Step 1–6, reports in
`docs/STEP1_preprocessing.md` … `docs/STEP6_testing.md`). Full handoff context, results and improvement ideas:
`docs/PROJECT_CONTEXT.md`; data origins: `data/PROVENANCE.md`.

## Owner's rules (always apply)

- Work only inside this folder. Files elsewhere may be **copied** in (record them in `data/PROVENANCE.md`), never
  modified, moved or deleted at their source.
- Use `.venv` only; pin every new package in `requirements.txt`.
- **Ask before** installing software, downloading large data, or deleting files. Show a plan and wait for OK
  before larger changes.
- Never fake or round up results; a result below 0.90 is reported as FAIL with error analysis and options.
  Thresholds are never lowered in code.
- Choose methods on train/validation data only; measure a test set **once**, with the prediction written in the
  docs and committed before the run. FINAL_TEST and the unseen-writer pages are already used up: new claims need
  new held-out pages.
- Commit after each completed step; update the step doc, README and PLAN.md.

## Commands

```bash
.venv/bin/python -m pytest -q                                  # all tests (pytest.ini sets pythonpath=. testpaths=tests)
.venv/bin/python -m pytest tests/test_step5.py -q              # one file
.venv/bin/python -m pytest tests/test_step5.py::test_plausibility_warning -q   # one test

.venv/bin/python run_pipeline.py page.jpg --out pipeline_out/  # images → .txt + .json  (--words joins|gaps, --correct off|auto|suggest)
.venv/bin/python app.py                                        # local Gradio app on http://127.0.0.1:7860
```

Stage runners live in `scripts/` (`stepN_*.py`); each calls `common.gates.require_gate(<previous step>)` and exits
if that gate is missing. There is no linter or build step.

## Architecture

The pipeline (`run_pipeline.run`, shared by the CLI, `app.py` and the evaluation scripts):

1. **Preprocess** — `preprocess/pipeline.py::preprocess_image` (ops in `preprocess/ops.py`, settings in
   `configs/step1_preprocess.yaml`); `pipeline/page.py::clean_page` adds speck/edge filtering.
2. **Layout** — `tesseract_ft/boxes.py::line_structures` (tracked line finder: lines as merged ink spans in
   right-to-left order plus dots) and `pipeline/layout.py::order_lines` (line pieces merged into rows).
3. **Words** — `pipeline/page.py::read_page`: `joins` (default) runs dynamic programming over runs of ≤ 6 spans
   scored by the recognizer's log-probability; `gaps` uses the Step 3 gap rule (`configs/step3_tesseract.yaml`
   `box_finder`).
4. **Recognize** — `cnn_transformer/models.py::WordRecognizer` (ResNet trunk from the letter CNN → Transformer
   encoder/decoder, greedy decoding); weights `checkpoints/step2_recognizer_ft/best.pt`, loaded via
   `scripts/step3_baseline.py::load_recognizer`.
5. **Correct** — `pipeline/page.py::Corrector` wraps `xlmr/`: `correct.py` builds candidates
   (lexicon `data/preprocessed/step4/lexicon.tsv`, weighted edit distance with look-alike letters) and `decide`s;
   `lm.py` scores candidates with the Sindhi-trained XLM-R masked LM and flags words with the detector. The variant
   and weights come from `results/step4/val_comparison.json`.
6. **Output** — text, JSON and `plausibility` warnings (usability only; thresholds set after seeing Step 6).

Correction modes (`pipeline/page.py::correct_mode`): `off`, `auto`, `suggest`. Suggest uses
`xlmr/correct.py::suggest` (what `decide` would change, top-3 candidates) and `apply_choices` / `apply_page_choices`
for the user's picks. What the app displays is built by `pipeline/present.py` (no Gradio inside; tested); `app.py`
only lays it out. The app's opt-in "save my choices" writes to `user_feedback/` (`pipeline/feedback.py`); nothing
may read that folder for training without the owner checking it first.

Tesseract (`tesseract_ft/`) and the Sindhi rules (`xlmr/rules.py`) were evaluated but are not used by the pipeline;
`tesseract_ft/` remains the home of the line finder, page loaders and box scoring.

Supporting modules: `common/paths.py` (`load_paths`, `load_config("stepN")`; every path comes from
`configs/paths.yaml`), `common/gates.py` (gate files in `results/gates/`; a failed gate the owner accepted carries
`user_decision` and passes `require_gate(..., allow_user_accepted=True)`), `common/metrics.py` (CER/WER),
`common/text.py` + `cnn_transformer/words.py::label_norm` (normalisation; `ھ` is mapped to `ه`).

## Constraints and gotchas

- Apple M2 with **8 GB RAM**. Scripts set `PYTORCH_MPS_HIGH_WATERMARK_RATIO` (0.7 for Step 4/pipeline) before
  importing torch; import `run_pipeline` first to get these limits. The two XLM-R models are loaded one at a time
  (`pipeline/page.py::free`). Training uses gradient checkpointing and small micro-batches.
- `data/raw/`, `data/preprocessed/`, `checkpoints/` (~9 GB), `logs/`, `results/*/cache` and `results/*/debug` are
  git-ignored; splits in `data/splits/` and results/gates are tracked.
- Page-level evaluation of batch-2 pages scores only the verified part of each page (`region` masks), because some
  page texts cover only part of the page.
- The recognizer has seen the batch-2 TRAIN writers; tuning anything on its scores for those pages overfits.
- Pages must be upright; pencil and ruled/dotted paper are not supported (Step 6).
