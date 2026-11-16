# 01 — Complete project inventory

Audited 2026-11-16 at commit `fbe4a5b` (branch `main`, 35 commits, first `6b08051` on 2026-10-17, working tree
clean, **no git remote**). The full machine-generated inventory (every folder, file, size, checkpoint, log and
import) is in `appendix/A_raw_file_inventory.md`; this file explains it.

---

## 1. Folders at a glance

| Folder | Size | In git? | What it holds |
|---|---|---|---|
| `app.py`, `run_pipeline.py` | — | yes | the two **entry points**: web app and command line |
| `pipeline/` | 96K | yes | page reading (lines, reading order, word joins), correction wrapper, warnings, display helpers, saved choices |
| `preprocess/` | 136K | yes | Step 1 image cleaning (`preprocess_image`) |
| `cnn_transformer/` | 192K | yes | Step 2 models (`LetterCNN`, `WordRecognizer`), datasets, segmentation helpers |
| `tesseract_ft/` | 104K | yes | Step 3: line finder, box finders (own + Tesseract), box scoring, evaluation pages, leakage filter |
| `xlmr/` | 124K | yes | Step 4: correction candidates, XLM-R scoring, detector, error model, text splits |
| `common/` | 52K | yes | paths, gates, CER/WER, Sindhi text normalisation |
| `scripts/` | 512K | yes (26 .py) | one runner per stage (prepare data, train, evaluate) |
| `tests/` | 288K | yes (9 .py) | 79 unit tests |
| `configs/` | 32K | yes | `paths.yaml` (all paths), `letter_map.yaml` (52 classes), step settings |
| `docs/` | 96K | yes | step reports `STEP1…STEP6`, `PROJECT_CONTEXT.md` |
| `data/` | 622M (76,345 files) | partly (splits, text_gt, PROVENANCE) | raw copies, preprocessed images, synthetic renders, fixed splits |
| `results/` | 95M | mostly (113 files; caches/debug ignored) | gate files, metrics, predictions, figures |
| `checkpoints/` | 8.8G | **no** | trained models |
| `logs/` | 512K | no | training/evaluation logs |
| `.venv/` | 1.7G | no | Python environment |
| `PROJECT_HANDOVER/` | — | not yet committed | this handover |

Not present: `pipeline_out/`, `user_feedback/`, `outputs/`, `.env` (all git-ignored; created on use).
No project-written notebooks exist. No standalone HTML/CSS/JS files exist in the project's code; the app's CSS is a
string in `app.py`; the only HTML file is a generated review page `results/step2/debug/batch2_review/index.html`.

---

## 2. Entry points and execution flow

| Entry point | Command | Flow |
|---|---|---|
| **Web app** | `.venv/bin/python app.py` | `app.py::main` → gates → `build()` (Gradio UI) → on "Read page": `process()` → `pipeline/page.py::clean_page` → `run_pipeline.run` → results screen; "Apply choices": `apply()` |
| **Command line** | `.venv/bin/python run_pipeline.py IMG… --out DIR` | `run_pipeline.py::main` → gates → `clean_page(load_rgb())` → `run()` → writes `.txt` + `.json` |
| Training / evaluation | `.venv/bin/python scripts/<stage>.py` | each script calls `require_gate(<previous step>)` first |
| Tests | `.venv/bin/python -m pytest -q` | `pytest.ini`: `pythonpath = .`, `testpaths = tests` |

`run()` itself: `load_recognizer()` (via `importlib` from `scripts/step3_baseline.py`, so that script is part of the
runtime) → `read_page()` per page → `plausibility()` → `Corrector.correct()` (XLM-R models loaded one at a time) →
`finish_page()`. Full diagrams: `02_SYSTEM_ARCHITECTURE.md`.

---

## 3. Important files

Status legend: **Working** = used by the app/CLI and covered by tests or a measured run; **Training** = used to
build models/data (ran successfully, results saved); **Experiment** = one-off analysis kept as evidence;
**Not used** = present but not used by anything.

### Runtime (needed to run the app)

| File | Purpose | What it does | Input | Output | Status |
|---|---|---|---|---|---|
| `app.py` | Web app | Two screens (Set up → Results), progress, Suggest review cards, downloads; binds to 127.0.0.1 | photo, options | web page | Working (tests `test_app.py`; checked in browser) |
| `run_pipeline.py` | CLI + shared `run()` | Loads recognizer, reads pages, corrects, writes files | image paths, `--words`, `--correct` | `<name>.txt`, `<name>.json` | Working |
| `pipeline/page.py` | Page reading | `clean_page`, `read_page`, `line_units`, `best_grouping`, `plausibility`, `Corrector`, `apply_page_choices`, `word_crop` | binary page | lines/words/readings | Working (`test_step5.py`, `test_suggest.py`) |
| `pipeline/layout.py` | Reading order | `order_lines` groups line pieces into rows | line boxes | order | Working (`test_step5.py`) |
| `pipeline/present.py` | What the app shows | stepper, image checks, boxes picture, comparison, summary, warning banner, review choices | page dict | HTML/images | Working (`test_present.py`) |
| `pipeline/feedback.py` | Opt-in saved choices | writes crops + `choices.jsonl` to `user_feedback/` | page, choices | files | Working (`test_suggest.py`); never read by any script |
| `preprocess/pipeline.py`, `preprocess/ops.py` | Step 1 cleaning | measure image → flatten background → Sauvola → deskew → remove lines/borders/specks | RGB image | binary + gray | Working (26 tests in `test_preprocess.py`) |
| `tesseract_ft/boxes.py` | Lines + boxes | `line_structures` (tracked line finder), `group_box`, `groups_from_cuts`, `own_layout`, box scoring | binary page | lines/spans | Working (`test_step3.py`) |
| `tesseract_ft/cuts.py` | Reading word groups | `read_groups` runs the recognizer on each group's ink | groups | (text, log-prob) | Working |
| `tesseract_ft/pages.py` | Helpers + evaluation pages | `filter_comps`, `keep_only` (runtime); `load_batch2_pages`, `load_tess_test_pages` (evaluation) | — | — | Working |
| `cnn_transformer/models.py` | Models | `LetterCNN`, `WordRecognizer` (`encode`, `decode`, `greedy`), `load_trunk_from_letter_cnn` | images | logits/text | Working (`test_step2.py`) |
| `cnn_transformer/words.py` | Recognizer data | `Vocab`, `label_norm`, image preparation (`to_tensor_image`, `collate`), `MixedWords` | crops | tensors | Working |
| `cnn_transformer/segment.py`, `align.py` | Components, line clustering, ruled-line erase, forced alignment | used by the line finder and data extraction | — | — | Working (`test_segment.py`) |
| `scripts/step3_baseline.py` | contains `load_recognizer()` + `RECOGNIZER` path used at runtime | also the Step 3 baseline experiment | — | — | Working |
| `xlmr/correct.py` | Correction logic | `Lexicon`, `build_positions`, `decide`, `suggest`, `apply_choices` | tokens | corrected tokens | Working (`test_step4.py`, `test_suggest.py`) |
| `xlmr/lm.py` | XLM-R scoring | `MaskedScorer.score`, `Detector.probs` | tokens, candidates | LM scores, flags | Working |
| `xlmr/errors.py` | Recognizer error model | `ErrorModel` (also used for synthetic corruption) | — | — | Training + runtime (via `correct.py`) |
| `common/gates.py`, `paths.py`, `metrics.py`, `text.py` | Shared helpers | gates, config/paths, CER/WER, normalisation | — | — | Working |
| `configs/*.yaml` | Settings | all paths and parameters | — | — | Working |
| Runtime data | `results/gates/*.json`, `results/step4/val_comparison.json`, `results/step4/error_model.json`, `data/preprocessed/step4/lexicon.tsv` (**not in git**) | read at start | — | — | Required |
| Runtime models | `checkpoints/step2_recognizer_ft/best.pt` (31 MB), `checkpoints/step4_xlmr_mlm/best/` (1.1 GB), `checkpoints/step4_xlmr_detector/best/` (1.0 GB) | **not in git** | — | — | Required |

### Training and data preparation

| File | Purpose | Status |
|---|---|---|
| `scripts/prepare_splits.py` | de-duplicated letter manifest, sheet-grouped splits, handwritten page splits | Training (ran) |
| `scripts/prepare_batch2.py` | batch-2 manifest and page splits | Training (ran) |
| `scripts/step1_compare_methods.py` | Step 1 benchmark (required before `step1_preprocess.py`) | Training (ran) |
| `scripts/step1_preprocess.py` | preprocess all datasets + Step 1 gate | Training (ran) |
| `scripts/step2_make_gray_letters.py` | grayscale letter crops used by the chosen CNN | Training (ran) |
| `scripts/step2_train_cnn.py` | 52-letter CNN runs, evaluation, gate | Training (ran) |
| `scripts/step2_make_synthetic.py` | word lists + printed-font word renders | Training (ran) |
| `scripts/step2_extract_real_words.py`, `step2_extract_batch2_words.py` | real word crops from pages (+ review page) | Training (ran) |
| `scripts/step2_train_recognizer.py` | word recognizer training / fine-tuning, test, gate | Training (ran) |
| `scripts/build_lexicon.py` | word list for Step 3 lever 2 | Training (ran) |
| `scripts/step3_levers.py` | Step 3 levers 1–2 | Experiment (ran) |
| `scripts/step4_prepare.py`, `step4_train_mlm.py`, `step4_train_detector.py`, `step4_compare.py`, `step4_test.py` | Step 4 data, training, validation, one-time test | Training (ran) |
| `scripts/step5_order.py`, `step5_eval.py` | reading-order choice, pipeline evaluation | Training/evaluation (ran) |
| `scripts/step6_final.py` | final one-time test | Evaluation (ran once; refuses to re-run) |
| `xlmr/data.py`, `tesseract_ft/texts.py` | text splits, leakage filter | Training (ran) |
| `cnn_transformer/letters.py`, `train_utils.py` | letter dataset, training plumbing | Training |
| `preprocess/synthetic.py` | synthetic benchmark pages | Training/tests |

### Experimental / one-off (kept as evidence)

| File | Note |
|---|---|
| `scripts/step2_ensemble_eval.py`, `scripts/step2_error_analysis.py` | letter-CNN ensemble/TTA and error analysis; not referenced elsewhere |
| `scripts/inspect_pos_data.py` | inspection of the POS datasets |
| `scripts/step4_page_diagnosis.py` | page error diagnosis (imported by `step5_order.py`, so keep) |
| `scripts/step5_probe_joins.py` | Step 5 planning probe |
| `preprocess/segmentation.py` (+ `checkpoints/rembg/`) | rembg comparison in Step 1; not used by the pipeline |
| `xlmr/rules.py` | Sindhi hard rules; tested in Step 4 and **not kept** |
| `tesseract_ft/cuts.py::best_cuts`, `snap_cuts` | Step 3 levers; kept as an option, not used by the pipeline |

---

## 4. Datasets, models, outputs (summary; details in 03 and 04)

- Datasets (`data/raw/`, 259 MB apparent): letters52, handwritten (22 pages), handwritten_batch2 (129 images),
  handwritten_batch3 (not used), pos_handwritten (998 images, no transcriptions, not used), pos_text (third-party
  POS corpus + 2 notebooks + CRF model, not used by code), roboflow_sindhi_ocrf (printed pages, not used by the
  pipeline), corpus/wikipedia_sd. Verified counts: `04_DATASET_AUDIT_AND_PROVENANCE.md`.
- Models (`checkpoints/`, 8.8 GB): runtime 3 items (~2.1 GB); training inputs (`step2_cnn/best.pt`,
  `step2_recognizer/best.pt`, `xlmr_base/`); experiment runs (`step2_cnn_runs/`); intermediate Hugging Face
  checkpoints (`checkpoint-*`, ~5 GB, only useful to resume training); `last.pt` resume files (~380 MB).
- Outputs (`results/`): gate files; Step 1 benchmark tables and figures; Step 2 metrics, per-class accuracy,
  confusions, predictions; Step 3 box metrics; Step 4 validation/test metrics and error model; Step 5 metrics;
  Step 6 final metrics and the page outputs.

## 5. Dependency files

`requirements.txt` (pinned; Python 3.13.3, Apple M2), `pytest.ini`, `.gitignore`, `.env.example` (two empty
placeholders, `RF_API_KEY` for Roboflow and `HF_TOKEN` for Hugging Face; no `.env` exists and no code needs a key
to run — the models are local and `HF_HUB_OFFLINE=1` is set). Not in `requirements.txt`: SciPy
(installed via scikit-image), the Tesseract program (only for Step 3 experiments), `pycrfsuite` (only for the
third-party POS notebooks).

## 6. Unused, duplicate, obsolete or experimental items

| Item | Size | Evidence | Can be removed? |
|---|---|---|---|
| `checkpoints/step4_xlmr_mlm/checkpoint-1000/1250/1300`, `step4_xlmr_detector/checkpoint-800/938` | ~5.0 GB | no code reads `checkpoint-*` | yes, if you will not resume training (ask/back up first) |
| `checkpoints/step2_cnn_runs/` | 259 MB | experiment runs; `gray_aux_synth/best.pt` is byte-identical to `step2_cnn/best.pt` | archive |
| `*/last.pt` (8 files) | ~380 MB | resume-only | archive |
| `checkpoints/rembg/` | 168 MB | Step 1 comparison only | archive |
| `data/raw/handwritten_batch3/` | 11 MB | marked "not used" (cut-outs of batch-2 TRAIN pages) | keep as record or archive |
| `data/synthetic/step2_small/`, `*_small*` results | small | quick-test by-products | regenerable |
| `results/step3/cache/`, `results/step4/cache/`, `results/*/debug/` | ~85 MB | regenerable caches/figures | regenerable |
| `results/step3/run1_val_only/`, `run2_small_grid/` | small | superseded Step 3 runs | keep as evidence |
| 5 duplicate log pairs in `logs/step2/` | tiny | identical checksums | yes |
| `.DS_Store` (18), `__pycache__/` (9) | tiny | macOS/Python by-products | yes |
| `data/raw/handwritten/text_gt/D1.txt` = `D1..txt` | tiny | byte-identical; `D1.` is a second photo of the same text | **NOT VERIFIED** whether intended — confirm |

Nothing was deleted during this audit.
