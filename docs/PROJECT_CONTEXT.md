# Sindhi Handwriting OCR — complete project context (handoff document)

Written 2026-11-15 for anyone (person or AI assistant) who continues this project. Read this file first; it
explains what exists, why it was built this way, the rules the project follows, every result, and where to make
changes. Detailed per-step reports are in `docs/STEP1_preprocessing.md` … `docs/STEP6_testing.md`; the original
plan and dataset findings are in `PLAN.md`; data origins are in `data/PROVENANCE.md`.

---

## 1. What the project is

A system that turns a **photo of a handwritten Sindhi page** into **Unicode Sindhi text**. Sindhi uses a
Perso-Arabic script with 52 letters, many of which differ only by dots (for example ب ٻ ڀ پ ت ٿ ٽ ٺ ث), written
right to left in connected (cursive) form.

It was built step by step. Each step had to pass an **accuracy gate (≥ 0.90)** before the next could start;
a failed gate stopped the work until the project owner decided. The owner accepted every failure with a
documented override, so the full pipeline exists and runs, but **the overall accuracy is far below 90%**.

Current state (2026-11-15): finished and committed. Command line `run_pipeline.py`, local web app `app.py`,
79 unit tests passing.

---

## 2. Rules the project follows (keep following them)

These were set by the project owner and applied throughout. A continuing assistant should respect them.

1. **Work only inside the project folder** `~/Downloads/NEW`. Files from other folders may be *copied* in, never
   modified, moved or deleted at their original location; every copy is recorded in `data/PROVENANCE.md`.
2. **Use the project's `.venv`** only; never install globally. Pin every package in `requirements.txt`.
3. **Ask the owner before** any command that installs software, downloads large data, or deletes files.
4. **Never fake or round up results.** A result below its gate is reported as FAIL, with error analysis and options.
   The gate threshold (0.90) is never lowered in code.
5. **Test sets are measured once**, with the method chosen beforehand on training/validation data only. Predictions
   are written into the docs and committed *before* a test run. Nothing is re-tuned after looking at a test result.
6. **Commit after each step**; docs go in `docs/STEPn_*.md`, plus README and PLAN.md updates.
7. API keys (only `RF_API_KEY`, Roboflow) come only from `.env`, never from code.
8. Write plain, honest documentation: what was done, what failed, why.

---

## 3. Hardware and environment

| Item | Value |
|---|---|
| Machine | Apple M2 MacBook, **8 GB RAM** (the main constraint on every model choice) |
| OS | macOS (Darwin 25) |
| Python | 3.13.3 in `.venv` |
| Key packages | torch 2.14.1 (Apple MPS GPU), transformers 5.23.0, accelerate 1.15.0, opencv-python-headless 5.0, scikit-image 0.26, gradio 6.32, pytest 9.1 — full pinned list in `requirements.txt` |
| Tesseract | 5.5.2 system install (used only in Step 3 experiments, not in the final pipeline) |
| GPU memory | Capped with `PYTORCH_MPS_HIGH_WATERMARK_RATIO` (0.6 for Step 2 training, 0.7 for Step 4 and the pipeline) so jobs fail fast instead of swapping |
| Git | local repo, branch `main` (HEAD), author "drishti" |

Memory techniques used to fit 8 GB: gradient checkpointing, small micro-batches with gradient accumulation,
frozen embeddings stored in bfloat16 during XLM-R training, output layer applied only at masked positions,
loading the two XLM-R models one after another at inference time.

---

## 4. Final architecture

```
                         ┌──────────────────────────── run_pipeline.py / app.py ────────────────────────────┐
 page photo (jpg/png) ─► │ 1 PREPROCESS          preprocess/  (Step 1)                                       │
                         │   flatten background (morphological closing + division normalisation)            │
                         │   → best-separating gray channel → Sauvola binarisation → auto polarity          │
                         │   → deskew → remove straight ruled lines / borders / desk around the sheet       │
                         │   → drop specks; keep dots (min area 0.1 × stroke width²)                        │
                         │   → clean binary page (text 0, paper 255)                                        │
                         │                                                                                  │
                         │ 2 LAYOUT              tesseract_ft/boxes.py + pipeline/layout.py  (Steps 3, 5)    │
                         │   connected components → "tracked" line finder (follows sloped lines)            │
                         │   → each line = merged ink spans (right to left) + dots/diacritics               │
                         │   → reading order: line pieces that overlap vertically ≥ 60% and horizontally    │
                         │     ≤ 75% form one row; rows top→bottom, pieces right→left                       │
                         │                                                                                  │
                         │ 3 WORDS               pipeline/page.py  (Step 5)                                  │
                         │   --words joins (default): the recognizer reads every run of 1–6 neighbouring    │
                         │     spans; dynamic programming keeps the grouping with the best total            │
                         │     log-probability (no tuned settings)                                          │
                         │   --words gaps: cut where gap > max(0.3 × letter height, 0.75 × median gap)      │
                         │                                                                                  │
                         │ 4 RECOGNIZE           cnn_transformer/models.py: WordRecognizer  (Step 2)         │
                         │   word image 64 px high (≤ 320 px wide)                                          │
                         │   → ResNet trunk 32-64-128-256 ch (initialised from the letter CNN)              │
                         │   → collapse conv to d=256, ~1 step per 8 px → Transformer encoder (2 layers)    │
                         │   → Transformer decoder (3 layers, 4 heads, ff 1024) → characters (vocab 82)     │
                         │   greedy decoding; 8.1M parameters; checkpoint step2_recognizer_ft/best.pt       │
                         │                                                                                  │
                         │ 5 CORRECT             xlmr/  (Step 4)                                            │
                         │   candidates: 47,422-word lexicon, SymSpell-style lookup + weighted edit         │
                         │     distance (cheap look-alike-letter swaps, alef insert/delete), splits, merges │
                         │   scoring: XLM-RoBERTa-base further trained on Sindhi (masked-LM score of each   │
                         │     candidate in context) + word frequency + edit cost                           │
                         │   detector: XLM-R token classifier "is this word wrong?" flags known words       │
                         │   decision weights chosen on validation (results/step4/val_comparison.json)      │
                         │                                                                                  │
                         │ 6 OUTPUT              <name>.txt, <name>.json (boxes, readings, corrections),     │
                         │   plausibility warning if the page was probably not read as text                 │
                         └──────────────────────────────────────────────────────────────────────────────────┘
```

**Training-time architecture (how the models were made):**

```
letters52 crops (4,241 unique, 52 classes, 122 sheets)
  └─► LetterCNN (ResNet-style, 64×64 grayscale, auxiliary heads: body shape / dot position / dot count,
       25% font-rendered letters, 5-view test-time augmentation)  → test 0.8225
        └─ its trunk initialises ─► WordRecognizer
              trained on: printed renders of corpus words (8 macOS fonts; Tahoma held out)
                        + "composed" words assembled from real letter crops
                        + real word crops (97 from old pages)
              then fine-tuned with 1,104 real batch-2 word crops (mix 40% real / 30% composed / 30% printed)

xlm-roberta-base (278M params, copied from the local Hugging Face cache, SHA-256 checked)
  └─► further masked-LM training on Sindhi text (legal CSV + POS sentences ×20, Wikipedia; 1,300 steps;
       lower 6 layers and embeddings frozen)  → val loss 3.60 → 3.11
        └─► token-classification detector trained on text corrupted with a model of the recognizer's errors
```

**What is NOT in the final pipeline:** Tesseract (tested in Step 3; our own box finder was better), the 52-letter
CNN on its own (only its trunk is reused), rembg (only compared in Step 1), the Sindhi hard rules (tested in
Step 4, no gain), lever 2 of Step 3 (kept in code, overfit).

---

## 5. Data

All raw data lives in `data/raw/` (git-ignored). Splits and manifests in `data/splits/` are tracked.

| Dataset | Size | Used for | Notes |
|---|---|---|---|
| `letters52/` | 18,088 files, **4,241 unique** letter crops, 52 classes | Step 2 letter CNN; composed words | tiny phone crops (22–77 px tall); split by **sheet**: train 3,001 / val 367 / test 873 |
| `handwritten/` (22 pages + texts) | ~450 words, writers A, D, H, J | TESS_TRAIN (12 pages), TESS_TEST (3: H1, J2, J3), **FINAL_TEST (7: D2, D6., H3, H5, J1, J4, J5)** | split by story group; FINAL_TEST is now **used up** (measured once in Step 6) |
| `handwritten_batch2/` | 129 images + 122 texts | Step 2 fine-tuning, Step 3–5 tuning and tests | pages split TRAIN 98 / VAL 12 / **TEST_UNSEEN_WRITER 7** (purple-pen writer); word crops cut with the known text: 1,104 / 71 / 144 verified |
| `handwritten_batch3/` | 17 images | not used | all are cut-outs of batch-2 TRAIN pages |
| `pos_handwritten/` | 976 unique images, many student writers | **not used** (no transcriptions) | Mendeley Data, Soomro & Khalid 2024, doi:10.17632/phk66sgmp5.1, **CC BY 4.0**; best source of new writers once transcribed |
| `pos_text/` | 676 POS-tagged sentences → 518 clean | Step 4 text | no licence found; used only inside the project |
| `corpus/wikipedia_sd/` | 16,928 articles, ~4.1M words | Step 2 word lists, Step 4 text | CC BY-SA 3.0 + GFDL |
| `~/Downloads/Sindhi_Legal_Dataset.csv` | ~2,100 Q/A sentences | Step 4 text (read in place, never copied or changed) | the handwritten pages copy texts from it |
| `roboflow_sindhi_ocrf/` | 509 printed book pages, 113k word boxes, no text | not used in the pipeline | CC BY 4.0 |

**Leakage protection:** every sentence of a test or FINAL_TEST page is removed from all training text (4-gram
overlap or containment, `tesseract_ft/texts.py`). Writers: H and J appear in training *and* tests; only the
purple-pen writer is truly unseen. FINAL_TEST measures **seen writers, new text**.

**Text normalisation** (`common/text.py`): NFC, unified letter variants (notably `ھ` U+06BE → `ه` U+0647, because
Wikipedia and the handwritten texts disagree), harakat removed for scoring.

---

## 6. Step-by-step history and results

Metric names: **CER** = character edits / reference characters (spaces count); **1 − CER** is the "accuracy"
used by the gates; **word accuracy** = 1 − WER (can go below 0 when many extra words are inserted).

| Step | What was built | Gate (≥ 0.90) | Result | Status |
|---|---|---|---|---|
| 0 | Dataset inspection, `PLAN.md`, splits | — | — | done |
| 1 | Preprocessing (`preprocess/`) | page F1, dot recall, glyph F1, deskew on synthetic pages | F1 0.956, dot recall 0.993, glyph F1 0.982 | **PASS** |
| 2a | 52-letter CNN | letter test accuracy | 0.8225 | FAIL, accepted |
| 2b | Word recognizer (CNN + Transformer) | 1 − CER on composed test words | 0.866; **unseen writer's real words 0.912 (71.5% words right)** — the reference number | FAIL, accepted |
| 3 | Word/line boxes: Tesseract vs own finder | word-box F1 on unseen writer (IoU ≥ 0.5) | Tesseract best 0.828, own gap rule 0.862, lever 2 0.809 (measured final method) | FAIL, accepted; pipeline uses own finder |
| 4 | XLM-R correction | domain-text word accuracy, ≤ 2% correct words changed | 0.890, 2.07%; real handwriting: unseen writer 71.5% → 79.2% words right, 3 of 103 correct words wrongly changed | FAIL, accepted |
| 5 | `run_pipeline.py` with joins + reading order | 1 − CER, unseen writer's pages | 0.778 (the Step 4 gap-rule reading had 0.807); VAL pages CER 0.462 → 0.246 | FAIL, accepted |
| 6 | Final test on FINAL_TEST (whole pages, once) | 1 − CER | **−0.650** (writer D's 2 pages: 306 words read for 42); writers H + J: CER 0.354 (best page H5: 0.054) | FAIL, accepted |

Gate files with full metrics and the owner's decisions: `results/gates/*.json`. A gate stays "FAIL" with a
`user_decision` field when accepted; scripts call `require_gate(name, allow_user_accepted=True)`.

### Key lessons (evidence in the step docs)

1. **Word segmentation is the main error source.** On VAL pages: perfect word crops CER 0.12; whole pages through
   word boxes 0.24–0.44. Sindhi letters that do not join the next letter create gaps inside words.
2. **Recognizer joins over-join** (about 15% fewer words than truth) because the rule ignores gap width; this helps
   cramped writers and hurts writers with clear spacing (hence `--words gaps`).
3. **Few writers.** Settings tuned on the training writers did not carry over (Step 3 lever 2: 0.928 on tuning pages,
   0.809 on the unseen writer). The recognizer memorised the training writers, so its scores on their pages are
   too optimistic for tuning anything.
4. **Pencil and ruled/dotted paper break the pipeline** (Step 6): ruling dots, pencil fragments and speckle are found
   as dozens of overlapping "lines" and read as words. No text/non-text filter exists.
5. **The corrector is domain-tuned.** On legal/everyday text it changes ~2% of correct words; on Wikipedia ~4%.
6. Letters are hard: 52 classes differing by dots on tiny photos; learning curve suggests 90% would need tens of
   times more letter sheets.

---

## 7. Repository map

```
NEW/
  run_pipeline.py          command line: images → .txt/.json   (--words joins|gaps, --correct off|auto|suggest, --erase-rules, --remove-circled)
  app.py                   local Gradio web app (127.0.0.1:7860), same code as run_pipeline.run; Off / Auto-correct / Suggest
                           modes, review rows + "Apply choices", opt-in saving to user_feedback/ (pipeline/feedback.py)
  PLAN.md  README.md  requirements.txt  .env.example  pytest.ini
  configs/
    paths.yaml             every path (nothing hardcoded); load with common.paths.load_paths / load_config
    letter_map.yaml        class id 1..52 → letter
    step1_preprocess.yaml  binarisation method, profiles (page / glyph), thresholds
    step2_cnn.yaml         letter CNN + recognizer + fine-tune hyper-parameters, gates
    step3_tesseract.yaml   Tesseract grid, gate, box_finder used by the pipeline (gap_h 0.3, gap_med 0.75)
    step4_xlmr.yaml        MLM / detector training, gate, MPS memory ratio
  common/                  paths.py, gates.py (write/read/require/accept gates), metrics.py (CER/WER), text.py (normalisation)
  preprocess/              ops.py (one function per operation), pipeline.py (preprocess_image), synthetic.py, segmentation.py (rembg)
  cnn_transformer/         models.py (LetterCNN, WordRecognizer), letters.py, words.py (vocab, datasets, composed words),
                           segment.py (components, line assignment, ruled-line erase), align.py, train_utils.py
  tesseract_ft/            pages.py (evaluation pages), boxes.py (line finder, gap rule, box scoring), cuts.py (levers 1–2), texts.py (leak filter)
  xlmr/                    data.py (text splits), errors.py (recognizer error model, corruption), correct.py (candidates, decision, metrics),
                           lm.py (MaskedScorer, Detector), rules.py (Sindhi hard rules; not used)
  pipeline/                layout.py (reading order), page.py (clean_page, read_page, joins, Corrector, plausibility warning)
  scripts/                 one runner per stage: prepare_splits, step1_*, step2_*, step3_*, step4_*, step5_*, step6_final
  tests/                   test_preprocess, test_segment, test_step2..5, test_suggest, test_present, test_app  (79 tests: .venv/bin/python -m pytest -q)
  docs/                    STEP1..STEP6 reports, this file
  data/                    raw/ (ignored), preprocessed/ (ignored), splits/ (tracked), PROVENANCE.md
  results/                 gates/ (tracked), step1..step6 metrics (tracked), */cache and */debug (ignored)
  checkpoints/             (ignored, ~9 GB) step2_cnn, step2_recognizer(_ft), xlmr_base, step4_xlmr_mlm, step4_xlmr_detector
  logs/                    (ignored) training and run logs
```

Models needed to run the pipeline: `checkpoints/step2_recognizer_ft/best.pt`, `checkpoints/step4_xlmr_mlm/best`,
`checkpoints/step4_xlmr_detector/best` (the last two are ~1.1 GB each as safetensors; folders hold extra
training checkpoints that can be deleted to save space — ask the owner first).

---

## 8. How to run things

```bash
cd ~/Downloads/NEW
.venv/bin/python run_pipeline.py page.jpg --out pipeline_out/       # text + JSON per image
.venv/bin/python app.py                                             # web app → http://127.0.0.1:7860 (Ctrl+C stops)
.venv/bin/python -m pytest -q                                       # 79 tests

# re-running stages (each checks the previous gate first)
.venv/bin/python scripts/step1_preprocess.py
.venv/bin/python scripts/step2_train_cnn.py --run <name> [--eval --gate]   # letter CNN runs (see docs/STEP2 for flags)
.venv/bin/python scripts/step2_make_synthetic.py                    # word lists + printed renders (~6 min)
.venv/bin/python scripts/step2_train_recognizer.py [--finetune]     # recognizer (~3 h; fine-tune ~2 h)
.venv/bin/python scripts/step4_prepare.py                           # text splits, lexicon, error model
.venv/bin/python scripts/step4_train_mlm.py                         # ~1.5 h
.venv/bin/python scripts/step4_train_detector.py                    # ~16 min
.venv/bin/python scripts/step4_compare.py                           # validation comparisons
.venv/bin/python scripts/step5_eval.py                              # VAL pages through the pipeline
```

Python API:
```python
import run_pipeline
from pipeline.page import clean_page, load_rgb
from cnn_transformer.train_utils import get_device
page = run_pipeline.run([clean_page(load_rgb("page.jpg"))], get_device("auto"), correct=True, words="joins")[0]
page["text"], page["lines"], page["tokens"], page["corrected"], page["actions"], page["warnings"]
```

---

## 9. Where to change what

| To change… | Edit |
|---|---|
| Binarisation, line/ruling removal, deskew | `preprocess/ops.py`, `preprocess/pipeline.py`, `configs/step1_preprocess.yaml` |
| Text/non-text filtering (missing today) | new step between `clean_page` and `line_structures` in `pipeline/page.py` |
| Line finding | `tesseract_ft/boxes.py` (`line_structures`, `cluster_lines`) |
| Reading order | `pipeline/layout.py` (`order_lines`, thresholds in `ORDER`) |
| Word joining | `pipeline/page.py` (`best_grouping`, `line_units`, `MAX_SPANS`) |
| Recognizer model / decoding | `cnn_transformer/models.py` (`WordRecognizer.greedy`), training `scripts/step2_train_recognizer.py` |
| Correction | `xlmr/correct.py` (candidates, `decide`), weights `results/step4/val_comparison.json`, models `checkpoints/step4_*` |
| Plausibility warning | `pipeline/page.py` (`plausibility`, constants above it) |
| Suggest mode / saved choices | `xlmr/correct.py` (`suggest`, `apply_choices`), `pipeline/page.py` (`apply_page_choices`), `pipeline/feedback.py`, `app.py` |
| Web app | `app.py` (layout, theme), `pipeline/present.py` (what is displayed) |

Conventions: paths only via `configs/paths.yaml`; every runner script calls `require_gate` for its predecessor;
metrics via `common/metrics.py` and `xlmr.correct.evaluate`; text via `common/text.normalize` and
`cnn_transformer.words.label_norm`. Code style: short docstrings explaining *why*, plain names, no hardcoded paths.

---

## 10. Pitfalls a newcomer will hit

- **Test sets are spent.** FINAL_TEST was read once (Step 6). The unseen-writer pages were measured in Steps 3–5,
  so they are no longer a clean test either. Any new claim of improvement needs **new held-out pages**.
- Some batch-2 page texts cover only part of the page (e.g. VAL page A32): page-level evaluation in Steps 3–5 scores
  only the **verified part** of each page (`region` masks). Step 6 scored whole pages.
- `ھ` vs `ه`: always normalise before comparing text.
- The recognizer has seen the batch-2 TRAIN writers; tuning anything on its scores for those pages overfits.
- Memory: loading both XLM-R models plus training anything at the same time will exceed 8 GB. Keep the MPS caps.
- Pages must be upright; there is no automatic 90°/180° rotation detection.
- `results/step4/val_comparison.json` holds the chosen correction variant and weights; the `Corrector` refuses to
  run if the variant is not `xlmr_sindhi_det`.

---

## 11. Improvement ideas (ranked; see also README "Future work")

1. **New data + a new held-out test set** (prerequisite for every claim). Have people copy typed Sindhi text by hand
   (pen and pencil, plain and ruled paper), so the typed text is the ground truth; keep whole writers out of
   training. Transcribe the 976 Mendeley POS images (≈10–16 h).
2. **Text/non-text filter + ruling removal** (solid and dotted rulings, speckle, paper edges) before line finding.
3. **Better word joining**: combine gap width, recognizer score and LM score; tune with out-of-fold recognizers
   (leave one writer out) so the scores are honest.
4. **Beam search with a lexicon** in the recognizer instead of greedy decoding.
5. **Line-level recognizer** (CNN/Transformer + CTC, or fine-tuned TrOCR) that needs no word cutting.
6. **Human-in-the-loop app**: show low-confidence words, let users fix them, save fixes as new training data.
7. Evaluation rigour: leave-one-writer-out cross-validation, confidence intervals, ablation tables.
