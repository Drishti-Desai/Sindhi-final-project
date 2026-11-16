# 08 — Evaluation and results

Every number below is copied from a result file in the repository (path given). Nothing was re-measured for this
handover. The project's rule was: choose on training/validation data, measure each test set **once**, record a
prediction before the test, and never lower the 0.90 threshold.

---

## 1. What the metrics mean

| Metric | Meaning | Code |
|---|---|---|
| **Accuracy (letters)** | share of letter images classified as the right one of 52 classes | `scripts/step2_train_cnn.py` |
| **Top-3 accuracy** | right class among the model's 3 best guesses | error analysis json |
| **CER** (character error rate) | (substitutions + deletions + insertions) ÷ reference characters, on normalized text (spaces count). 0 = perfect. **Can exceed 1** when the output has many extra characters. | `common/metrics.py::cer` |
| **1 − CER** | "character accuracy" used by the gates; negative when CER > 1 | gates |
| **WER / word accuracy** | word edits ÷ reference words; word accuracy = 1 − WER (negative when many extra words are inserted) | `common/metrics.py::wer`, `xlmr/correct.py::word_accuracy` |
| **False-correction rate** | share of already-correct words that the corrector changed | `xlmr/correct.py::evaluate` |
| **Word-box F1** | boxes matched to true word boxes at IoU ≥ 0.5 (Hungarian matching); precision, recall, F1 | `tesseract_ft/boxes.py::score_boxes` |
| **F1 / dot recall (Step 1)** | text-pixel F1 against a known ink mask on synthetic pages; share of dots kept | `scripts/step1_compare_methods.py` |

---

## 2. Results by step

### Step 1 — Preprocessing (`results/gates/step1_preprocessing.json`, `results/step1/method_comparison.md`) — PASS

Measured on synthetic Sindhi pages with a known ink mask (12 conditions), so no real-page accuracy exists for this
step.

| Check | Result | Threshold |
|---|---|---|
| page mean F1 | 0.9559 | ≥ 0.90 |
| page dot recall | 0.9931 | ≥ 0.90 |
| glyph F1 ÷ oracle | 0.982 | ≥ 0.95 |
| glyph dot recall | 0.9444 | ≥ 0.90 |
| deskew mean error | 0.029° | ≤ 1.0° |

Method comparison (page mean / worst F1): flatten + Sauvola 0.956 / **0.931** (chosen); flatten + Otsu 0.970 /
0.822 (pencil); rembg + Otsu 0.158 / 0.039 (loses 60–90 % of dots).

### Step 2a — 52-letter CNN (`results/gates/step2_cnn.json`, `results/step2/cnn_metrics.json`, `cnn_per_class.csv`, `cnn_confusions.csv`) — FAIL, accepted

Test set: 873 letter images from 23 sheets never used in training.

| Run (`docs/STEP2_cnn_transformer.md` §5) | Validation | Test |
|---|---|---|
| baseline (binary) | 0.758 | 0.721 |
| + auxiliary heads | 0.768 | 0.751 |
| + font-rendered letters | 0.782 | 0.735 |
| grayscale input | 0.815 | 0.814 |
| **grayscale + 5-view test-time augmentation (chosen)** | **0.826** | **0.8225** (gate) |

Other test figures (without TTA, `cnn_metrics.json`): accuracy 0.8144, macro-F1 0.8183, best epoch 76; top-3
accuracy 0.9393; letter body right 0.9381; dots right given the body 0.8681.
Weakest letters (`cnn_per_class.csv`, ~17 test images each): ڄ 0.50, ڃ 0.50, ٽ 0.556, ذ 0.579, ج 0.611, ر 0.611.
Weakest families: jeem 0.683, hamza 0.692, reh 0.769, noon 0.769, beh 0.778.
Most frequent confusions (`cnn_confusions.csv`): ڃ→ڄ (6), ٽ→ٿ (4), ڄ→چ (4) — letters that differ only by dots.
A full confusion **matrix** image is **NOT FOUND**; the confusion list (25 most frequent pairs) and per-class
accuracy are saved as CSV.

### Step 2b — Word recognizer (`results/step2/recognizer_metrics.json`, `recognizer_ft_metrics.json`) — FAIL, accepted

| Test set (never trained or tuned on) | n | Original model CER | Fine-tuned CER (used) | Word accuracy (fine-tuned) |
|---|---|---|---|---|
| composed words (gate) | 2,000 | 0.138 (1 − CER 0.862) | 0.134 (**0.866**) | 0.506 |
| printed words, held-out fonts | 2,000 | 0.023 | 0.023 | 0.900 |
| real words, writers H and J (seen) | 56 | 0.303 | 0.227 | 0.429 |
| **real words, unseen writer** | 144 | 0.124 (doc) | **0.088 (1 − CER 0.912)** | **0.715** |

Validation selection: best epoch 13 (original, val mean 1 − CER 0.901) and 8 (fine-tuned, 0.885).
"Composed words" are made by putting real isolated-letter crops side by side; they look less like real cursive
writing than real word crops do.

### Step 3 — Word and line boxes (`results/step3/baseline_metrics.json`, `levers_metrics.json`, gate `step3_tesseract.json`) — FAIL, accepted

Unseen writer, 7 pages, 144 word boxes:

| Method | Word-box precision | recall | **F1** | Line-box F1 |
|---|---|---|---|---|
| Tesseract 5.5.2 on the whole page | 0.892 | 0.688 | 0.777 | 0.629 |
| Tesseract inside our own lines | 0.837 | 0.819 | 0.828 | 0.944 |
| **Own gap rule** (used by `--words gaps`) | — | — | **0.862** | — |
| Lever 2 (final measured method) | 0.859 | 0.764 | **0.809** (gate) | — |

Tesseract's own text on these pages: CER 0.487 (whole page). Recognizer on lever-2 boxes: found-word CER 0.095,
71 % of found words right, page CER 0.216.

### Step 4 — XLM-RoBERTa correction (`results/step4/val_comparison.json`, `test_metrics.json`, gate `step4_xlmroberta.json`) — FAIL, accepted

Synthetic test text (20 % of words corrupted with the recognizer's error model):

| Variant | Domain test word accuracy | False changes | Wikipedia test | False changes |
|---|---|---|---|---|
| no correction | 0.775 | — | 0.772 | — |
| word list only | 0.843 | 1.3 % | 0.808 | 3.7 % |
| **Sindhi XLM-R + detector (used)** | **0.890** (gate ≥ 0.90) | **2.07 %** (gate ≤ 2 %) | 0.858 | 4.3 % |
| XLM-R as downloaded (reference) | 0.874 | 1.6 % | 0.851 | 3.0 % |

Real handwriting (the main result): unseen writer's word crops **71.5 % → 79.2 %** words right, CER 0.071 → 0.051,
3 of 103 correct words wrongly changed; writers H/J 42.9 % → 60.7 %.
Masked-LM validation loss 3.60 → 3.11 (`results/step4/mlm_log.json`); detector validation loss 0.135
(`detector_log.json`).

### Step 5 — Full pipeline (`results/step5/val_metrics.json`, `test_metrics.json`, gate `step5_pipeline.json`) — FAIL, accepted

| Set | Reading | CER before → after correction |
|---|---|---|
| VAL pages (71 words; development) | Step 4 gap-rule boxes | 0.441 → 0.462 |
| | **pipeline (joins + reading order)** | **0.240 → 0.246** |
| **Unseen writer, 7 pages (gate)** | Step 4 gap rule | 0.208 → 0.193 |
| | pipeline | 0.237 → **0.222** (1 − CER **0.778**) |
| H/J TESS_TEST pages | pipeline | 0.314 → 0.303 |

Prediction recorded before the run: CER 0.12–0.15. Actual 0.222 (prediction wrong; analysis in
`docs/STEP5_pipeline.md` §4: joins over-join a writer with clear gaps).

### Step 6 — Final test (`results/step6/final_metrics.json`, gate `step6_final.json`) — FAIL, accepted

FINAL_TEST: 7 pages, 147 words, whole page, measured once, writers seen in training (new text).

| Set | Words read / true | Lines found / true | CER before → after |
|---|---|---|---|
| **FINAL_TEST (gate)** | 396 / 147 | 119 / 23 | 1.709 → **1.650** (1 − CER −0.650) |
| writer D (2 pages: faint pencil, dotted rulings) | 306 / 42 | 99 / 6 | 4.950 → 4.724 |
| writer H (2 pages) | 15 / 19 | 3 / 3 | 0.187 → 0.173 |
| writer J (3 pages) | 75 / 86 | 17 / 14 | 0.373 → 0.388 |
| writers H + J | 90 / 105 | 20 / 17 | 0.343 → 0.354 |
| reference: TESS_TEST whole page | 55 / 61 | 12 / 11 | 0.307 → 0.291 |

Best page: H5, CER 0.054. 95 % bootstrap interval of CER over the 7 pages: 0.32–3.53.
Prediction recorded before the run: CER 0.28–0.40 (right for H/J, wrong for D).

---

## 3. Training and inference time

| Item | Time | Source |
|---|---|---|
| Letter CNN run | ~10–30 min (chosen run: 106 epochs, ~21 min) | `logs/step2/letter_cnn_*.log` |
| Word recognizer, 15 epochs | ~3 h 51 min (summed epoch times; earlier attempts crashed/ran out of memory) | `logs/step2/recognizer_run4.out`, `recognizer_run4_resume.out` |
| Fine-tuning, 8 epochs | ~2 h 7 min | `logs/step2/recognizer_ft_run1.out` |
| Sindhi masked LM, 1,300 steps | ~1 h 37 min | `logs/step4/mlm.log` |
| Detector, 938 steps | ~21 min (trainer runtime 1,265 s) | `results/step4/detector_log.json` |
| Inference, page A1 (3 lines, 17 words) | ~2 s (correction off), ~9–10 s (auto-correct), ~8 s (suggest) | app test runs, 2026-11-15/16 |

Inference time was measured on one page only; a systematic timing study is **NOT DONE**.

---

## 4. How reliable are these numbers?

- **Duplicates:** the letter dataset had 77 % byte-identical copies; they were removed before splitting
  (4,241 unique images) and the split is by sheet, so the same writer's sheet is not in train and test. See
  `04_DATASET_AUDIT_AND_PROVENANCE.md` for the verified duplicate counts.
- **Leakage controls:** sentences of every test/FINAL_TEST page were removed from all training text
  (`tesseract_ft/texts.py`); batch-2 copies of FINAL_TEST pages were excluded; the unseen writer was never trained on.
- **Small samples:** test sets have 56–147 words; per-writer numbers are noisy (bootstrap interval in Step 6).
- **Seen writers:** FINAL_TEST writers appear in training (new text only); only one unseen writer exists.
- **Synthetic vs real:** Step 1 and the Step 4 gate are synthetic benchmarks; they overstate real performance.
- **Test reuse:** the unseen-writer pages were measured in Steps 3, 4 and 5 (each time with a method fixed
  beforehand), so they are no longer a fully clean test; FINAL_TEST was read exactly once.

## 5. What still needs testing

- A **new held-out set** from new writers (pen and pencil, plain and ruled paper) — every future improvement needs it.
- Leave-one-writer-out cross-validation of the recognizer.
- Inference time over many pages; CPU-only and Linux/Windows runs.
- The plausibility warning on new pages (it was designed after seeing FINAL_TEST).
