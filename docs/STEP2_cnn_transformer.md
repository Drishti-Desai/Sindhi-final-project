# Step 2 — CNN + Transformer decoder

Two models, trained in order. Each one is proven before it is used:

1. **Letter CNN** (`cnn_transformer/models.py: LetterCNN`): one 64×64 letter image → one of the 52 classes.
2. **Word recognizer** (`WordRecognizer`): the letter CNN's convolution trunk reads a whole **word
   image** (height 64, width ≤ 512) as a sequence of feature vectors. A Transformer decoder
   then writes the word's Unicode characters one at a time. The architecture was approved at
   Step 1 because cursive handwriting cannot be reliably cut into letters (see PLAN.md §4).

## 1. Where the data goes

```
data/
  preprocessed/letters52/<class_id>/<stem>.png   Step-1 output (binary). Label = folder name.
  preprocessed/letters52/index.csv               relpath, class_id, sheet, split (train/val/test)
  splits/letters52/split.csv                     the fixed split (grouped by sheet, seed 42)
  splits/step2_words/{train,val,test}.txt        word lists (one word per line, UTF-8)
  splits/step2_real_words/verified.csv           real word crops: file, label, split, verified(0/1)
  synthetic/step2/printed/{train,val,test}/*.png + labels.csv (file,label,font,ptsize)
  preprocessed/handwritten_words/{TESS_TRAIN,TESS_TEST}/*.png + labels.csv
configs/letter_map.yaml   class id -> letter, plus `structure` (body shape, dot position, dot count)
configs/step2_cnn.yaml    all hyper-parameters; `small:` block for quick tests
```

To add letter data: put PNG/JPG crops in a new sheet-named batch under `data/raw/letters52/<split>/<class_id>/`,
rerun `scripts/prepare_splits.py` and `scripts/step1_preprocess.py --dataset letters52`.

## 2. Commands

```bash
# letters (needs gate step1_preprocessing)
python scripts/step2_train_cnn.py --run aux --small --fresh     # 3-epoch pipeline check
python scripts/step2_train_cnn.py --run aux --synthetic-p 0     # an experiment (resumable: rerun to continue)
python scripts/step2_train_cnn.py --run aux --eval --gate       # promote the run with the best VAL accuracy

# words (needs gate step2_cnn)
python scripts/step2_make_synthetic.py                          # word lists + printed renders (~6 min)
python scripts/step2_extract_real_words.py                      # real word crops from TESS pages
python scripts/step2_train_recognizer.py [--small] [--fresh]    # train + test + gate step2_cnn_transformer

python -m pytest tests/test_step2.py -q
```

Device: CUDA if present, else Apple MPS, else CPU (`device: auto`). Checkpoints are saved every epoch
(`last.pt`, with optimizer, scheduler and RNG state) plus `best.pt`. Re-running a command resumes.

## 3. Outputs

| what | where |
|---|---|
| letter CNN experiments | `checkpoints/step2_cnn_runs/<run>/`, `results/step2/cnn_runs/<run>.json`, `logs/step2/letter_cnn_<run>.log` |
| promoted letter CNN | `checkpoints/step2_cnn/best.pt`, `results/step2/cnn_metrics.json`, `cnn_per_class.csv`, `cnn_confusions.csv`, `cnn_errors.png` |
| recognizer | `checkpoints/step2_recognizer/`, `results/step2/recognizer_metrics.json`, `recognizer_predictions_<set>.csv` |
| gates | `results/gates/step2_cnn.json`, `results/gates/step2_cnn_transformer.json` |

## 4. Accuracy definitions and pass criteria

| gate | metric | test data | pass |
|---|---|---|---|
| `step2_cnn` | top-1 accuracy over 52 classes | 873 crops from **23 sheets never seen in training** | ≥ 0.90 |
| `step2_cnn_transformer` | the letter gate **and** 1 − CER | 2,000 **held-out words** built from **test-sheet** letter crops (`composed_test`) | ≥ 0.90 |

CER = (substitutions + deletions + insertions) / reference characters (`common/metrics.py`), on
normalised text without harakat (`common/text.py`). Also reported for the recognizer, but not gated:
`printed_test` (held-out words in held-out fonts) and `real_test` (56 verified real handwritten
word crops from TESS_TEST pages H1, J2, J3).

Rules: runs are compared on the validation split only. The test split is evaluated once per run
and reported; it never chooses hyper-parameters. Thresholds are never lowered in code.

## 5. Results (2026-10-18) — letter gate FAIL, Step 2 stopped before the recognizer

All runs use the same sheet-disjoint split (train 3,001 / val 367 / test 873 crops; 88 / 11 / 23 sheets).
The model is chosen on **validation**; test is only reported.

| run | input | what changed | val | test |
|---|---|---|---|---|
| baseline | binary | ResNet-style CNN, augmentation, label smoothing | 0.758 | 0.721 |
| aux | binary | + auxiliary heads (body shape, dot position, dot count) | 0.768 | 0.751 |
| aux_synth | binary | + font-rendered letters (25% of samples) | 0.782 | 0.735 |
| lc25 / lc50 | binary | aux_synth on 25% / 50% of training sheets | 0.730 / 0.749 | 0.709 / 0.729 |
| **gray_aux_synth** | **grayscale** | aux_synth on the cleaned grayscale *before* thresholding | **0.815** | **0.814** |
| **chosen: gray_aux_synth + TTA** | grayscale | 5-view test-time augmentation | **0.826** | **0.8225** |

Gate `step2_cnn`: **0.8225 < 0.90 → FAIL** (`results/gates/step2_cnn.json`).

Error analysis of the chosen model (`results/step2/cnn_error_analysis_gray_aux_synth.json`):
- letter **body** correct 93.8%, top-3 correct 93.9%; dots correct given the right body 86.8%
- weakest families: jeem (ج ڄ ڃ چ ڇ ح خ) 68%, hamza 69%, reh/noon 77%, beh (9 dot variants) 78%
- manual check of 48 binary-model errors: ~52% dots lost or fused in the crop, ~15% writer style or
  suspect label, ~33% model mistakes on clear input. Grayscale input recovered part of the first group.
- learning curve: each doubling of training sheets adds about 1–3 points, so 90% from data volume alone
  would need tens of times more sheets.

Why 90% is hard here: the crops are small phone photos (22–77 px tall) with ink only 10–45% darker than
the paper, and many letters differ only in a few dots, which writers often draw as dashes or squiggles.

Practical note: the chosen CNN takes the **grayscale** crop (`PreprocessResult.gray`), not the binary one.

## 6. Word recognizer results (2026-10-20) — option A, word-level gate FAIL

The user chose option A (2026-10-18): keep the 82.25% letter CNN as the backbone (its gate stays FAIL,
with the acceptance recorded in `results/gates/step2_cnn.json`) and gate Step 2 on word-level 1 − CER.

Training: 15 epochs × 900 steps, effective batch 32 (2 × 16 with gradient accumulation), word width
≤ 320 px, gradient checkpointing and a 3.2 GB Apple-GPU memory cap. Earlier attempts ran out of memory or
swapped on this 8 GB Mac. Best epoch by validation: 13.

| test set (never used for training or selection) | n | CER | 1 − CER | word accuracy |
|---|---|---|---|---|
| **composed_test** (held-out words, isolated letters of unseen writers) — gated | 2,000 | 0.138 | **0.862** (95% CI 0.854–0.869) | 0.492 |
| printed_test (held-out words, held-out fonts Tahoma/Tahoma Bold) | 2,000 | 0.023 | 0.977 | 0.899 |
| **real_test** (verified real handwritten word crops, pages H1, J2, J3) | 56 | **0.303** (95% CI 0.245–0.363) | 0.697 | 0.232 |

Gate `step2_cnn_transformer`: **0.862 < 0.90 → FAIL**.

Real handwriting is **much worse** than composed words: CER is about 2.2× higher (0.30 vs 0.14) and
word accuracy is half. Per page: H1 CER 0.09 (9 words), J2 0.31 (21), J3 0.36 (26). J2 and J3 are the
same writer and largely the same text, so the real test is effectively about two writers and is noisy.
Typical errors are dot and short-stroke confusions in connected script (ر/د, ۽/۾, ڌ/ن, ق/ف) and
merged/split letter groups (گرائونڊ → گالوعبر). The model learned connected script mostly from printed
fonts and isolated letters; it saw only 97 real handwritten words in training.

## 7. Fine-tuning with real handwriting, batch 2 (2026-11-14) — real handwriting improved; word gate still FAIL

`python scripts/step2_train_recognizer.py --finetune` continued from `checkpoints/step2_recognizer/best.pt`
into `checkpoints/step2_recognizer_ft/` (settings: `finetune:` in `configs/step2_cnn.yaml`). Real training pool:
1,104 batch-2 TRAIN crops + 97 older real words; mix 40% real / 30% composed / 30% printed; lr 1e-4; 8 epochs ×
900 steps; same memory-safe settings. The vocabulary gained `؟` and `/` (the old model could not write `؟`), so
CER is reported with and without punctuation. Checkpoints were selected on validation only (printed, composed and
the 71 batch-2 VAL crops); the starting model was the score to beat. Best epoch: 8.

| test set (never trained or tuned on) | n | old model CER | fine-tuned CER | word accuracy old → new |
|---|---|---|---|---|
| composed_test (gate: 1 − CER ≥ 0.90) | 2,000 | 0.138 (0.862) | 0.134 (**0.866**) | 49.2% → 50.6% |
| printed_test | 2,000 | 0.023 | 0.023 | 89.9% → 90.0% |
| old real test, writer H (seen in training) | 9 | 0.091 | 0.030 | 67% → 89% |
| old real test, writer J (seen in training now) | 47 | 0.338 | 0.259 | 15% → 34% |
| **unseen writer (purple pen)** | **144** | **0.124** | **0.088** | **62.5% → 71.5%** |
| unseen writer, no punctuation | 144 | 0.107 | 0.068 | |

Paired bootstrap (5,000 resamples of the same words for both models), CER improvement with 95% interval:
unseen writer +0.036 [+0.015, +0.057] (no punctuation +0.039 [+0.020, +0.060]); 24 words better, 4–6 worse.
Old 56 words +0.077 [+0.020, +0.138], but writers H and J are now seen in training, so part of that is familiarity.

Honest reading:
- Real handwriting **did** improve, including on a writer the model never saw: about 29% fewer character errors.
- The gate is still **FAIL**: composed words 0.866 < 0.90. Fine-tuning on real words barely moves it; the composed
  test is limited by the isolated-letter crops themselves (see section 5).
- The unseen-writer test is one neat writer (144 words), and its word boundaries were chosen with the old model's
  help (that can only favour the old model). Rare letters (ڍ ڦ ڄ ڀ ٻ ث غ ڱ) are almost absent from both training
  and test, so nothing here measures them.

## 8. Decision (2026-11-14) — Step 2 accepted with the word gate FAIL

The user chose to accept Step 2 as it is (option 1), as was done for the letter gate. Nothing was relaxed: the gate
file `results/gates/step2_cnn_transformer.json` still says **FAIL** (composed 1 − CER 0.866 < 0.90), with the
decision stored next to it (`user_decision`). Step 3 opens with `require_gate(..., allow_user_accepted=True)` and
prints a warning every time.

- Model used from now on: `checkpoints/step2_recognizer_ft/best.pt`.
- **Reference number for real handwriting** (stored as `extra.reference_real_handwriting` in the gate file):
  unseen writer (purple pen, 144 words) **1 − CER 0.912**, CER 0.088 (0.068 without punctuation), **71.5% of words
  fully correct**. This assumes correct word boundaries. Later steps report their real-handwriting results against it.
- `handwritten_batch3` was **not** used: all 17 of its images are cut-outs of batch 2 TRAIN pages that were already
  trained on, so they would only have counted the same writing twice (`data/PROVENANCE.md`).
