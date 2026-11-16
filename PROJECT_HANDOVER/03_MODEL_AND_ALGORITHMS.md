# 03 — Every model and algorithm

Parameter counts were read from the saved checkpoints; hyper-parameters from the code, configs and stored training
arguments (file:line evidence in `appendix/C_raw_models_references_safety.md`, Part A). Nothing was retrained.

## Overview: how the models work together

```
page photo ──► [no model] preprocessing (classical image processing)
            ──► [no model] line finding / reading order (geometry)
            ──► (2b) WordRecognizer reads each word image  ◄── trunk initialised from (1) LetterCNN
            ──► (4) Sindhi masked LM scores correction candidates
            ──► (5) Detector flags words that look wrong
            ──► text
```

Why several models: no single model is available for handwritten Sindhi. (1) is trained on the only large labelled
set (isolated letters) and gives the recognizer good visual features; (2) reads whole words, avoiding letter
segmentation; (4)–(5) use language knowledge the image model cannot have, to fix dot-level mistakes.

| # | Model | Kind | Parameters | Pretrained / from scratch | Saved at | Used at run time? |
|---|---|---|---|---|---|---|
| 1 | LetterCNN | ResNet-style CNN classifier, 52 classes (+3 auxiliary heads) | 2,817,140 | **from scratch** | `checkpoints/step2_cnn/best.pt` (11 MB) | No (only its trunk is reused) |
| 2a | WordRecognizer (original) | CNN + Transformer encoder–decoder | 8,100,976 (80 symbols) | trunk from #1, rest from scratch | `checkpoints/step2_recognizer/best.pt` (31 MB) | No (superseded) |
| **2b** | **WordRecognizer (fine-tuned)** | same | **8,102,002** (82 symbols) | fine-tuned from 2a | `checkpoints/step2_recognizer_ft/best.pt` | **Yes** |
| 3 | xlm-roberta-base (as copied) | multilingual Transformer masked LM | 278,885,778 stored | **pretrained** (Facebook AI), not trained here | `checkpoints/xlmr_base/` (1.1 GB) | No (start point + reference) |
| **4** | **Sindhi masked LM** | XLM-RoBERTa, continued pre-training | 278,295,186 stored, 43.4 M trained | further-trained from #3 | `checkpoints/step4_xlmr_mlm/best/` | **Yes** |
| **5** | **Wrong-word detector** | XLM-RoBERTa token classifier, 2 labels | 277,454,594 stored, ~42.5 M trained | fine-tuned from #4 | `checkpoints/step4_xlmr_detector/best/` | **Yes** |
| 6 | Tesseract 5.5.2 (`snd`, `urd`, `ara`, `script/Arabic`) | external LSTM OCR engine | — | pretrained, not trained here | system install | No — evaluated in Step 3; its `text2image` tool **rendered the printed training words** for #2 |
| 7 | rembg / U²-Net | salient-object segmentation | — | pretrained | `checkpoints/rembg/` (176 MB onnx) | No — evaluated in Step 1 and rejected |

Not implemented (mentioned only as future work): CTC line recognizer, TrOCR, beam search, the CRF POS tagger that came
with `pos_text` (never loaded), the old Keras CNN of an earlier prototype (not in this repository).

---

## 1. LetterCNN (`cnn_transformer/models.py::LetterCNN`, `Trunk`, `BasicBlock`)

- **Simple explanation:** looks at one 64×64 picture of a single letter and says which of the 52 letters it is.
- **Architecture:** stem Conv3×3 (1→32) + BatchNorm + ReLU → 4 stages of 2 residual blocks each (32, 64, 128, 256
  channels; strides 1, 2, 2, 2) → global average pool → dropout 0.3 → Linear 256→52. Each residual block: Conv3×3 →
  BN → ReLU → Conv3×3 → BN + shortcut → ReLU. 17 convolution layers. Training-only auxiliary heads: body shape (22),
  dot position (5), dot count (5).
- **Input:** 1×64×64 grayscale crop (ink bright, paper dark), aspect kept, 4 px margin. **Output:** 52 scores.
- **Activation:** ReLU. **Loss:** cross-entropy with label smoothing 0.1 + 0.3 × auxiliary cross-entropy.
- **Optimizer:** AdamW, lr 0.002, weight decay 0.0005; OneCycle schedule. **Batch:** 64.
- **Epochs:** up to 150 with early stopping (patience 30); chosen run `gray_aux_synth` stopped after 106 epochs,
  best epoch 76; ~21 min on the M2.
- **Data:** train 3,001 / val 367 / test 873 unique letter images (sheet-disjoint); augmentation (rotation ±12°,
  shear 10°, scale 0.8–1.1, translation, stroke width, elastic warps); 25 % of samples replaced by font-rendered
  letters (7 macOS fonts).
- **Metrics:** test accuracy 0.8144, **0.8225 with 5-view test-time augmentation**; macro-F1 0.8183; top-3 0.9393.
- **Loading:** only for training/evaluation (`scripts/step2_ensemble_eval.py`, `step2_error_analysis.py`); its trunk is
  copied into the recognizer by `load_trunk_from_letter_cnn` (126 tensors).

## 2. WordRecognizer (`cnn_transformer/models.py::WordRecognizer`)

- **Simple explanation:** looks at the picture of a whole handwritten word and writes it out letter by letter, like
  someone transcribing; it never needs to cut the word into letters.
- **Architecture:**

| Part | Details | Parameters |
|---|---|---|
| CNN trunk | same as LetterCNN (overall stride 8 in height and width) | 2,795,552 |
| collapse | Conv2d 256→256, kernel (8,1): turns the 8 feature rows into one; sequence length = width/8 | 524,544 |
| encoder | Transformer encoder, **2 layers**, d_model 256, **4 heads**, feed-forward 1024, dropout 0.1, pre-norm, sinusoidal positions, padding mask | 1,579,520 |
| embedding | character embedding (V × 256) + sinusoidal positions | 20,992 |
| decoder | Transformer decoder, **3 layers**, d_model 256, 4 heads, FF 1024, causal mask | 3,160,320 |
| output | Linear 256→V | 21,074 |
| **total (2b)** | | **8,102,002** |

- **Input:** word ink map, height 64 (52 px ink + padding), width ≤ 320 in training, flipped horizontally so right-to-
  left text runs left to right (`cnn_transformer/words.py::to_tensor_image`). **Output:** characters from an
  82-symbol vocabulary (`<pad> <bos> <eos> <unk>`, digits, punctuation, Sindhi letters, `/`, `؟`); max 24 label
  characters, decoding capped at 26 steps.
- **Activations:** ReLU (CNN and Transformer default). **Decoding:** greedy (no beam search).
- **Loss:** cross-entropy with label smoothing 0.1, teacher forcing, padding ignored.
- **Optimizer:** AdamW, weight decay 0.01; trunk learning rate × 0.3; gradient clipping 1.0; linear warmup then cosine.
- **Training run 2a:** lr 5e-4, warmup 750, **15 epochs × 900 steps**, effective batch 32 (2 × 16, gradient
  accumulation), gradient checkpointing; best epoch 13; **~3 h 51 min** (after three runs that crashed/ran out of
  memory). Mix per batch: printed 45 % (38,791 Tesseract `text2image` renders of 20,000 Wikipedia words in 8 fonts) /
  composed 35 % (words built from real train-split letter crops) / real 20 % (97 verified old-page crops).
- **Fine-tuning 2b:** lr 1e-4, warmup 200, **8 epochs × 900 steps**, best epoch 8, **~2 h 07 min**; mix 30/30/40 with
  1,201 real crops (1,104 batch-2 TRAIN + 97).
- **Validation:** printed_val (600), composed_val (600), + real_val (71 batch-2 VAL crops) for 2b; selection by mean 1 − CER.
- **Metrics (2b):** composed test 1 − CER 0.866 (gate); printed 0.977; real H/J CER 0.227; **unseen writer CER 0.088
  (1 − CER 0.912), 71.5 % words right**.
- **Loading for inference:** `run_pipeline.py::load_recognizer` → `scripts/step3_baseline.py::load_recognizer`
  (`RECOGNIZER = "checkpoints/step2_recognizer_ft/best.pt"`) → used by `tesseract_ft/cuts.py::read_groups`.

## 3. xlm-roberta-base (pretrained)

- **Simple explanation:** a large language model trained by Facebook AI on text in 100 languages (including Sindhi).
  It can guess a hidden word from its context; it cannot read images.
- 12 layers, hidden size 768, 12 heads, feed-forward 3072, GELU, vocabulary 250,002 (SentencePiece).
- Copied from the local Hugging Face cache on 2026-11-14 (the repository records no Hub id; the cache folder was
  `models--FacebookAI--xlm-roberta-base`). Weights file SHA-256
  `6fd4797bc397c3b8b55d6bb5740366b57e6a3ce91c04c77f22aafc0c128e6feb` (1,115,567,652 bytes; recomputed during this
  audit; compared with the Hub's published value at copy time — that comparison is not in code).

## 4. Sindhi masked LM (`scripts/step4_train_mlm.py`)

- **Simple explanation:** we continued XLM-R's training on Sindhi text so it knows Sindhi words and legal phrasing
  better.
- **Training:** whole-word masking of 15 % of words (80 % `<mask>`, 10 % random, 10 % unchanged); loss only at masked
  positions; AdamW (fused), lr 5e-5, weight decay 0, linear schedule, 130 warmup steps; batch 4 × 8 accumulation = 32
  chunks of ≤ 128 tokens; **1,300 steps (0.62 epoch), 1 h 36 min 43 s**.
- **Frozen to fit 8 GB:** embeddings (kept in bfloat16 during training, exact fp32 restored on saving) and the lower
  6 of 12 layers; 43.4 M parameters trained.
- **Data:** Step 4 TRAIN split: Wikipedia 243,553 sentences + legal 1,722 + POS 414 sentences (legal/POS ×20).
- **Validation:** legal/POS/Wikipedia val (293 chunks); loss 3.604 → **3.112** (perplexity 36.8 → 22.5). (The
  step-1300 evaluation said 3.078: masks are re-drawn at each evaluation.)
- **Inference:** `xlmr/lm.py::MaskedScorer`: replace a word by k `<mask>` tokens, sum the log-probabilities of the
  candidate's pieces.

## 5. Wrong-word detector (`scripts/step4_train_detector.py`)

- **Simple explanation:** reads a recognized sentence and marks each word as "probably right" or "probably wrong".
- XLM-R encoder (from #4) + dropout + Linear 768→2 per token; first sub-word piece of each word carries the label.
- **Training:** 30,000 sentences corrupted with the recognizer error model at 20 % of words (legal + POS TRAIN ×3,
  plus Wikipedia); AdamW (fused), lr 3e-5, weight decay 0, linear schedule with 6 % warmup; batch 8 × 4 = 32;
  **1 epoch = 938 steps, ~21 min**; embeddings and lower 6 layers frozen. Best checkpoint step 800, validation loss
  **0.1346**.
- **Inference:** `xlmr/lm.py::Detector.probs`; a word is flagged when P(wrong) > θ (from `val_comparison.json`).

---

## Algorithms (non-neural), where they are implemented

| Algorithm | Where | Role |
|---|---|---|
| Background estimation (morphological closing, median) + division normalisation | `preprocess/ops.py::estimate_background`, `flatten_illumination` | remove shadows / uneven light |
| Best-separating grey channel | `preprocess/ops.py::to_gray`, `ink_separability` | coloured paper / ink |
| Sauvola local threshold (scikit-image) | `preprocess/ops.py::binarize` | binarisation (chosen) |
| Otsu, adaptive threshold | `preprocess/ops.py::binarize` | compared in Step 1 |
| CLAHE, non-local means denoising (OpenCV) | `preprocess/ops.py::apply_clahe`, `denoise` | low contrast, noise |
| Projection-profile deskew | `preprocess/ops.py::estimate_skew` | straighten pages |
| Morphological line removal | `preprocess/ops.py::remove_lines`; `cnn_transformer/segment.py::erase_ruled_lines` | rulings / frames |
| Connected components, speck filter, paper mask | `preprocess/ops.py`; `tesseract_ft/pages.py::filter_comps` | clean-up |
| Line clustering (simple / tracked) | `cnn_transformer/segment.py::cluster_lines_simple`, `cluster_lines` | find lines |
| Union-find | `pipeline/layout.py::order_lines`; `xlmr/data.py::legal_splits` | rows of line pieces; grouping legal Q/A |
| Dynamic programming over span groupings | `pipeline/page.py::best_grouping`; `tesseract_ft/cuts.py::best_cuts`; `cnn_transformer/align.py::forced_align` | word grouping; aligning known text to ink |
| Gap rule | `tesseract_ft/boxes.py::own_layout`; `pipeline/page.py::line_units` | word cutting (`--words gaps`) |
| Hungarian assignment on IoU (SciPy `linear_sum_assignment`) | `tesseract_ft/boxes.py::score_boxes` | box evaluation |
| Levenshtein edit distance / alignment | `common/metrics.py::edit_distance`; `xlmr/errors.py::align_ops` | CER/WER; error model |
| Weighted edit distance (look-alike letter families) | `xlmr/correct.py::wdist`, `COSTS` | correction cost |
| Symmetric-delete lexicon lookup (SymSpell-style, up to 2 deletions) | `xlmr/correct.py::deletes`, `Lexicon.near` | fast candidate lookup |
| Masked-LM candidate scoring | `xlmr/lm.py::MaskedScorer.score` | correction ranking |
| Random search + coordinate refinement | `scripts/step4_compare.py`, `scripts/step3_levers.py` | weight tuning on validation |
| 4-gram overlap leakage filter | `tesseract_ft/texts.py::LeakFilter` | remove test sentences from training text |
| MD5 de-duplication | `scripts/prepare_splits.py` | unique letter images |
| Bootstrap 95 % interval over pages | `scripts/step6_final.py::bootstrap` | Step 6 uncertainty |
| Test-time augmentation (5 views) | `scripts/step2_ensemble_eval.py` | letter CNN evaluation |

**Documented but not found in code (NOT VERIFIED as code):** the "paired bootstrap, 5,000 resamples" mentioned in
`docs/STEP2_cnn_transformer.md`; the xlm-roberta-base SHA-256 comparison (done by hand; value recorded above).
**Stale comments found:** `cnn_transformer/models.py` header says "binary letter image" (the chosen CNN takes
grayscale); `docs/STEP2_cnn_transformer.md` says word width "≤ 512" (code uses 320).
