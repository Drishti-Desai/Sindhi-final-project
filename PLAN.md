# Sindhi OCR — Project Plan (Step 0)

Status: Step 1 done (gate PASS). **Step 2 accepted by the user with its gates FAIL** (letter CNN 0.8225; word recognizer composed 1-CER 0.866 < 0.90), using the fine-tuned recognizer `checkpoints/step2_recognizer_ft/best.pt`. Reference number for real handwriting: unseen writer 1-CER 0.912, 71.5% words correct (docs/STEP2_cnn_transformer.md section 8). **Step 3 accepted by the user with its gate FAIL** (word-box F1 0.809 < 0.90); Step 5 box finder: our own gap rule (unseen writer 0.862). **Step 4 accepted by the user with its gate FAIL** (Sindhi-trained XLM-R + detector, chosen on validation: domain test word accuracy 0.890 < 0.90, 2.07% of correct words changed > 2%); on real handwriting the unseen writer goes 71.5% → 79.2% words correct (3 of 103 correct words changed). Known limitation: too many changes on non-legal text (Wikipedia 4.3%). The pipeline uses this final variant (docs/STEP4_xlmroberta.md section 10). **Step 5 accepted by the user with its gate FAIL** (run_pipeline.py with reading-order rows and span-level recognizer joins, chosen on VAL: unseen writer's pages 1 − CER 0.778 < 0.90, worse than the Step 4 gap-rule reading 0.807; VAL pages CER 0.462 → 0.246; the joins over-join on writers with clear gaps). Default `--words joins`; `--words gaps` for writers with clear spacing (docs/STEP5_pipeline.md section 6). **Step 6 accepted by the user with its gate FAIL** (FINAL_TEST, whole page, measured once: 1 − CER −0.650; writer D's two pages produce 306 words for 42 because rulings, pencil fragments and speckle are read as text; writers H + J: CER 0.354). The project closes as it stands (docs/STEP6_testing.md section 6); future work in the README.
Date of inspection: 2026-10-16.

## Decisions and updates (2026-10-17)

- Letter map: user verified classes 7, 8, 20, 21, 22, 23, 50; the other classes keep the draft (`configs/letter_map.yaml`).
- The user's fixes were made in the NEW copies (2 mislabeled letters deleted, A1.txt corrected, D4.png replaced), not in the originals. NEW is the source of truth; a re-copy would undo them (`data/PROVENANCE.md`).
- Letters: **4,241 unique images** (73–87 per class). Split by sheet, seed 42: train 3,001 / val 367 / test 873.
- D4 is now a valid pair (new photo matches its text). It is a separate story (G12), assigned to **TESS_TRAIN**. D3 and D4 are no longer duplicates.
- Step 2: word-image recognizer approved; the 52-letter CNN is trained and gated first and used as the backbone.
- Step 3 gate: option (b), word/line **box quality**, with text CER/WER also reported.
- Roboflow `sindhi-ocrf` (copied from ~/Downloads): 509 **printed** book pages (Sindhi Bhagavad Gita, some Devanagari), stretched to 640×640, about 113k **word boxes**, **no text labels**, one usable class (named `json`), CC BY 4.0. No overlap with letters52 (hash or filename). It can serve as box ground truth for *printed* text and as detector training data, but not as handwriting box ground truth.
- Wikipedia-sd downloaded: 16,928 articles, ~4.1M words. Note: it writes aspirated letters with `ھ` (U+06BE) where the handwritten texts use `ه` (U+0647), so a normalisation rule is needed in Step 4.
- `text_gt/` drafted for all 22 pairs, with 19 review flags (`data/raw/handwritten/text_gt/REVIEW.csv`). **Needs user verification before Step 3 measures CER.**

---

## Data available for future work (2026-11-14)

- **`data/raw/pos_handwritten/`**: 976 unique images of handwritten Sindhi sentences by many student writers
  (Soomro & Khalid, Mendeley Data, doi:10.17632/phk66sgmp5.1, CC BY 4.0). **No transcriptions exist** (checked
  against every text we have). They are the largest source of new writers we have, and usable as soon as someone
  transcribes them. Recognizer drafts exist (`data/splits/pos_handwritten/images.csv`) only as a typing aid:
  **never train on unverified drafts.** Estimated effort: 40–60 s per image, roughly 10–16 h for all.

---

## 1. Dataset findings

All originals were read only. Copies live in `data/raw/` (see §1.6).

### 1.1 52 Sindhi letters — `SINDHI-PROJECT/data/alphabets_clean` → `data/raw/letters52/`

| Item | Finding |
|---|---|
| File type | JPG, RGB, **very small** (typical 40–65 × 29–40 px) — photo crops from Roboflow exports (`<photo>_jpg.rf.<hash>_<n>.jpg`) |
| Files on disk | 18,088 (train 15,607 / valid 2,124 / test 357) plus an empty `train 2/` folder |
| **Unique images** | **4,243** — 13,845 files (77%) are byte-identical copies (`_1`, `_2`, `(1)` suffixes; old oversampling). No exact duplicate crosses splits; no duplicate has conflicting labels. |
| Classes | Exactly 52 folders `1..52`, present in every split |
| Balance (unique) | 73–87 per class (mean 81.6) → balanced |
| Source photos | 122 sheets (81 `IMG-20250418-WA*`, 41 `WhatsApp-Image-2025-08-23*`), 8–53 crops each. Each sheet ≈ one person writing the alphabet. |
| Existing split | Test = 12 whole sheets (357 imgs, 2%), source-disjoint from train. Train/valid share 85 sheets. |
| Label → glyph map | **Missing.** Folders are numbers only. I inferred the map visually (§1.1.1) — **needs your confirmation.** |
| Label noise | Visible in montages (e.g. some class-25 `ر` crops look like `ذ`; class-36 contains a `غ`-like crop). |
| Prior result | Old Keras CNN: **65.5–67.2 % accuracy** on the clean test (TRANSFER_NOTES.md). |

#### 1.1.1 Draft class-ID → letter map (inferred from montages — please verify)

The order matches the traditional Sindhi alphabet including the digraphs `جھ` and `گھ`, **without `ژ`**:

```
 1 ا   2 ب   3 ٻ   4 ڀ   5 ت   6 ٿ   7 ٽ   8 ٺ   9 ث  10 پ
11 ج  12 ڄ  13 جھ 14 ڃ  15 چ  16 ڇ  17 ح  18 خ  19 د  20 ڌ
21 ڏ  22 ڊ  23 ڍ  24 ذ  25 ر  26 ڙ  27 ز  28 س  29 ش  30 ص
31 ض  32 ط  33 ظ  34 ع  35 غ  36 ف  37 ڦ  38 ق  39 ڪ  40 ک
41 گ  42 ڳ  43 گھ 44 ڱ  45 ل  46 م  47 ن  48 ڻ  49 و  50 ه
51 ء  52 ي
```
Least certain (similar dot patterns at low resolution): 7/8 (`ٽ`/`ٺ`), 20/21 (`ڌ`/`ڏ`), 22/23 (`ڊ`/`ڍ`), 50 (`ه` vs `ھ`).

**Characters in real text that the 52 classes cannot produce:** `آ ئ ۾ ۽`, diacritics (`ِ َ ُ`), digits, punctuation (`، . ( ) : -`). `۾` alone occurs 18× in the handwritten texts. The recognizer must get extra classes for these (see §4, Step 2).

### 1.2 Handwritten pages — `SINDHI-PROJECT/input` + `textextract` → `data/raw/handwritten/`

Note: the folder is `input/` (singular), not `inputs/`.

| Item | Finding |
|---|---|
| Pairs | 22 PNG ↔ 22 TXT, matched by filename stem (`A1.png` ↔ `A1.txt`); two odd names `D1..png`, `D6..png` (double dot, kept as-is) |
| Images | 15 RGBA + 7 RGB; mostly 1240–1668 px wide × 219–665 px tall; `D2.png` is a 4032×3024 phone photo of faint pencil |
| Writers (by prefix) | A (blue pen, ruled paper), D (pencil/ink, mixed), H (pencil on grey paper with drawn frame), J (blue pen, some redaction blobs) |
| Text | UTF-8, 1–5 lines per file, **~470 words / ~2,000 chars total, ~60 lines** |
| Script | Proper Sindhi code points (`ڪ` U+06AA, `ي` U+064A, `ه` U+0647, `۾` U+06FE, `۽` U+06FD), Latin digits, some diacritics, one tatweel |

#### 1.2.1 Ground truth or OCR output? → **Neither, strictly: it is the *source text* the writers copied by hand.** Treat it as near-ground-truth that needs a small correction pass.

Evidence it is **not** OCR output:
- Contains web-page boilerplate that a person copy-pasted: `H2` ends with `- سنڌ ٽي وي نيوز _ سنڌ ٽي وي نيوز` (a browser-tab title "… – Sindh TV News _ Sindh TV News"); `D5` is `(گهربل)(ڇاپيو نه ويندو)` = "(required) (will not be published)", a website comment-form label. The writers copied these onto paper, so the images contain them too.
- Typographic details Tesseract does not emit: tatweel + damma `شـُڪلا`, fatha in `کڏڙَي`, kasra in `راجيوِ`; consistent Sindhi-specific code points; no OCR-style confusions (dot-count swaps, broken words).
- Bidi copy-paste artefacts: full stop at the start of the line (`.ايف آءِ اي …`), trailing/leading spaces, staircase indentation in J3–J5.
- A typo that the writer reproduced: `عرفام` (A3) vs `عرفان` (A4) — visible in both image and text.

Evidence it is **not human-verified against the images**:
- **A1:** image line 1 is `ميچ فڪسنگ جي الزام ۾`; text line 1 is a copy of line 3 (`جتي هو بيان رڪارڊ ڪرائيندا`).
- **D4:** `D4.png` is byte-identical to `D3.png` (same MD5), but `D4.txt` is a different sentence (`سرفراز احمد …`). Broken pair.
- **Line breaks** often do not follow the image (H1–H5 are one text line but 2 image lines; D2 differs).
- **D5:** phrase order differs from the image (RTL paste issue).
- Commas written on the page (e.g. line ends in A1/A2) are missing from the text.

Consequence: usable as **ground truth after a correction pass** (fix A1, drop D4 image, re-break lines to match the image, fix D5 order). I can draft the corrections from the images, but **a Sindhi reader (you) should verify them** — I should not be the only judge of handwriting. It is **not** a source of real OCR errors; real OCR errors will come from Steps 2–3 output.

#### 1.2.2 Duplicated / overlapping content (important for leakage)

| Story group | Images | Note |
|---|---|---|
| G1 match-fixing / FIA | A1 A2 A3 A4 A5 D1 D1.. H2 | same story, heavy phrase overlap; D1 and D1.. = same text, two different photos |
| G2 Supreme Court contempt | D6.. J1 | same sentence |
| G3 Dharamshala cricket | J2 J3 | J3 extends J2 |
| G4 Sanghar court | J4 J5 | heavy overlap |
| G5 U19 World Cup | D2 | |
| G6 NZ ODI | D3 | (D4 image = duplicate of D3 → excluded) |
| G7 web form label | D5 | not a sentence |
| G8–G11 | H1, H3, H4, H5 | one each |

Splits must be **by story group**, never by image, or near-identical sentences leak across splits.

### 1.3 "Tesseract test data" — **not found**

Searched `~/Desktop` (6 levels) for `*tess*`, `*.box`, `*.gt.txt`, `*.lstmf`, `*.tif`, `ground*truth*`:
- `~/Desktop/SINDHIMP/data/ground_truth/` — **empty**
- `~/Desktop/SINDHIMP/data/test_images/sample.jpg` — one 67×46 image, no label
- `SINDHIMP/data/alphabets` — another copy of the same letter set (23,439 jpg)
- `SINDHIMP/data/nlp/pos_handwritten` — placeholder CSV with only a header row; the referenced "POS Tagging on handwritten sindhi sentences" dataset is not on the Desktop

→ As you suggested, the Tesseract data will be **the handwritten pairs from 1.2**, split by story group (§2).

### 1.4 Public Sindhi text corpus (nothing downloaded yet)

| Source | Size | License | Notes |
|---|---|---|---|
| `wikimedia/wikipedia` `20231101.sd` (HF) | 17.6 MB parquet | CC BY-SA 3.0 + GFDL | Clean, encyclopedic; small. **Recommended first.** |
| `HuggingFaceFW/fineweb-2` `snd_Arab` (HF) | 414 MB parquet (train) | ODC-By 1.0 | Filtered web text, includes news sites (domain match). **Recommended: stream only ~50–100 MB.** |
| CC-100 `sd.txt.xz` (data.statmt.org) | 67 MB compressed | No formal license; Common Crawl terms; "no IP claims" | XLM-R was pretrained on this exact data, so it adds little. Not on HF as a config. |
| `DanishMahdi/snd-headline-ocr` (HF) | ~1 GB, 10K–100K samples | Apache-2.0, gated (contact-info agreement) | **Synthetic printed** TV-headline images + text. Useful for printed Tesseract fine-tuning, not handwriting. Optional. |

**Leakage hazard:** the handwritten texts are real Sindhi news headlines, so web corpora may contain those exact sentences. Any corpus sentence with high n-gram overlap with the Step-6 test texts will be removed before training.

### 1.5 Roboflow `iitbresearchwork/sindhi-ocrf` (optional)

The page returns HTTP 403 without login and no `.env` / `RF_API_KEY` exists yet, so contents are unverified. Caution: the letter set's filenames already carry Roboflow export hashes, so this project **may be the same source** as `alphabets_clean`. If you add a key, it will be de-duplicated by image hash and source-photo name before use.

### 1.6 What was copied into NEW

| From (read only) | To |
|---|---|
| `SINDHI-PROJECT/data/alphabets_clean/` (incl. `split_metadata.csv`, `duplicate_groups.json`, excl. `.DS_Store`) | `data/raw/letters52/` (18,088 jpg, 84 MB) |
| `SINDHI-PROJECT/input/` | `data/raw/handwritten/images/` (22 png) |
| `SINDHI-PROJECT/textextract/` | `data/raw/handwritten/text_original/` (22 txt) |

`data/raw/` and `data/preprocessed/` are git-ignored; manifests and splits are tracked.

---

## 2. Splits (decided up front, recorded in `data/splits/` at the start of Step 1)

### 2.1 Letters (Step 2)
De-duplicate to 4,243 unique images → **80/20 split grouped by source photo (sheet)**, seed 42, stratified so both photo batches (IMG-2025-04 / WhatsApp-2025-08) appear in test. About 98 sheets / ~3.4k images train, ~24 sheets / ~850 images test. Validation = ~10% of training *sheets*.
Why grouped: a random image-level split puts the same writer's other letters (same pen, paper, lighting) in train and test, which inflates accuracy. I'm deliberately not reusing the old 2% test split (too small for stable numbers).

### 2.2 Handwritten pairs (21 usable images; D4 excluded)

| Split | Story groups | Images | ≈ words | Used by |
|---|---|---|---|---|
| **FINAL_TEST** (Step 6 only, never trained or tuned on) | G2, G4, G5, G9, G11 | D6.., J1, J4, J5, D2, H3, H5 (7) | ~150 | Step 6 only |
| **TESS_TRAIN** | G1, G6, G7, G10, G12 | A1–A5, D1, D1.., H2, D3, D5, H4, D4 (12) | ~240 | Tesseract fine-tune; XLM-R real-text train |
| **TESS_TEST** | G3, G8 | J2, J3, H1 (3) | ~60 | Tesseract after-FT eval; XLM-R real-text test |

- Tesseract "before" (no fine-tune) is also measured on all 14 TESS_* images, since no training is involved and a bigger sample is more reliable.
- XLM-R may only see TESS_TRAIN / TESS_TEST texts plus `D4.txt` (text-only), never FINAL_TEST text.
- Public corpus and word lists are filtered against FINAL_TEST sentences (n-gram overlap).
- **Honest caveat:** 3 test images (~60 words) gives very noisy Tesseract metrics (±10 points easily). 21 images is far too few to train any handwriting recognizer; they are an evaluation set, not a training set.

---

## 3. Folder structure

```
NEW/
  README.md  PLAN.md  requirements.txt  .env.example  .gitignore
  run_pipeline.py
  configs/   paths.yaml (single source of all paths), step2_cnn.yaml,
             step3_tesseract.yaml, step4_xlmr.yaml, pipeline.yaml, letter_map.yaml
  docs/      STEP1_preprocessing.md … STEP6_testing.md
  data/
    raw/<dataset>/              letters52/, handwritten/{images,text_original,text_gt}/, corpus/
    preprocessed/<dataset>/     same filenames/labels as raw
    splits/<step>/              CSV manifests (file, label, group, split)
    synthetic/                  generated word/sequence samples (Step 2/4)
  common/    paths.py (loads paths.yaml), gates.py (read/write/check gates), metrics.py (CER/WER)
  preprocess/  cnn_transformer/  tesseract_ft/  xlmr/  pipeline/
  scripts/   step1_preprocess.py … step6_evaluate.py
  checkpoints/<step>/   logs/<step>/   outputs/<image_name>/
  results/   gates/ step1/ step2/ step3/ step4/ step6/
  tests/
```
Additions to your layout: `common/` (shared path/gate/metric code so every step measures CER identically), `configs/letter_map.yaml`, `data/synthetic/`, `text_gt/` (corrected transcriptions, alongside untouched originals).

---

## 4. Tech choices and how each step will work

| Area | Choice | Reason |
|---|---|---|
| Python | 3.13.3 in `.venv` | Already installed; PyTorch / transformers support it |
| Images | OpenCV, scikit-image (Sauvola), Pillow | Standard, fast on CPU |
| Segmentation comparison | `rembg` (U²-Net, onnxruntime) | Option (b) in Step 1; model ~170 MB, downloaded on first use (will ask) |
| DL | PyTorch with MPS (Apple M2), CPU fallback | Your hardware: M2, **8 GB RAM** |
| NLP | `transformers` Trainer + `datasets`, `xlm-roberta-base` (~1.1 GB) | As requested |
| OCR | Tesseract **5.5.2 (verified installed)**, `pytesseract`; `snd`, `ara`, `urd`, `script/Arabic` present; `lstmtraining` / `text2image` present | |
| Metrics | own CER/WER on NFC-normalised text (Levenshtein), reported raw and "normalised" (punctuation/whitespace-insensitive) | Same function used everywhere |

**Step 1 — preprocessing.** Compare (a) background estimation (large median blur / morphological closing) + division normalisation + Sauvola/Otsu vs. (b) rembg. My expectation, to be proven on samples: (a) wins for paper photos. rembg is built for foreground *objects* and tends to treat the whole sheet as foreground or erase thin strokes and dots. Evidence = dot/stroke preservation measured on letter crops plus a visual before/after grid. Note: the letter crops are tiny (≈50×30), so they get a gentler path (upscale before thresholding, no aggressive denoise that would erase dots).

**Step 2 — CNN + Transformer decoder.**
- CNN: a small ResNet-style net on 64×64 grayscale, strong augmentation (elastic, stroke-width, rotation, blur). Gate = top-1 accuracy on the sheet-disjoint test set.
- Decoder training (no real letter sequences exist): synthetic sequences. Sample a word from a Sindhi word list (corpus), pick a **test-split** (for eval) / **train-split** (for training) image per letter, run the CNN, and feed the per-letter feature vectors to a Transformer decoder that outputs Unicode text (and learns `جھ`→2 code points, spelling priors, etc.). Gate = 1−CER on synthetic test sequences built only from test-split letter images and held-out words.
- **The cursive problem, honestly:** isolated-letter training will *not* transfer well to real handwriting. Inside words letters take initial/medial/final forms, join, and lose their tails; dots shift between letters. More importantly, nothing reliably cuts cursive words into letters (next section). A 90% gate on synthetic sequences of isolated letters says little about real pages.
- **My recommendation (needs your OK):** keep the CNN + Transformer-decoder architecture but feed it **word images** (CNN encoder over the whole word → Transformer decoder emits characters; TrOCR-style, segmentation-free). Training data: synthetic word images rendered with Sindhi/Naskh fonts (incl. handwriting-style fonts where licensed) + words assembled from the isolated-letter crops + heavy augmentation. Real word crops come from TESS_TRAIN images only. The 52-letter classifier is still trained and gated first and becomes the encoder's pre-trained backbone, so it stays a proven component.

**Step 3 — Tesseract.**
- What `image_to_boxes` can and cannot give: with the LSTM engine, character boxes for Arabic-script text are **approximations** derived from where the LSTM emitted each character, not real glyph segmentation. In cursive they often overlap, collapse to slivers, or are missing, and when recognition is wrong the boxes are wrong too. Connected-component splitting gives *sub-word ligature pieces* plus separate dots, not letters. **Neither gives clean letter crops of Sindhi handwriting.** Realistic approach: use `image_to_data` (and/or my own projection-profile line/word segmentation) for **line and word boxes**, which are much more reliable.
- Expectation: `snd` on handwriting will be poor (I expect CER well above 50%). Fine-tuning on 11 images is unlikely to reach 90%. Fine-tuning needs `tessdata_best/snd.traineddata` (~10–20 MB download; the installed model is an integer "fast" model, which cannot be fine-tuned).

**Step 4 — XLM-RoBERTa.**
- Formulation: XLM-R cannot generate text, so correction = **candidate generation + MLM scoring.** (1) Flag suspicious words (not in the lexicon, or a low masked-LM pseudo-log-likelihood). (2) Generate candidates from a corpus lexicon by weighted edit distance using a Sindhi visual-confusion table (`ب/ٻ/ڀ/پ/ت/ٿ/ٽ/ٺ/ث` dot variants, `ڪ/ک`, missing `۾` dots, merges/splits at spaces). (3) Re-rank candidates by masked-LM score plus edit cost. (4) Fine-tune XLM-R as MLM on Sindhi text, plus a token-classification head for "is this token wrong", both via Trainer.
- Metric: on corrupted held-out sentences at a **fixed corruption level of 20% of words** (so input word accuracy ≈ 80%), gate = **word accuracy of the corrected output ≥ 90%** *and* false-correction rate on clean words ≤ 2%. Also report CER/WER before vs. after. Without the fixed corruption level the gate could pass trivially (the input alone is already ~90% correct).
- Synthetic-only accuracy **will overstate real accuracy**, because real OCR errors are burstier and stranger than synthetic ones. Check: re-run on the real Step 2/3 outputs for TESS_TEST once available, reported separately.
- Memory: fine-tuning all 278M parameters with AdamW will not fit comfortably in 8 GB. Plan: **freeze the 192M-parameter embedding matrix** (250k vocab × 768), train only the transformer layers, sequence length 64–128, batch 8 with gradient accumulation. If that still fails, train on Colab/the GPU PC.
- Rule-based / finite-state check: probably useful for *hard* constraints, e.g. `ھ` only after aspirable consonants, no word-initial `ڻ`, digits/punctuation untouched. **Only if you approve.**

**Steps 5–6** as you specified. Reading order: lines by y-centre clustering, words right-to-left by x, paragraph breaks from line-gap statistics.

---

## 5. Gates

- `common/gates.py` writes `results/gates/stepN_<name>.json`:
  `{"step":..., "metric":..., "value":..., "threshold":0.90, "status":"PASS|FAIL", "test_set":..., "n":..., "git_commit":..., "timestamp":...}`
- Every script calls `require_gate("stepN-1")` on start and exits with a clear message if the gate is missing or FAIL.
- The threshold lives in config but is **never lowered by code**. A FAIL stops the step, and I bring you error analysis plus options.
- Step 5 requires the Step 2, 3 and 4 gates = PASS.

---

## 6. Risks, honestly

1. **Step 3 gate (Tesseract ≥ 90% on handwriting) will almost certainly FAIL**, even after fine-tuning. Handwritten Sindhi is far from Tesseract's printed-text training, and 11 training images is tiny. Decision needed: (a) keep the text-accuracy gate and accept it blocks Step 5; (b) gate Tesseract on its *actual pipeline role*, i.e. **line/word box quality** (e.g. word-box recall at IoU ≥ 0.5 ≥ 90%, measured against boxes I annotate on TESS_TEST), with text accuracy still reported; or (c) gather more handwritten line data.
2. **Step 2 CNN ≥ 90%** is uncertain but plausible: the old CNN got ~66%, but it used RGB 64×64 with no preprocessing, light augmentation and noisy duplicates. ~65 unique training images per class plus label noise and dot-only distinctions set a ceiling. If it plateaus, options are label cleaning, merging/hierarchical dot heads, or more data.
3. **End-to-end (Step 6) on handwriting will be far below 90%** with this data volume. Realistically CER might be 30–60%. The pipeline will be correct and honestly measured, but the data limits it. Biggest lever: more transcribed handwritten lines.
4. Ground-truth text needs your verification (§1.2.1).
5. Unicode variants (`ه`/`ھ`, `ك`/`ڪ`, `ی`/`ي`, `ۀ`) will be NFC-normalised plus a fixed Sindhi mapping, documented in `configs/`.
6. 8 GB RAM: XLM-R fine-tuning is tight (mitigation in §4); 32 GB free disk is enough.
7. Leakage: handled by story-group splits, sheet-grouped letter splits, and corpus filtering. Step 6 reports any suspiciously good numbers.

---

## 7. Decisions needed from you before Step 1

1. Confirm or correct the **class-ID → letter map** (§1.1.1), especially 7/8, 20/21, 22/23, 50.
2. Approve the **handwritten split** (§2.2) and that I draft corrected transcriptions in `text_gt/` for you to verify.
3. Approve the **word-image recognizer** recommendation for Step 2 (vs. strictly letter-crop decoding).
4. Choose the Step 3 gate handling (§6 risk 1: a / b / c).
5. Corpus: OK to download Wikipedia-sd (17.6 MB) now-ish, and a ~100 MB FineWeb-2 `snd_Arab` sample at Step 4?
6. Roboflow: add `RF_API_KEY` to `.env` if you want it inspected.
7. Git identity for commits (no `user.name`/`user.email` configured on this Mac).
