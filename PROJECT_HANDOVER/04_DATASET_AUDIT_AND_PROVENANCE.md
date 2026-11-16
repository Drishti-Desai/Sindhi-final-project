# 04 — Dataset audit and provenance

All counts below were **recomputed from the files on 2026-11-16** (MD5 of every file under `data/raw/`, every image
opened with Pillow). Full numbers, scripts and per-class tables: `appendix/B_raw_dataset_audit.md`. Where the
project's older docs say something different, this file is the corrected version.

Result in one line: **0 unreadable images, no train/test leakage found in any of four checks, but several datasets
have no recorded licence, and one test-page copy sits in an excluded folder.**

---

## 1. Dataset provenance table

| Dataset (folder) | Original source (as recorded) | Author / organisation | Licence | Script / language | Size verified | Used for |
|---|---|---|---|---|---|---|
| **letters52** (`data/raw/letters52/`) | Copied 2026-10-16 from the team's earlier project `SINDHI-PROJECT/data/alphabets_clean/`; filenames are Roboflow exports | **NOT FOUND** (original photographer/annotator not recorded) | **NOT FOUND** | Sindhi isolated letters (52 classes) | 18,086 JPG files, **4,241 unique** | **Training** letter CNN; composed words for the recognizer |
| **handwritten** (`data/raw/handwritten/`) | `SINDHI-PROJECT/input/` + `textextract/` (2026-10-16); `text_gt/` drafted by this project | the team's own handwritten pages (writers A, D, H, J) — **NOT VERIFIED** who wrote them | **NOT FOUND** (own data) | Sindhi page photos + text | 22 image/text pairs (+1 replaced photo), **402 words**, 65 text lines | **Training** (TESS_TRAIN word crops); TESS_TEST and **FINAL_TEST** for testing |
| **handwritten_batch2** (`data/raw/handwritten_batch2/`) | `~/Desktop/new data/` (2026-11-13) | own collection (writers A, H, J, several others, one "purple pen" writer) — **NOT VERIFIED** | **NOT FOUND** | Sindhi page photos + texts | 129 images (128 unique), 128 texts | **Training** (fine-tuning), VAL, unseen-writer test |
| handwritten_batch3 | `~/Desktop/new data/data/` (2026-11-14) | own | NOT FOUND | — | 17 images, 20 texts | **Not used** |
| **pos_handwritten** | Mendeley Data: M. Soomro, M. Khalid, "POS Tagging on Handwritten Sindhi Sentences", 2024, doi:10.17632/phk66sgmp5.1 (University of Sindh) | Soomro & Khalid | **CC BY 4.0** (recorded; online page NOT re-checked — no network during audit) | Handwritten Sindhi sentence images | **998 files, 976 unique** | **Not used** (no transcriptions) |
| **pos_text** | `~/Downloads/Sindhi-Pos-Tagging-main/` (a GitHub-style repo; its README has a placeholder URL) | **NOT FOUND** | **NOT FOUND** (README has none) | Sindhi POS-tagged sentences | 676 sentences → **518 clean**, 4,764 words | **Training text** (lexicon, masked LM) |
| **Wikipedia-sd** (`data/raw/corpus/wikipedia_sd/`) | Hugging Face `wikimedia/wikipedia`, config `20231101.sd` | Wikimedia / Wikipedia contributors | **CC BY-SA 3.0 + GFDL** | Sindhi text | **16,928 articles, 4,095,631 words**; sha256 matches the record | **Training text** (word lists, lexicon, masked LM) |
| **Sindhi_Legal_Dataset.csv** (read in place from `~/Downloads/`) | **NOT RECORDED in `data/PROVENANCE.md`** | **NOT FOUND** | **NOT FOUND** | Sindhi legal Q&A text | **1,060 rows × 2 columns** (`input_sd`, `output_sd `) | **Training text**; also the source of the batch-2 page texts |
| roboflow_sindhi_ocrf | `~/Downloads/Sindhi OCRf.v1i.coco/`; `universe.roboflow.com/iitbresearchwork/sindhi-ocrf` | "iitbresearchwork" workspace on Roboflow | **CC BY 4.0** (stated in its README) | **printed** Sindhi book pages (some Devanagari) | 509 images (640×640, stretched), 113,017 word boxes, no text | Only Step 1 preprocessing checks; **not used** for training |

**Action needed:** record the origin and licence of the legal CSV, the letters52 photos and the pos_text corpus
before publishing anything derived from them (see `10_GITHUB_PUBLICATION_GUIDE.md`).

---

## 2. Per-dataset details

### letters52 (isolated letters)
- Files: 18,086 JPG in old folders `train/` 15,607, `valid/` 2,124, `test/` 355 (+ empty `train 2/`). The docs'
  18,088 is before two mislabeled files were deleted (recorded in PROVENANCE).
- **Duplicates:** 13,845 files (77 %) are byte-identical copies of others (old oversampling); 3,145 duplicate groups.
  **4,241 unique images.** No image has two different labels.
- Classes: 52, unique images per class 73–87 (ratio 1.19 → **balanced**).
- Image size (median) 57 × 36 px; 5th–95th percentile height 21–73 px; **448 files are under 20 px tall**; one
  degenerate **3 × 1 px** crop (train split, class 19). (Docs saying "22–77 px tall" describe a 400-crop sample.)
- Annotation: class = folder name, mapped to letters by `configs/letter_map.yaml` (7 classes confirmed by the owner,
  the rest a draft — **NOT VERIFIED** by a Sindhi expert).
- **Split (by source sheet, seed 42):** train 3,001 images / 88 sheets, val 367 / 11, test 873 / 23. **0 sheets and
  0 image hashes in two splits.** All 52 classes in every split.
- Preprocessing: Step 1 glyph profile (`data/preprocessed/letters52/`, grayscale variant `letters52_gray/`).

### handwritten (22 page photos, the original pages)
| Split | Pages | Writers | Text lines | Words |
|---|---|---|---|---|
| TESS_TRAIN | 12 (A1–A5, D1, D1., D3, D4, D5, H2, H4) | A, D, H | 31 | 194 |
| TESS_TEST | 3 (H1, J2, J3) | H, J | 11 | 61 |
| **FINAL_TEST** | 7 (D2, D6., H3, H5, J1, J4, J5) | D, H, J | 23 | 147 |
| total | 22 | 4 | 65 | **402** (older docs said ~450–470) |
- Images 944–4032 px wide (PNG, 13 RGBA + 9 RGB). Text: `text_original/` (source) and `text_gt/` (re-broken to
  match the image; marked "needs user verification", 19 open rows in `REVIEW.csv`).
- Word-level ground truth: `data/splits/step2_real_words/verified.csv` — 97 verified TESS_TRAIN + 56 verified
  TESS_TEST word crops (cut with the known text and checked).
- `text_gt/D1.txt` and `D1..txt` are byte-identical (two photos of the same text — **NOT VERIFIED** as intended).

### handwritten_batch2
- 129 images (PNG, 180–4284 px wide). Manifest status: ok 116, text_mismatch 5, ok_text_corrected 1,
  duplicate_no_text 7.
- **Pages per split:** TRAIN 103 (98 with word crops + 5 text mismatches), VAL 12, TEST_UNSEEN_WRITER 7
  ("purple pen" writer only), EXCLUDED 7 (copies of writer D's pages).
- **Word crops (`data/splits/handwritten_batch2/words.csv`, 1,624 rows):** TRAIN 1,368 (1,104 verified), VAL 110 (71),
  TEST_UNSEEN_WRITER 146 (144). Only verified crops are used. (DECISIONS.md's 157 test words is wrong: 146.)
- 238 stale crop files in `data/preprocessed/handwritten_batch2_words/` are not in `words.csv` (unused).
- **Portability bug:** 33 files are `H102.PNG`–`H134.PNG` on disk but `.png` in the manifest; works on macOS only.

### pos_handwritten (Mendeley) — verification of the "998 images / 22 duplicate groups" claim
- **998 images: confirmed.** **22 exact-duplicate groups (pairs): confirmed** (`N.jpg` = `xyz (N−7).jpg` for
  N = 90…113), so **976 byte-unique** images.
- **Near duplicates: 2 pairs, not 1** — `20240616_143013.bmp` ≈ `.jpg` (recorded) and **`sdf (66).jpg` ≈
  `xyz (1).jpg`** (same scan, cropped differently; checked by eye) → about **974 distinct photos**. Further near
  duplicates beyond the 12 closest pairs inspected are NOT VERIFIED.
- 56 `.png` files are really JPEG; one 1-bit BMP. Sizes 644–2914 × 104–707 px (sentence/line images).
- **No transcriptions exist** in the folder. `data/splits/pos_handwritten/images.csv` holds machine drafts and
  nearest-text matches (median CER 0.69), **not labels**.
- No image equals any image in our other datasets.

### Text corpora
- pos_text: 6,869 token rows (CoNLL-like: ID, FORM, LEMMA, UPOS, XPOS, FEATS, Pronunciation), 676 sentences, 18 UPOS
  tags; cleaned to 518 unique sentences (`data/preprocessed/pos_text/sentences_clean.txt`).
- Wikipedia-sd: 16,928 articles, 4.10 M words, 20.07 M characters.
- Legal CSV: 1,060 Q/A rows; 8,749 question words, 16,802 answer words; 1 fully duplicated row.
- Step 4 text splits (`data/splits/step4/`): legal 1,722 / 215 / 169 lines, POS 414 / 52 / 52, Wikipedia
  243,553 / 1,001 / 1,037. Lexicon `data/preprocessed/step4/lexicon.tsv`: 47,422 words.
- Exact-line overlaps between Step 4 splits: legal val∩train 1 line; Wikipedia test∩train 20, val∩train 45 lines
  (probably short repeated lines such as headings — NOT VERIFIED). Small effect on the synthetic Step 4 numbers.

---

## 3. Ground truth: what exists

| Level | Exists? | Where | Amount |
|---|---|---|---|
| Isolated letters | yes | letters52 folders | 4,241 unique images, 52 classes |
| **Handwritten words** | **yes (small)** | `words.csv` (batch 2), `verified.csv` (old pages) | 1,319 batch-2 verified crops + 153 old-page verified crops |
| Handwritten lines / pages | yes | `text_gt/`, batch-2 texts (`words.csv` `line` column) | 22 old pages (65 lines) + 122 batch-2 texts (276 lines with word crops: TRAIN 234, VAL 23, unseen writer 19) |
| Handwritten sentences (Mendeley) | **no** | — | 976 images without text |
| Printed words | synthetic | `data/synthetic/step2/printed/` (renders of corpus words) | generated |

---

## 4. Quality and leakage checks

| Check | Result |
|---|---|
| Unreadable images | **0** in every dataset |
| Exact duplicates | letters52: 77 % of files (removed before splitting); batch 2: 1 pair (`D3` = `D4`, both EXCLUDED); Mendeley: 22 pairs |
| Near duplicates | Mendeley: 2 pairs found |
| Class imbalance | letters52 balanced (73–87 unique per class) |
| Incorrect labels | 2 mislabeled letters deleted by the owner (recorded); `letter_map.yaml` partly unconfirmed; `text_gt` drafts flagged for review |
| Missing annotations | Mendeley images (no text); 3 batch-3 texts without images |
| **(a)** letter test vs train/val | **no leakage** (0 shared hashes or sheets) |
| **(b)** unseen-writer test vs train/val | **no leakage** (no shared pages, writers, images or text lines) |
| **(c)** FINAL_TEST elsewhere | **no leakage in use**, but `handwritten_batch2/…/D6..png` is a byte-identical copy of FINAL_TEST `D6.` and batch-2 `D2.png` is an older photo of FINAL_TEST D2; both are marked EXCLUDED / "never use" and nothing reads them — a **latent hazard** for future code that loads the whole folder. Writers H and J appear in both FINAL_TEST and training (documented: "seen writers, new text"). |
| **(d)** Mendeley vs our images | no matches |
| Text leakage | 0 of 23 FINAL_TEST lines appear in any training text (one single-word substring only) |

---

## 5. Public vs personally collected

| Public (third party) | Personally / team collected | Unknown |
|---|---|---|
| Wikipedia-sd, Mendeley pos_handwritten, Roboflow sindhi-ocrf | handwritten pages (A, D, H, J), batch 2, batch 3 (by the owner's account; writer identities **NOT VERIFIED**) | letters52 (origin beyond SINDHI-PROJECT not recorded), pos_text, legal CSV |

**Datasets required to retrain:** letters52, handwritten (+ `text_gt`), handwritten_batch2, pos_text, Wikipedia-sd,
the legal CSV. **Required to run the app:** none of the raw datasets (only `lexicon.tsv` and the A1 example image).
