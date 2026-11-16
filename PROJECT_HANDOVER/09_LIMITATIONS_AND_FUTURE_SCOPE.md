# 09 — Limitations and future scope

The two halves of this file are kept strictly apart. **Part A lists what is true of the current implementation**
(evidence in `docs/STEP*.md` and `results/gates/*.json`). **Part B lists proposals that are NOT implemented.**

---

## Part A — Current limitations (implemented system, as measured)

| # | Limitation | Evidence |
|---|---|---|
| 1 | **Accuracy is far below the 90 % target.** Final test (7 held-out pages, whole page): 1 − CER = −0.650 (CER 1.650). Writers H + J only: CER 0.354. New writer with clear spacing (Step 5): CER 0.222. | `results/gates/step6_final.json`, `step5_pipeline.json` |
| 2 | **Small real handwriting data.** Real word-level ground truth: 1,104 batch-2 TRAIN crops + 97 older crops for training, 71 VAL crops, 144 unseen-writer crops, 56 TESS_TEST crops. Only a handful of writers (A, D, H, J, the numbered-page writers, one unseen "purple pen" writer). | `data/splits/handwritten_batch2/words.csv`, `data/splits/step2_real_words/verified.csv`, `docs/STEP2_cnn_transformer.md` §7 |
| 3 | **Word segmentation is the main error source.** On VAL pages, perfect word crops give CER 0.122; the same pages read through found word boxes give 0.24 (joins) to 0.44 (gap rule). | `docs/STEP4_xlmroberta.md` §8, `docs/STEP5_pipeline.md` §3 |
| 4 | **Join rule over-joins** (about 15 % fewer words than truth), which hurts writers who leave clear gaps; the alternative gap rule over-splits cramped writing. Neither uses both gap width and recognizer confidence. | `docs/STEP5_pipeline.md` §4 |
| 5 | **No text/non-text filter.** Ruled or dotted paper, pencil fragments and speckle are found as lines and read as words (Step 6: 306 words read for 42 on writer D's pages). Only a warning exists (usability, not a fix). | `docs/STEP6_testing.md` §4, `pipeline/page.py::plausibility` |
| 6 | **Pencil and ruled/dotted paper are not supported.** | Step 6 |
| 7 | **Pages must be upright.** No automatic 90°/180° rotation detection (only small deskew). | `preprocess/pipeline.py` (deskew ≤ 10°) |
| 8 | **Letters that differ only by dots** remain hard: the 52-letter CNN reaches 82.25 %; weakest families jeem (68 %), hamza (69 %), reh/noon (77 %), beh (78 %). | `docs/STEP2_cnn_transformer.md` §5 |
| 9 | **Diacritics (harakat) are not recognized.** Text is normalized without harakat for training and scoring. | `common/text.py::normalize(diacritics=False)` |
| 10 | **Generalization to new writers is limited and hard to measure**: only one truly unseen writer exists with labels; settings tuned on training writers did not transfer (Step 3 lever 2: 0.928 on tuning pages, 0.809 on the unseen writer). | `docs/STEP3_tesseract.md` §8 |
| 11 | **The corrector is domain-tuned** (legal + everyday text): it changes ~2 % of correct words there and ~4 % on Wikipedia-style text; it can turn a right word wrong (3 of 103 on the unseen writer). | `docs/STEP4_xlmroberta.md` §7 |
| 12 | **Recognizer decoding is greedy** (no beam search, no lexicon constraint while decoding). | `cnn_transformer/models.py::WordRecognizer.greedy` |
| 13 | **Evaluation limits**: test sets are small (FINAL_TEST 7 pages / ~147 words; unseen writer 7 pages / 144 words); FINAL_TEST and the unseen-writer pages are now used up; writers of FINAL_TEST were seen in training (seen writers, new text). | `docs/STEP6_testing.md` §1 |
| 14 | **No paragraph/layout analysis** beyond lines (no columns, tables, headings). | code: only `line_structures` + `order_lines` |
| 15 | **Runs on one Mac, one page at a time**; correction loads two 1 GB models per page (~10 s per page with correction on A1). Linux/Windows not tested. | `app.py` `concurrency_limit=1`; app test runs |
| 16 | **976 handwritten sentence images (Mendeley) are unused** because they have no transcriptions. | `PLAN.md` "Data available for future work" |

---

## Part B — Future scope (proposals, NOT implemented)

Difficulty: Low / Medium / High. "Realistic next step" assumes one semester and this hardware (8 GB Mac; a free
Colab or lab GPU helps for the larger models).

| Proposal | What it is | Why it could help here | How it fits the current architecture | Data needed | Difficulty | Realistic next step? |
|---|---|---|---|---|---|---|
| **New data by copying typed text** | Writers copy printed Sindhi sentences by hand; the typed text is the ground truth | Fixes limitation 2, 10 and 13 at once; no transcription work | New folders in `data/raw/`, split by writer; existing scripts | 15–30 writers × ~20 lines; keep ≥ 5 writers as a new test set | Low (organisation) | **Yes — do first** |
| **Annotation tool / transcribe Mendeley images** | A small page (the app's Suggest mode is already half of one) to type or confirm line texts | 976 images, many writers (CC BY 4.0) | Use `run_pipeline.py --correct suggest` drafts as a typing aid; never train on unchecked drafts | ~10–16 h of typing | Low–Medium | Yes |
| **Text/non-text filter + ruling removal** | Classify each connected component as text/non-text (small CNN or rules); remove dotted/solid rulings | Fixes limitation 5–6 | New step in `pipeline/page.py` between `clean_page` and `line_structures` | Synthetic rulings/speckle on clean pages + a new held-out set with ruled/pencil pages | Medium | Yes |
| **Better word joining** | Combine gap width, recognizer confidence and LM score in the DP; tune with out-of-fold recognizers (leave one writer out) | Attacks the biggest error source (limitation 3–4) | `pipeline/page.py::best_grouping` gets a gap term; `tesseract_ft/cuts.py::best_cuts` already has the form | Existing data; 4 extra recognizer fine-tunes (~2 h each) | Medium | Yes |
| **Beam search + lexicon** | Keep the k best partial readings; prefer known words | Usually a solid gain over greedy decoding | `WordRecognizer` gets a `beam()` next to `greedy()` | None new | Low–Medium | Yes (quick win) |
| **CRNN / CNN + BiLSTM + CTC** | Convolutional features → bidirectional LSTM → CTC loss (no alignment needed between image columns and characters) | The standard architecture for line-level handwriting; CTC removes the need for word cutting when used on lines | Replace or complement `WordRecognizer` for whole lines; same preprocessing and line finder | Several hundred to thousands of real **line** images; printed line renders | Medium | Yes, as the line-level experiment |
| **CTC-based recognition** | Training objective that sums over all alignments | Lets the model learn from line images with only the line text | As above | Line images + text | Medium | With the CRNN item |
| **Line- and paragraph-level recognition** | Read whole lines (and later blocks) instead of words | Removes word segmentation errors (limitation 3) | A line recognizer after `order_lines`; correction unchanged | ~290 real lines exist now (too few); needs the new data | Medium–High | After data collection |
| **Transformer-based OCR / TrOCR fine-tuning** | Pretrained image Transformer encoder + text Transformer decoder (Li et al., TrOCR) fine-tuned on Sindhi words/lines | Strong pretrained visual features; may need less Sindhi data | Drop-in replacement for `WordRecognizer` (same input: word or line image → text); tokenizer must cover Sindhi letters | Same as above; GPU strongly advised (the published TrOCR-base has ~334 M parameters; TrOCR-small ~62 M is the realistic size for 8 GB — sizes as reported by Li et al., to be checked against the paper) | High | Possible with a GPU |
| **Vision Transformer (ViT) encoder** | Image split into patches + Transformer encoder | Alternative to the ResNet trunk | Replace `Trunk` in `WordRecognizer` | More data than the CNN | Medium–High | Only as a comparison experiment |
| **Handwritten word segmentation model** | Learned word boundary detector instead of gap rules | Alternative fix for limitation 3–4 | Replaces `line_units` / `best_grouping` | Word boxes exist for 1,319 verified batch-2 crops (1,104 TRAIN + 71 VAL + 144 unseen writer) | Medium | Possible |
| **Language-model correction** | Already implemented (Step 4) | — | — | — | — | Improve instead: domain-adaptive weights, stricter 1 % false-change setting, smaller/faster model (distillation, quantization) |
| **Sindhi POS tagging** | Tag each recognized word (noun, verb…) | Downstream use of the text; a POS-tagged corpus exists in `data/raw/pos_text` (676 sentences + a CRF model; **no licence found**) | Separate module after correction | Licensed POS data | Medium | Only if the licence is clarified |
| **Named-entity recognition (NER)** | Find names, places, dates | Searchable archives, metadata | After correction | Labelled Sindhi NER data — **NOT FOUND** in the project | High | Not a next step |
| **Searchable digitized manuscripts** | Store page text + word boxes, search and highlight words on the image | The JSON already has word boxes per page | A small index (e.g. SQLite full-text) over `run_pipeline.py` outputs | Scanned collections with permission | Medium | Yes as a demo, if accuracy is stated honestly |
| **Writer adaptation** | Fine-tune briefly on 10–20 lines of a new writer | Large gains per writer are common | `step2_train_recognizer.py --finetune` with a writer's pages | 10–20 labelled lines per writer | Low–Medium | Yes |
| **Automatic rotation detection** | Detect 90°/180° rotation | Fixes limitation 7 | Before `clean_page` | Rotated copies of existing pages | Low | Yes |
| **Human-in-the-loop learning** | Use the app's saved choices (`user_feedback/`) as new training data after manual checking | Grows data from real use | `pipeline/feedback.py` already records crops + choices | Checked records + writer consent | Low–Medium | Yes |
