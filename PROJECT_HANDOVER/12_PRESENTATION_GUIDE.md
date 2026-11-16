# 12 — Presentation guide (10 slides, ~10–12 minutes)

Figures you can use already exist in the repository (paths given). Screenshots of the app: run `app.py`, press
*Try an example page*, and capture screen 1 and screen 2 (Auto-correct and Suggest).

| # | Slide title | Main points | Diagram / screenshot | What to say |
|---|---|---|---|---|
| 1 | **Preserving Sindhi Heritage Through AI — Handwritten Sindhi OCR** | team, guide, institute; one-line statement | app screen 2 with Sindhi text | "We built an offline system that reads photos of handwritten Sindhi pages and outputs editable text." |
| 2 | **Problem and motivation** | Sindhi: Perso-Arabic script, right to left, 52 letters; handwritten heritage not searchable; very few tools for handwritten Sindhi (published work mostly isolated letters/numerals) | a sample page photo (`data/raw/handwritten/images/A1.png`) next to its text | Why it matters (digitisation, search, accessibility) and why no off-the-shelf tool works (Tesseract text CER 0.49 on our pages). |
| 3 | **Why handwritten Sindhi is hard** | dot-only differences (ب ٻ ڀ پ ت ٿ ٽ ٺ ث); cursive joins; gaps inside words; diacritics; few labelled data | a row of dot-variant letters; `results/step2/cnn_errors.png` (misread letters) | Point to the confusions the model actually makes: ڃ→ڄ, ٽ→ٿ. |
| 4 | **Objectives and approach** | step-by-step build with a 0.90 gate per step; honest evaluation (test once, predictions recorded) | a 6-step timeline: preprocess → letters/words → boxes → correction → pipeline → final test | Explain the gate idea: no step proceeds without measured evidence. |
| 5 | **System architecture** | photo → clean → lines → words → CNN+Transformer recognizer → XLM-R correction → text; CLI + local web app | the Mermaid system diagram from `02_SYSTEM_ARCHITECTURE.md` | Walk through one page; name the key files. |
| 6 | **Preprocessing** | background flattening + Sauvola, deskew, line/speck removal; chosen on a synthetic benchmark (worst-case F1 0.931); rembg rejected (lost 60–90 % of dots) | `results/step1/comparison_real_pages.png` | "We chose the method by its worst case, because a user's photo can have any background." |
| 7 | **Recognition model** | Letter CNN (82.25 % on 52 classes) → trunk reused; word recognizer: ResNet CNN + 2-layer Transformer encoder + 3-layer decoder, 8.1 M params; trained on printed + letter-composed + real words; fine-tuned on ~1,100 real words | model diagram from `02_…` §5; `results/step2/cnn_augment_samples.png` | Explain why words, not letters, and how we overcame little real data. |
| 8 | **Language-model correction + Suggest mode** | candidates from a 47k-word lexicon with dot-aware costs; XLM-R further trained on Sindhi scores them; detector flags wrong words; Suggest mode lets the user choose | screenshot of Suggest review cards | Real effect: 71.5 % → 79.2 % fully-correct words on an unseen writer. |
| 9 | **Results (honest)** | table: letters 0.8225; unseen-writer words 1 − CER 0.912; unseen-writer pages CER 0.222; final test H+J CER 0.354, overall CER 1.65 (two pencil/dotted pages); main error: word segmentation | the results table from `08_…` §2; `results/step6/debug/D6._lines.png` (dotted rulings read as lines, git-ignored figure) | Say clearly: below 90 %; explain why with evidence. Examiners respect honest negative results. |
| 10 | **Limitations, future work, conclusion** | few writers, no pencil/ruled paper, segmentation; next: copy-typed-text data collection, text/non-text filter, better joins, beam search, CRNN/TrOCR line model | a short roadmap graphic | End with the contribution and the next concrete step. Live demo if time allows (`python app.py`). |

## Delivery tips
- Demo: start the app **before** presenting (models load ~10 s per page); have a clear pen page ready; use Suggest
  mode to show the review cards.
- Keep numbers consistent with `08_EVALUATION_AND_RESULTS.md`; don't round up.
- If asked "is it the first?": "A web search showed earlier work on isolated Sindhi characters and numerals and some
  datasets; we did not find an open end-to-end page reader, but we have not done a full literature review, so we
  don't claim to be first."
- Backup: screenshots in case the live demo fails.
