# 06 — Project uniqueness and innovation (honest assessment)

## 1. Why handwritten Sindhi is hard

- **52 letters, many differing only by dots.** One body shape can be ب ٻ ڀ پ ت ٿ ٽ ٺ ث depending on the number
  and position of dots; similarly ج ڄ ڃ چ ڇ ح خ and د ڊ ڌ ڍ ڏ ذ. In handwriting, dots become dashes, merge with
  each other or shift to the neighbouring letter. Our letter CNN gets the letter **body** right 93.8 % of the time
  but the dots right (given the body) only 86.8 % (`results/step2/cnn_error_analysis_gray_aux_synth.json`).
- **Connected (cursive) script, right to left.** Letters take different forms at the start, middle and end of a
  word and join each other, so a word cannot simply be cut into letters.
- **Gaps inside words.** Several letters (e.g. ا د ر و) never join the next letter, so a word is often several ink
  pieces with gaps that look like word spaces. This made **word segmentation the largest error source** in our
  pipeline (`docs/STEP4_xlmroberta.md` §8).
- **Diacritics and marks** (harakat, hamza, small marks like ۽ ۾) are small and easily lost in a photo.
- **Little data.** Public Sindhi handwriting data with transcriptions is scarce (see §3); our own transcribed real
  handwriting is ~1,300 word crops from a few writers.
- **Photos, not scans:** uneven light, shadows, desk backgrounds, ruled paper, pencil.

## 2. Why ordinary OCR struggles

Tesseract 5.5.2 with its Sindhi model, run on our unseen writer's pages, reached word-box F1 0.777 and a text CER of
0.487 (`results/step3/baseline_metrics.json`). Tesseract's models are trained mainly on printed text; handwriting,
dot variation and the gaps inside Sindhi words break its assumptions.

## 3. Existing work (from a web search on 2026-11-16; only what the search results showed)

| Work (as listed by the search result) | Scope | Status of the citation |
|---|---|---|
| "Handwritten Optical Character Recognition system for Sindhi numerals", IEEE conference paper, 2016 (IEEE Xplore document 7495235) | handwritten **numerals**, template matching | authors NOT VERIFIED |
| A. A. Sanjrani, J. Baber, M. Bakhtyar *et al.*, "Extended framework for Sindhi numerals OCR using gradient orientation histograms", *Journal of Intelligent & Fuzzy Systems*, 2022, doi:10.3233/JIFS-219304 | handwritten numerals | verified from search result |
| "Handwritten Sindhi Character Recognition Using Neural Networks", *Mehran University Research Journal of Engineering and Technology*, 2018 | isolated characters, neural network | authors NOT VERIFIED |
| "Deep learning-based isolated handwritten Sindhi character recognition", *Indian Journal of Science and Technology*, 2020 (Chandio *et al.* per Semantic Scholar) | isolated characters, deep residual CNN | first author partly verified |
| "OHSCR: Benchmarks Dataset for Offline Handwritten Sindhi Character Recognition", ~2024 (ResearchGate) | character dataset | authors/venue NOT VERIFIED |
| "A Comprehensive Database for Offline Sindhi Handwritten Text Recognition" (Academia.edu) | text database, 300 dpi scans | authors/year NOT VERIFIED |
| Awan, Hakro *et al.*, "A Complete Off-line Sindhi Handwritten Text Recognition: A Survey", 2017 (Zenodo 3469359) | survey; notes the lack of systems and databases | partly verified |
| Hakro *et al.*, "A Study of Sindhi Related and Arabic Script Adapted languages Recognition", 2014, arXiv:1412.4217 | survey | verified from search result |
| S. Naz *et al.*, "The optical character recognition of Urdu-like cursive scripts", *Pattern Recognition*, 2014 | survey incl. Sindhi | verified from search result |
| Solangi *et al.*, "Offline printed Sindhi optical text recognition: survey", ICETAS, 2018/2019 | **printed** Sindhi | verified from search result |
| M. Soomro, M. Khalid, "POS Tagging on Handwritten Sindhi Sentences", Mendeley Data, 2024, doi:10.17632/phk66sgmp5.1 | handwritten sentence images + POS (we have the images; no transcriptions) | verified (in `data/PROVENANCE.md`) |

**What this shows:** published handwritten Sindhi work found by the search focuses mostly on **isolated characters
or numerals**, plus surveys and datasets. The search did **not** show an open end-to-end system that reads whole
handwritten Sindhi pages and corrects the text with a Sindhi language model — but a web search is not a full
literature review, so **we must not claim to be the first**. A proper literature survey (Google Scholar, IEEE Xplore,
ACL Anthology) is still needed before any "first" claim.

## 4. What our implementation actually contributes

| Contribution | Kind | Evidence |
|---|---|---|
| An **end-to-end, offline, runnable** handwritten Sindhi page reader: photo → cleaned page → lines → words → recognized text → language-model correction → text/JSON, with a CLI and a local web app | engineering | `run_pipeline.py`, `app.py` |
| A **word recognizer trained mostly without real word data**: printed renders + words composed from real isolated-letter crops, then fine-tuned on ~1,100 real crops; CNN trunk transferred from the 52-letter classifier | method combination (adapted from known techniques) | `cnn_transformer/models.py::load_trunk_from_letter_cnn`, `docs/STEP2_cnn_transformer.md` |
| **Sindhi-adapted XLM-RoBERTa correction** with candidates from a Sindhi visual-confusion table (letters that differ by dots cost less to swap) and a recognizer error model learned from the recognizer's own mistakes | method combination for Sindhi | `xlmr/correct.py`, `xlmr/errors.py` |
| **Suggest mode**: the user reviews each proposed correction with the word image; opt-in saving of reviewed words for future (checked) training data — a human-in-the-loop path | engineering / usability | `xlmr/correct.py::suggest`, `pipeline/feedback.py` |
| **Gated, honest evaluation**: every step had a 0.90 gate, test sets measured once, predictions recorded before tests, all failures documented | methodology | `results/gates/*.json`, `docs/STEP*.md` |
| Small but real **evaluation data with leakage control**: sheet-disjoint letter split, de-duplication, unseen-writer and held-out-text tests | data engineering | `data/splits/`, `tesseract_ft/texts.py` |
| Documented finding that **word segmentation** (not character recognition) dominates page-level error for this script, with measurements | empirical finding | `docs/STEP4_xlmroberta.md` §8, `docs/STEP5_pipeline.md` |

**Engineering, not research novelty:** preprocessing (Sauvola, background flattening), ResNet + Transformer
recognizer, XLM-R masked-LM scoring, Gradio app — all are established techniques applied and adapted to Sindhi.

**Not achieved:** 90 % accuracy at any real-handwriting step; pencil / ruled-paper support; line-level
recognition; a large multi-writer dataset; a full literature review.

## 5. Comparison with typical OCR approaches

| Approach | Typical use | Compared with ours |
|---|---|---|
| Tesseract (LSTM, printed text) | printed documents | we measured it on our pages (CER 0.49); we use our own box finder instead |
| Character classifier + segmentation | early Sindhi/Arabic OCR | we classify letters only as pre-training; words are read whole, avoiding letter segmentation |
| CRNN + CTC on lines | modern handwriting OCR | not implemented (proposed, needs line data) |
| TrOCR / transformer OCR | modern handwriting OCR with large pretraining | not implemented; our recognizer is a small (8 M) CNN + Transformer word model trained on an 8 GB laptop |
| Spell checkers (dictionary only) | post-processing | we compared: word list alone 0.843 vs XLM-R + detector 0.890 on synthetic test text |

## 6. Ready-to-use explanations

**One-line innovation statement:** An offline, end-to-end reader for handwritten Sindhi pages that combines a
CNN-Transformer word recognizer trained largely from synthetic and letter-composed data with a Sindhi-adapted
XLM-RoBERTa corrector, evaluated with strict, honestly reported gates.

**30 seconds:** Sindhi has 52 letters, many differing only by dots, and almost no tools read handwritten Sindhi.
We built a system that takes a phone photo of a handwritten page, cleans it, finds lines and words, reads each word
with a deep-learning model and fixes likely mistakes with a Sindhi language model. It runs fully offline with a
simple web app. It works best on clear pen writing; accuracy is still well below 90 %, and we measured exactly
where the errors come from.

**1 minute:** Add to the 30 seconds: the recognizer is a CNN plus Transformer; because real labelled handwriting
was scarce we trained it on printed word images and on words assembled from real letter crops, then fine-tuned it
on about 1,100 real words, reaching 91 % character accuracy on a writer it had never seen. A Sindhi-trained
XLM-RoBERTa then corrects words, raising fully-correct words on that writer from 71.5 % to 79.2 %. On whole pages
the main problem is deciding where words begin and end, because Sindhi words contain gaps. The app also has a
Suggest mode where the user approves each correction.

**3-minute technical explanation (outline):**
1. Problem and difficulty (dots, cursive, gaps, little data) — 30 s.
2. Pipeline: Sauvola-based cleaning chosen on a synthetic benchmark (worst-case F1 0.93); tracked line finder;
   reading order; word grouping by dynamic programming over recognizer scores — 45 s.
3. Recognizer: ResNet trunk (from the 82 %-accurate letter CNN) → 2-layer Transformer encoder → 3-layer decoder,
   82-symbol vocabulary, trained on printed + composed + real words; 0.912 1 − CER on an unseen writer — 45 s.
4. Correction: lexicon candidates with a dot-confusion cost table, XLM-R masked-LM scoring further trained on
   Sindhi, detector; 71.5 % → 79.2 % words right — 30 s.
5. Results and honesty: gates, final test (H/J CER 0.35, pencil/dotted pages fail), limitations, next steps
   (more writers, line recognizer, text/non-text filter) — 30 s.

**Detailed viva explanation:** use `13_VIVA_QUESTIONS_AND_ANSWERS.md` and `03_MODEL_AND_ALGORITHMS.md`.
