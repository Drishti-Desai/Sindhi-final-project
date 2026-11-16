# 11 — BTech report content (draft text based on the actual project)

Write the final report in your own words; these drafts use only verified facts. Numbers: `08_EVALUATION_AND_RESULTS.md`.
References: `05_RESEARCH_PAPERS_AND_REFERENCES.md`.

---

## 1. Abstract (≈ 220 words)

Sindhi is written in a Perso-Arabic script of 52 letters, many of which differ only in the number or position of
dots, and handwritten Sindhi documents remain largely undigitised. This project presents an offline, end-to-end
system that converts a photograph of a handwritten Sindhi page into Unicode text. The page is first cleaned by
illumination flattening and Sauvola binarisation, a method selected on a synthetic benchmark for its worst-case
performance (F1 0.931 across twelve page conditions). Text lines and ink pieces are located with a line finder that
follows sloped handwriting, and words are formed by dynamic programming over the recogniser's confidence. Each word
is read by a convolutional–Transformer recogniser (8.1 M parameters) whose convolutional trunk is pre-trained as a
52-class letter classifier (82.25 % test accuracy) and which is trained on printed renders, words composed from real
letter crops and about 1,100 real handwritten words. A Sindhi-adapted XLM-RoBERTa language model ranks correction
candidates generated with a dot-aware edit distance. On a writer never seen in training, the recogniser reaches
0.912 character accuracy on word images, and correction raises fully correct words from 71.5 % to 79.2 %. On whole
pages, accuracy is limited mainly by word segmentation (character error rate 0.22 for the unseen writer and 0.35 on
held-out pages of seen writers), and pencil or ruled paper is not yet supported. The system runs locally through a
command-line tool and a web interface with a human review mode, and all results are reported against pre-defined
gates.

## 2. Introduction

- Context: Sindhi language and script; handwritten records (personal, legal, educational, historical) are not
  searchable; OCR enables digitisation, search and accessibility.
- OCR basics: printed OCR is mature; handwritten OCR is harder (writer variation, cursive joins).
- Sindhi specifics: 52 letters; dot-only distinctions; right-to-left cursive; letters that do not join create gaps
  inside words; diacritics.
- This project: build a working system step by step, each step measured against a 0.90 gate, with honest reporting.
- Report organisation.

## 3. Problem statement

"To design and implement an offline system that accepts a photograph of a handwritten Sindhi page and produces the
corresponding Unicode Sindhi text, and to measure its accuracy honestly on handwriting from writers and texts not
used in training."

## 4. Objectives

1. Preprocess page photos and letter crops so that thin strokes and dots survive (Step 1).
2. Train a 52-letter classifier and a word-level recogniser from limited real data (Step 2).
3. Locate text lines and words; compare with Tesseract (Step 3).
4. Correct recognition errors with a Sindhi language model (Step 4).
5. Integrate a complete pipeline with a command-line tool and a local web interface (Step 5).
6. Evaluate the complete system on held-out pages (Step 6).
7. Each step: a measurable gate (≥ 0.90), test sets used once, all failures analysed and documented.

## 5. Existing system

- General OCR engines such as Tesseract are trained mainly on printed text; on our handwritten pages Tesseract's
  Sindhi model gave a text CER of 0.487 and word-box F1 0.777.
- Published handwritten Sindhi work found in our search concerns mostly isolated characters and numerals (neural
  networks, gradient-orientation histograms, residual CNNs), plus surveys and datasets (see `06_…` §3 and `05_…`).
- Limitations of existing approaches for our goal: no open end-to-end page reader found in the search; character-level
  systems require letter segmentation, which is unreliable for cursive script.

## 6. Proposed system

- Word-level recognition (no letter segmentation) with a CNN + Transformer encoder–decoder.
- Training data strategy for a low-resource script: printed renders + letter-composed words + a small real set.
- Sindhi language-model correction (XLM-RoBERTa adapted by further masked-LM training) with dot-aware candidates.
- Page pipeline with line finding, reading order and recogniser-guided word grouping.
- Local web application with Suggest mode (human review) and opt-in saving of reviewed words.
- Gated, reproducible evaluation.

## 7. Literature survey (structure)

1. **Document image binarisation:** Otsu (global), Sauvola (local) — basis of Step 1.
2. **Deep CNNs:** residual networks (He et al.) — the letter classifier and the recogniser's trunk.
3. **Sequence models for text recognition:** Transformer (Vaswani et al.); CTC-based CRNN and TrOCR as alternatives
   (future work).
4. **Language models:** BERT (masked LM), XLM-RoBERTa (multilingual, includes Sindhi) — Step 4.
5. **Spelling correction:** edit distance (Levenshtein), symmetric-delete lookup (SymSpell).
6. **OCR engines:** Tesseract (Smith).
7. **Sindhi and Urdu-like script OCR:** surveys (Naz et al.; Hakro et al.; Awan, Hakro et al.; Solangi et al.);
   handwritten Sindhi numerals (IEEE 2016; Sanjrani et al. 2022); isolated characters (MUET 2018; Indian J. Sci.
   Tech. 2020); datasets (OHSCR; offline Sindhi handwritten text database; Mendeley POS handwritten sentences).
   Read the full papers before citing their numbers.

## 8. System architecture

Use the diagrams in `02_SYSTEM_ARCHITECTURE.md` (system, data flow, model, sequence). Describe the six stages and
name the implementing files.

## 9. Methodology

1. **Data preparation:** de-duplication (77 % of letter files were copies), sheet-grouped splits, unseen-writer and
   held-out-text test sets, sentence-level leakage filter.
2. **Preprocessing selection:** synthetic benchmark with known ink masks; worst-case F1 as the criterion.
3. **Letter CNN:** ResNet-style, augmentation, label smoothing, auxiliary heads, font-rendered letters, grayscale
   input, test-time augmentation.
4. **Word recogniser:** trunk transfer; mixed training sources; fine-tuning on real words; greedy decoding.
5. **Box finding:** text-free gap rule vs Tesseract; Hungarian matching at IoU ≥ 0.5 for scoring.
6. **Correction:** lexicon candidates, recogniser error model (confusions learned from the recogniser's own
   validation errors), XLM-R masked scoring, detector, weights chosen on validation (max accuracy with ≤ 2 % false
   changes).
7. **Integration:** reading-order rule chosen on training pages; recogniser-guided word joins; plausibility warning.
8. **Evaluation protocol:** each test once; predictions recorded before tests; CER/WER definitions.

## 10. Dataset description

Use the provenance table and counts in `04_DATASET_AUDIT_AND_PROVENANCE.md` (letters52 4,241 unique images; 22
pages / 402 words; batch 2: 129 images, 1,319 verified word crops; text corpora; Mendeley images not used). State
licences and the items whose licence is unknown.

## 11. Implementation

- Environment: Python 3.13, PyTorch 2.14 (Apple MPS), Transformers 5.23, OpenCV, scikit-image, Gradio 6.32; Apple M2,
  8 GB RAM; memory techniques (gradient checkpointing, accumulation, frozen bfloat16 embeddings).
- Code organisation: packages per step, scripts per stage, configs, 79 unit tests, git history (35 commits).
- Training details: `03_MODEL_AND_ALGORITHMS.md`. Run instructions: `07_INSTALLATION_AND_RUN_GUIDE.md`.
- Screenshots: app screen 1, screen 2, Suggest cards.

## 12. Results and discussion

- Present the step table from `08_…` §2.
- Discuss: (a) synthetic vs real gap (composed words 0.866 vs unseen real words 0.912 — real fine-tuning mattered);
  (b) correction helps real words (71.5 → 79.2 %) but can change correct words (3 of 103); (c) segmentation dominates
  page error (0.12 vs 0.24–0.44); (d) final test: H/J pages 0.354 CER, writer D's pencil/dotted pages fail
  (306 words read for 42); (e) predictions vs outcomes (Step 5 and Step 6 predictions were wrong and why).

## 13. Limitations

Copy the list from `09_LIMITATIONS_AND_FUTURE_SCOPE.md` Part A (accuracy, data, segmentation, rulings/pencil,
upright pages, dots, diacritics, generalisation, domain-tuned corrector, greedy decoding, small test sets).

## 14. Future scope

From `09_…` Part B: data collection by copying typed text; text/non-text filter; better joins; beam search; CRNN/CTC
or TrOCR line recognition; writer adaptation; searchable archive.

## 15. Conclusion (draft)

"We built and evaluated an offline end-to-end system for handwritten Sindhi page recognition. Word-level
recognition with a convolutional–Transformer model, trained largely from synthetic and letter-composed data and
fine-tuned on a small real set, reaches 0.912 character accuracy on an unseen writer's words, and a Sindhi-adapted
XLM-RoBERTa corrector improves fully correct words from 71.5 % to 79.2 %. Whole-page accuracy remains below our
90 % target, mainly because word boundaries are hard to find in Sindhi handwriting and because ruled paper and pencil
are not yet handled. Every step was measured against a fixed gate and reported honestly, which identifies precisely
where future work should focus: more writers, segmentation and line-level recognition."

## 16. References

Use the IEEE list in `05_RESEARCH_PAPERS_AND_REFERENCES.md` (section "References for the BTech report").
