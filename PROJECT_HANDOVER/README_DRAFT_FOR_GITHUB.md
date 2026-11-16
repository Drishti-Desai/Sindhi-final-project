# Preserving Sindhi Heritage Through AI — Handwritten Sindhi OCR

*Draft README for a future public repository. Review the data/model licence notes before publishing.*

An offline system that turns a photo of a **handwritten Sindhi page** into **Unicode Sindhi text**. It cleans the
photo, finds text lines and words, reads each word with a CNN + Transformer recognizer, and corrects likely mistakes
with a Sindhi-adapted XLM-RoBERTa language model. It runs locally from the command line or a web app.

> **Status:** research prototype. Accuracy is well below 90 % on whole pages (see Results). Works best on clear pen
> writing on plain paper. Pencil and ruled or dotted paper are not supported.

## Features
- Page cleaning for phone photos (shadows, coloured paper, tilt, straight rulings, desk background)
- Line finding that follows sloped handwriting; right-to-left reading order
- Word recognizer (8.1 M parameters) trained on printed renders, words composed from real letter crops and ~1,100
  real handwritten words
- Correction with a Sindhi-trained XLM-RoBERTa (candidate ranking) and a wrong-word detector
- Three correction modes: **Off**, **Auto-correct**, **Suggest** (review each change with the word image)
- Local web app (Gradio): two-screen flow, progress messages, boxes view, comparison view, `.txt` download
- Plausibility warning when a page was probably not read as text
- 79 unit tests; every step evaluated against a fixed 0.90 gate

## Architecture
```
photo → preprocess (flatten + Sauvola) → lines + reading order → word grouping (recognizer-guided)
      → WordRecognizer (ResNet CNN + Transformer encoder/decoder) → XLM-R correction (+ detector) → text / JSON
```
Details: `PROJECT_HANDOVER/02_SYSTEM_ARCHITECTURE.md`, `PROJECT_HANDOVER/03_MODEL_AND_ALGORITHMS.md`.

## Installation
Requires Python 3.13 (tested on macOS, Apple M2, 8 GB RAM).
```bash
git clone <repository-url> sindhi-ocr && cd sindhi-ocr
python3.13 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```
Download the three model folders (link to be added) into:
`checkpoints/step2_recognizer_ft/best.pt`, `checkpoints/step4_xlmr_mlm/best/`, `checkpoints/step4_xlmr_detector/best/`,
and the lexicon into `data/preprocessed/step4/lexicon.tsv`.

## Usage
```bash
python app.py                                   # web app at http://127.0.0.1:7860 (Ctrl+C to stop)
python run_pipeline.py page.jpg --out out/      # command line → out/page.txt, out/page.json
python run_pipeline.py page.jpg --correct suggest --words gaps
```

## Datasets (summary)
| Dataset | Use | Licence |
|---|---|---|
| Isolated Sindhi letters (4,241 unique images, 52 classes) | letter CNN | to be confirmed |
| Handwritten pages (22 + 129 images; 1,319 verified word crops) | recognizer fine-tuning and tests | own data — writer consent to be confirmed |
| Sindhi Wikipedia (20231101.sd) | text for lexicon and language model | CC BY-SA 3.0 + GFDL |
| Sindhi POS-tagged sentences; Sindhi legal Q&A text | text for lexicon and language model | **unknown — not redistributed** |
| Soomro & Khalid, Mendeley Data 2024 (doi:10.17632/phk66sgmp5.1) | inspected, not used for training | CC BY 4.0 |
Raw data is not included in this repository. See `PROJECT_HANDOVER/04_DATASET_AUDIT_AND_PROVENANCE.md`.

## Results
| Test | Result |
|---|---|
| 52-letter classification (873 test images, unseen sheets) | 82.25 % |
| Word images, writer never seen in training (144 words) | character accuracy 91.2 %; 71.5 % → **79.2 %** words fully correct after correction |
| Whole pages, unseen writer (7 pages) | CER 0.222 |
| Final test, 7 held-out pages, writers H + J | CER 0.354 |
| Final test incl. two pencil / dotted-paper pages | CER 1.65 (rulings and pencil fragments read as text) |
Full tables: `PROJECT_HANDOVER/08_EVALUATION_AND_RESULTS.md`.

## Limitations
Few writers and little real labelled handwriting; word segmentation is the main page-level error; no text/non-text
filter (pencil, ruled/dotted paper); greedy decoding; small test sets.

## Future work
More writers (people copy typed Sindhi text, so no transcription is needed); ruling and non-text removal; better
word joining; beam search; line-level CRNN/CTC or TrOCR. See `PROJECT_HANDOVER/09_LIMITATIONS_AND_FUTURE_SCOPE.md`.

## References
See `PROJECT_HANDOVER/05_RESEARCH_PAPERS_AND_REFERENCES.md`.

## Contributors
*(add names, roles and the project guide)*

## Licence
Code: *(choose MIT or Apache-2.0 — add a LICENSE file)*. Data and models: not licensed for redistribution until the
open permissions listed in `PROJECT_HANDOVER/10_GITHUB_PUBLICATION_GUIDE.md` §3 are resolved. Third-party
attributions: Wikipedia (CC BY-SA 3.0), Soomro & Khalid Mendeley dataset (CC BY 4.0), Roboflow sindhi-ocrf (CC BY 4.0),
xlm-roberta-base (Facebook AI).
