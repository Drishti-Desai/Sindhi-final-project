# 13 — Viva questions and answers

Answers use the project's actual numbers (sources in `08_EVALUATION_AND_RESULTS.md`). Keep answers short in the
viva; the extra detail is here so you understand them.

## A. Basics

1. **What does your project do?** It reads a photo of a handwritten Sindhi page and outputs Sindhi Unicode text.
   It cleans the photo, finds lines and words, recognizes each word with a CNN + Transformer model and corrects
   likely mistakes with a Sindhi-adapted XLM-RoBERTa language model. It runs offline with a web app.
2. **Why Sindhi?** Sindhi is spoken by tens of millions of people (Pakistan and India) and has a large handwritten
   heritage, but very few tools read handwritten Sindhi; most published work covers isolated characters or numerals.
3. **What is OCR?** Optical Character Recognition: converting images of text into machine-readable text.
4. **Why is handwritten OCR harder than printed?** Every writer is different (shape, size, slant, spacing), letters
   join, pens vary, and photos add shadows and noise. Printed fonts are consistent.
5. **What makes Sindhi especially hard?** 52 letters, many differing only by dots (ب ٻ ڀ پ ت ٿ ٽ ٺ ث); cursive
   right-to-left writing with position-dependent letter forms; letters that do not join the next one create gaps
   inside words; small marks and diacritics.
6. **What is the input and output?** Input: an upright JPG/PNG photo of a page. Output: text (one line per text
   line), a JSON with boxes and readings, and in the app a downloadable `.txt`.
7. **Does it need the internet or a paid AI service?** No. Everything runs locally; models are on disk.

## B. Architecture

8. **Describe the pipeline.** Preprocess (`clean_page`) → find lines (`line_structures`) → reading order
   (`order_lines`) → group ink pieces into words and read them (`read_page`, `best_grouping`, `WordRecognizer`) →
   correct (`Corrector`) → output.
9. **Why not use Tesseract?** We measured it: on the unseen writer's pages Tesseract's text CER was 0.49 and its
   word-box F1 0.78; our own box finder reached 0.86. Tesseract is trained mainly on printed text.
10. **Why a word recognizer and not a character classifier?** Cursive words cannot be reliably cut into letters.
    Reading the whole word image avoids letter segmentation; the letter CNN is used only to pre-train the encoder.
11. **What is the recognizer's architecture?** A ResNet-style CNN (4 stages, 32–256 channels) turns the 64-px-high
    word image into a sequence of features (one per 8 pixels of width); a 2-layer Transformer encoder adds context;
    a 3-layer Transformer decoder writes characters one by one (82-symbol vocabulary). About 8.1 M parameters.
12. **Why CNN + Transformer?** The CNN extracts local stroke and dot features; the Transformer attends to the whole
    word, handles variable length, and outputs a character sequence without needing alignment labels.
13. **What does the decoder output and how?** Greedy decoding: at each step it outputs the most likely next
    character until the end symbol (max 26 steps).
14. **How are words found on a line?** Each line is split into ink spans; the recognizer reads every run of up to 6
    neighbouring spans and dynamic programming keeps the grouping with the highest total log-probability
    ("joins"). Alternatively, the gap rule cuts at wide gaps ("gaps").
15. **What is XLM-RoBERTa and why use it?** A multilingual Transformer language model (Conneau et al., 2020)
    pre-trained on 100 languages including Sindhi. It cannot write text, but it can score how well a word fits its
    context. We use it to rank correction candidates.
16. **How does the correction work?** For each word: generate candidates from a 47,422-word lexicon by a weighted
    edit distance (dot-variant swaps are cheap), plus splits and merges; score each candidate with XLM-R's masked-LM
    probability in context, word frequency and edit cost; a detector flags likely-wrong words; replace only if the
    best candidate beats the original by a threshold chosen on validation.
17. **What is the detector?** The same XLM-R encoder with a 2-class head per token: "is this word wrong?". Trained
    on text corrupted to look like recognizer errors.
18. **What is Suggest mode?** Instead of changing words automatically, the app shows each proposed change with the
    word image and lets the user choose keep / candidate 1–3. Choices can optionally be saved for future training.
19. **How does the frontend talk to the backend?** Gradio runs a local web server (127.0.0.1); button clicks call
    Python functions (`process`, `apply`) and the results update the page. No data leaves the computer.

## C. Preprocessing

20. **What does preprocessing do?** Flattens uneven lighting (background estimate by morphological closing, then
    division), chooses the best grey channel, binarises with Sauvola, inverts dark backgrounds, deskews, removes
    straight ruled lines, borders and specks, and keeps dots.
21. **Why Sauvola?** It is a local threshold, so it handles shadows and faint pencil. On our synthetic benchmark it
    had the best worst-case F1 (0.931) across 12 page conditions; Otsu dropped to 0.82 on faint pencil.
22. **Why not deep-learning background removal (rembg/U²-Net)?** We tested it: it treats text on paper as
    background and lost 60–90 % of the dots (F1 0.16).
23. **How did you evaluate preprocessing without ground truth?** Synthetic Sindhi pages with a known ink mask under
    12 conditions; text-pixel F1 and dot recall; Step 1 gate PASS (F1 0.956, dot recall 0.993).

## D. Data

24. **What data did you use?** 4,241 unique isolated-letter images (52 classes) for the letter CNN; printed-font
    word renders and words composed from letter crops; ~1,100 real handwritten word crops (batch 2 TRAIN) + 97 from
    older pages for fine-tuning; Sindhi text (legal Q&A dataset, POS sentences, Wikipedia) for the language model.
25. **Where did the data come from?** Letter images and the first 22 handwritten pages were copied from the team's
    earlier project folder (the letters' original source and licence are not recorded); a second batch of
    handwritten pages collected by the team; Sindhi Wikipedia (CC BY-SA 3.0); a Sindhi legal Q&A CSV and a POS-tagged
    corpus for text (origin/licence not recorded); a Mendeley handwritten dataset (CC BY 4.0, not used for training
    because it has no transcriptions). See `04_DATASET_AUDIT_AND_PROVENANCE.md`.
26. **How did you avoid data leakage?** Removed byte-identical duplicates (77 % of letter files) before splitting;
    split letters by source sheet; held one writer out completely; removed every sentence of test pages from all
    training text; excluded copies of test pages from the second batch.
27. **Why split by sheet, not by image?** Letters from the same sheet share writer, pen and lighting; a random
    image split would put near-identical samples in train and test and inflate accuracy.
28. **Is there word-level ground truth?** Yes, but small: 1,319 verified batch-2 word crops (1,104 TRAIN, 71 VAL,
    144 unseen writer) and 56 verified TESS_TEST crops; page-level text for 22 older pages and the batch-2 pages.
29. **What is "composed" data?** Words built by placing real isolated-letter crops side by side; used because real
    word images were scarce. It looks less like real cursive writing.

## E. Training

30. **How was the letter CNN trained?** ResNet-style CNN on 64×64 grayscale crops with augmentation (rotation,
    shear, scale, stroke width, elastic warps), label smoothing 0.1, auxiliary heads (body shape, dot position, dot
    count), 25 % font-rendered letters, early stopping on validation; test-time augmentation at test.
31. **How was the recognizer trained?** Trunk initialised from the letter CNN; mixed batches of printed (45 %),
    composed (35 %) and real (20 %) words; 15 epochs × 900 steps, batch 32 (2 × 16 with gradient accumulation);
    then fine-tuned 8 epochs with 40 % real batch-2 words.
32. **How did you fit training in 8 GB?** Gradient checkpointing, micro-batches with gradient accumulation, a GPU
    memory cap; for XLM-R: frozen embeddings stored in bfloat16, lower 6 layers frozen, output layer only at
    masked positions.
33. **What is masked-LM training?** Hide 15 % of words (whole-word masking) and train the model to predict them;
    we continued XLM-R's training on Sindhi text (validation loss 3.60 → 3.11).
34. **What is label smoothing?** Training targets put e.g. 0.9 on the right class and spread 0.1 over the others,
    which reduces over-confidence.
35. **What is test-time augmentation?** Averaging predictions over several slightly changed copies of a test image
    (5 views); it raised letter accuracy from 0.815 to 0.8225.

## F. Evaluation

36. **What is CER?** Character edits (substitutions, deletions, insertions) divided by reference characters. 0 is
    perfect; it can exceed 1 when many extra characters are produced.
37. **What are your main results?** Letter CNN 82.25 %; word recognizer on an unseen writer's words 1 − CER 0.912
    (71.5 % of words fully right), 79.2 % after correction; whole pages of that writer CER 0.22; final test CER 0.35
    on writers H and J, but 1.65 overall because two pencil/dotted-paper pages were read as hundreds of false words.
38. **Why did you not reach 90 %?** Too little real handwriting from too few writers; dot-level ambiguity; word
    segmentation errors on whole pages; no handling of rulings/pencil.
39. **What is a "gate"?** A pass/fail check (≥ 0.90) each step had to pass before the next; failures were analysed
    and explicitly accepted by the project owner, and the files record this.
40. **Why measure the test set only once?** Re-testing after changes turns the test set into a tuning set and
    inflates results. We also wrote predictions before each test.
41. **What is the biggest error source?** Word segmentation: perfect word crops give CER 0.12, whole pages 0.24–0.44.
42. **Are your results reliable?** They are honest but based on small test sets (56–147 words), few writers, and
    the final-test writers were seen in training (new text). One truly unseen writer exists.

## G. Technology

43. **Which technologies?** Python 3.13, PyTorch (Apple GPU), Hugging Face Transformers, OpenCV, scikit-image,
    NumPy, Pillow, Gradio, pytest, git.
44. **Why PyTorch?** Flexible model definition, Apple-GPU (MPS) support, and compatibility with Hugging Face models.
45. **Why Gradio?** Quick, local web interface for ML in Python, no separate frontend code; runs offline.
46. **How is the project organised?** Packages per step (`preprocess`, `cnn_transformer`, `tesseract_ft`, `xlmr`,
    `pipeline`), one script per stage, all paths in `configs/paths.yaml`, gate files in `results/gates/`, 79 tests.

## H. Innovation, limitations, future

47. **What is new in your work?** An offline end-to-end handwritten Sindhi page reader combining a word recognizer
    trained mostly from synthetic and letter-composed data with a Sindhi-adapted XLM-R corrector, a human review
    mode, and strict, documented evaluation. The individual techniques are known; the combination and the Sindhi
    adaptation (confusion-aware candidates, error model) are our contribution. We do not claim to be the first.
48. **Main limitations?** Accuracy far below 90 %; few writers; no pencil or ruled-paper support; word segmentation
    errors; greedy decoding; small test sets.
49. **What would you do next?** Collect more writers by having people copy typed Sindhi text (no transcription
    needed), add a text/non-text filter and ruling removal, improve word joining with gap width + recognizer score,
    beam search, and a line-level CRNN/CTC or TrOCR model.
50. **Can it be used on manuscripts?** Not reliably yet; old manuscripts differ in script style and condition. It
    would need training data from such manuscripts.
51. **How would you handle a new writer?** Fine-tune briefly on 10–20 labelled lines of that writer (writer
    adaptation), or use Suggest mode and save the corrections.
52. **What did you learn?** Honest evaluation matters: our prediction for the final test was wrong for two pages,
    and the measurement showed exactly why (rulings and pencil fragments read as text).

---

## The 20 most important functions (beginner-friendly)

| # | Function | File | What it does, simply |
|---|---|---|---|
| 1 | `main()` | `app.py` | Checks the gate files and starts the local web app. |
| 2 | `process()` | `app.py` | When you press *Read page*: checks the photo, cleans it, runs the pipeline, fills the results screen. |
| 3 | `apply()` | `app.py` | In Suggest mode, applies your choices, updates the text and optionally saves them. |
| 4 | `run()` | `run_pipeline.py` | The heart: loads the recognizer, reads every page, adds warnings, corrects the text. Shared by app and CLI. |
| 5 | `clean_page()` | `pipeline/page.py` | Turns a photo into a clean black-on-white page and drops specks. |
| 6 | `preprocess_image()` | `preprocess/pipeline.py` | Measures the image (contrast, noise, tilt) and applies only the needed fixes. |
| 7 | `binarize()` | `preprocess/ops.py` | Decides for every pixel: ink or paper (Sauvola local threshold). |
| 8 | `line_structures()` | `tesseract_ft/boxes.py` | Finds text lines (following sloped lines) and the ink pieces inside them. |
| 9 | `order_lines()` | `pipeline/layout.py` | Puts line pieces in reading order: rows top to bottom, right to left. |
| 10 | `read_page()` | `pipeline/page.py` | For each line, tries groupings of ink pieces as words and keeps the best one. |
| 11 | `best_grouping()` | `pipeline/page.py` | Dynamic programming that picks the word grouping with the highest total recognizer confidence. |
| 12 | `read_groups()` | `tesseract_ft/cuts.py` | Cuts a group's ink, resizes it and asks the recognizer what it says. |
| 13 | `WordRecognizer.greedy()` | `cnn_transformer/models.py` | The model reads a word image and writes characters one by one. |
| 14 | `load_trunk_from_letter_cnn()` | `cnn_transformer/models.py` | Copies the letter CNN's learned layers into the word recognizer as a starting point. |
| 15 | `build_positions()` | `xlmr/correct.py` | For each recognized word, lists possible corrections (similar dictionary words, splits, merges). |
| 16 | `MaskedScorer.score()` | `xlmr/lm.py` | Hides a word and asks XLM-R how likely each candidate is in that sentence. |
| 17 | `decide()` | `xlmr/correct.py` | Combines edit cost, word frequency and language-model score and decides whether to change each word. |
| 18 | `suggest()` / `apply_choices()` | `xlmr/correct.py` | Lists what would change (top-3) without changing; applies the user's picks. |
| 19 | `plausibility()` | `pipeline/page.py` | Warns when the page was probably not read as text (stacked lines, many one-letter words). |
| 20 | `require_gate()` | `common/gates.py` | Stops a script unless the previous step's gate file exists (and passed or was accepted). |
| + | `cer()` / `wer()` | `common/metrics.py` | Compute the error rates used everywhere. |
