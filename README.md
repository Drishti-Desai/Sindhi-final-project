# Sindhi OCR

A Sindhi OCR system built one proven component at a time. Each step must pass its **gate**
before the next step runs. Full plan and dataset findings: `PLAN.md`; per-step guides: `docs/`.
**New to the project (person or AI assistant)? Start with `docs/PROJECT_CONTEXT.md`.**

**Licence:** all rights reserved, no use without written permission (`LICENSE`). **Data:** raw datasets and model weights
are not included; result figures show handwriting excerpts that may not be reused. Datasets and sources: `DATASETS.md`.

## Steps

| Step | What | Guide | Status |
|---|---|---|---|
| 1 | Preprocessing: any background → dark text on white | `docs/STEP1_preprocessing.md` | ✅ gate PASS |
| 2 | 52-letter CNN + Transformer decoder (word images) | `docs/STEP2_cnn_transformer.md` | ⚠️ gates FAIL, **accepted by the user** (letters 0.8225, composed words 0.866 < 0.90). Model: `step2_recognizer_ft`. Reference for real handwriting: unseen writer 1 − CER **0.912**, 71.5% words correct |
| 3 | Tesseract 5.5.2: word/line boxes (+ text CER/WER) | `docs/STEP3_tesseract.md` | ⚠️ gate FAIL, **accepted by the user** (word-box F1 0.809 < 0.90). Step 5 box finder: our own gap rule (unseen writer 0.862) |
| 4 | XLM-RoBERTa spelling/grammar correction | `docs/STEP4_xlmroberta.md` | ⚠️ accepted by the user with gate FAIL (domain test 0.890 < 0.90, 2.07% correct words changed); pipeline uses Sindhi XLM-R + detector. Real handwriting: unseen writer 71.5% → 79.2% words correct |
| 5 | Combined pipeline (`run_pipeline.py`) | `docs/STEP5_pipeline.md` | ⚠️ accepted by the user with gate FAIL: unseen writer's pages 1 − CER 0.778 < 0.90 (worse than the Step 4 gap-rule reading, 0.807); VAL pages CER 0.462 → 0.246. Default `--words joins`; `--words gaps` for writers with clear spacing |
| 6 | End-to-end test on held-out handwriting | `docs/STEP6_testing.md` | ⚠️ accepted by the user with gate FAIL: FINAL_TEST 1 − CER −0.650 (writer D's pencil / dotted-ruling pages read as hundreds of false words); writers H + J: 0.646 |

## Gates

Each step writes `results/gates/<step>.json` with its metrics and PASS/FAIL. Each step's script
first calls `common.gates.require_gate(<previous step>)` and refuses to run if that gate is
missing or failed. Thresholds live in `configs/` and are never lowered in code.

## Folders

| Folder | Contents |
|---|---|
| `configs/` | `paths.yaml` (every path; nothing is hardcoded), per-step parameters, `letter_map.yaml` |
| `common/` | shared helpers: path loading, gates, Sindhi text normalisation, CER/WER |
| `cnn_transformer/` | Step 2 code: `models.py` (LetterCNN, WordRecognizer), `letters.py`, `words.py`, `train_utils.py` |
| `preprocess/` | Step 1 code: `ops.py` (one function per operation), `pipeline.py` (`preprocess_image`), `synthetic.py` (benchmark data), `segmentation.py` (rembg comparison) |
| `scripts/` | runners: `prepare_splits.py`, `step1_compare_methods.py`, `step1_preprocess.py`, … |
| `data/` | `raw/` (copies, not in git, see `data/PROVENANCE.md`), `preprocessed/`, `splits/` (fixed, in git) |
| `results/` | `gates/` and per-step metrics/plots |
| `pipeline/` | Step 5 page reading: reading order, word joins, correction wrapper, plausibility warning |
| `run_pipeline.py`, `app.py` | command line and local web frontend (same pipeline code) |
| `checkpoints/`, `logs/`, `pipeline_out/` | model weights, run logs, per-image pipeline outputs (not in git) |
| `tests/` | unit tests (`python -m pytest -q`) |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # only needed for optional API keys
```

## How to use

```bash
.venv/bin/python run_pipeline.py page1.jpg page2.png --out pipeline_out/
```

Per image it writes `pipeline_out/<name>.txt` (one line of text per found line) and `<name>.json` (line and word
boxes, the recognizer's readings, the corrected words, the action taken on each, and any warnings). Options:

| option | effect |
|---|---|
| `--words joins` (default) | the recognizer decides which neighbouring ink pieces form one word (best for cramped writing) |
| `--words gaps` | cut words at wide gaps only (better for writers who leave clear spaces between words) |
| `--correct off\|auto\|suggest` | correction mode (default `auto`); `suggest` changes nothing and lists the candidates in the JSON |
| `--no-correct` | same as `--correct off` |
| `--erase-rules` / `--remove-circled` | remove curved notebook rulings / circled question numbers |

Pages must be upright. Models needed (not in git): `checkpoints/step2_recognizer_ft/best.pt`,
`checkpoints/step4_xlmr_mlm/best`, `checkpoints/step4_xlmr_detector/best`.

**Plausibility warning.** When far more text lines or words are found than a page of text can hold (rulings, speckle
or pencil fragments being read as text), `run_pipeline.py` prints a warning that the output is probably unreliable.
This is a **usability warning only, not a measured improvement**: its thresholds were set by hand after seeing the
Step 6 test pages (`pipeline/page.py`, `plausibility`).

## Web frontend (local only)

```bash
cd ~/Downloads/NEW
.venv/bin/python app.py              # then open http://127.0.0.1:7860 in a browser
```

Stop it with **Ctrl+C** in that terminal (or, from another terminal: `kill $(lsof -ti tcp:7860)`).
`--port 7861` uses another port. The app (`app.py`, Gradio 6.32) runs on this computer only (127.0.0.1, no share
link, Gradio analytics off, no fonts or files fetched from the internet) and uses the same pipeline code as
`run_pipeline.py`. A step indicator guides the flow **1 Upload → 2 Read → 3 Review (Suggest only) → 4 Download**
over two screens:

**Screen 1, "Set up".** Upload a page photo (JPG or PNG; at least 120 px on the shorter side), or press **Try an
example page** (a training page, A1). Choose the correction: **Off** (the recognizer's text only), **Auto-correct**
(default) or **Suggest** (you choose each change). Word finding (`joins` / `gaps`) is under **Advanced settings**.
Press **Read page**: a progress bar under the button shows *Cleaning page… Finding words and reading… Correcting…*
(about 10–60 s; the XLM-R models are loaded per page to stay within the 8 GB memory limit). Wrong file types and
images that are too small get a plain explanation instead of an error trace.

**Screen 2, "Results"** (screen 1 is hidden):
- a top bar with **← Read another page** and one line saying what was used and what came out, e.g.
  *Auto-correct · word finding: joins · 3 lines · 17 words · 4 changed by the correction · 8.8 s*;
- a polite banner when the page was probably not read as text (ruled or dotted paper, pencil, a poor photo), with
  what to try;
- the final text, sized to its content, right to left (macOS SF Arabic / Geeza Pro, which cover all Sindhi
  letters), with a copy button and **Download text (.txt)**;
- in Suggest mode, the review cards and **Apply choices** right under the text;
- a **Pages** tab with the cleaned page and your photo shown large, each with a zoom (fullscreen) button, the colour
  legend (blue lines, orange words, green changed words) and *Show boxes*; a **Compare** tab with the recognizer's
  output next to the final text, changed words highlighted.

Uploading a new photo or pressing **← Read another page** clears every earlier result (and the latter clears the
upload), and *Apply choices* / *Show boxes* refuse to act on a page read from a different photo, so results never
show for another image than the current one (`tests/test_app.py`).

The display logic lives in `pipeline/present.py` (plain functions, tested in `tests/test_present.py`); `app.py` only
arranges it with a custom theme (calm blues and greys, rounded cards, light and dark mode, stacks to one column on
narrow windows).

**Suggest mode.** Nothing is changed automatically. Every word the corrector would change gets a review card: the
word's image, the recognizer's reading, and a choice of *keep original* (first) or the corrector's top-3 candidates
(a candidate that joins the word with the next one says so). **Accept all suggestions** and **Keep all originals** set
every card at once. **Apply choices** updates the final text, the highlighted
view, the boxes image and the `.txt` download; it runs no model, so it is instant. The command line has the same
modes: `run_pipeline.py --correct off|auto|suggest` (`--no-correct` = `off`); in `suggest` the `.txt` keeps the
recognizer's text and `<name>.json` lists the candidates under `"suggestions"`.

### Saved choices (opt-in)

Ticking **"Save my choices for future training"** before *Apply choices* stores the reviewed words of that page in a
separate top-level folder, outside `data/`:

```
user_feedback/<YYYYmmdd-HHMMSS>_<image name>/
  word_001.png, word_002.png, …   each word's crop from the cleaned page
  choices.jsonl                    per word: time, image, box, recognizer reading, candidates offered,
                                   chosen text, kept_original (true/false), word-finding mode
```

The folder is git-ignored (it contains your images). **Nothing in the project reads it, and it is never used for
training automatically.** Before any future training use, a person must check the records (a chosen candidate is
not necessarily the true word) and keep the writers of these pages out of every test set. The tests point the
folder elsewhere with `SINDHI_OCR_FEEDBACK_DIR`.

## How to get good results

What the system was built and measured on (docs/STEP6_testing.md):

- **Pen on plain, unruled paper.** Dark ballpoint or gel pen reads best.
- **An upright photo or scan**, the sheet filling the frame, evenly lit, no strong shadows or reflections.
- **Leave clear spaces between words** (or use `--words gaps` if the writer does).
- **Not supported: pencil, and ruled or dotted paper.** In Step 6 a faint pencil photo and a page written on dotted
  rulings were read as hundreds of false words (CER above 1). Expect the plausibility warning on such pages.
- Measured accuracy: about 2 in 3 characters right on clean pages of writers seen in training (some pages almost
  perfect), about 4 in 5 on a new writer with clear spacing. Far from 90% — check the output.

## Future work

- **Non-text ink filtering and ruling removal** (solid and dotted rulings, speckle, paper edges) before line
  finding. FINAL_TEST is now used up, so this needs **new held-out test pages** (pencil and ruled pages included).
- **More writers.** Every weak step (word joins, honest tuning of lever 2, the Step 4 error model) is limited by
  having real handwriting from only a few writers.
- **A line-level recognizer** (no word cutting; docs/STEP5_pipeline.md option 3), once several hundred more lines
  are transcribed.
- **Combining gap width with the recognizer score** when deciding joins (needs writers outside training).
- **Cheap data collection tip:** ask people to **copy typed Sindhi text by hand** (one sentence per line, pen on
  plain paper). The typed text is then the ground truth, so **no transcription is needed**; record the writer and
  keep whole writers out of training for testing.

## Data and attribution

Where every dataset came from, and what was changed, is recorded in `data/PROVENANCE.md`. Third-party data:

- **POS Tagging on Handwritten Sindhi Sentences** — Maria Soomro, Muhammad Khalid (University of Sindh), Mendeley
  Data, V1, 2024, doi:10.17632/phk66sgmp5.1, licensed **CC BY 4.0**. Kept in `data/raw/pos_handwritten/`. Its
  images have no transcriptions, so they are **not used** for training (never train on unverified recognizer drafts).
- Wikipedia-sd (`wikimedia/wikipedia`, 20231101.sd): CC BY-SA 3.0 + GFDL.
- Roboflow `sindhi-ocrf`: CC BY 4.0.
- `Sindhi-Pos-Tagging` (676 POS-tagged sentences): no licence found, origin unknown; used only inside this project.
