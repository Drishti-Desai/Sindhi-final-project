# 07 — Installation and run guide (no AI assistant needed)

Everything here was checked against the project at commit `fbe4a5b`. The project needs **no Claude, no Claude API,
no Claude Code and no paid AI service** to run, train or evaluate. It needs no internet at run time
(`HF_HUB_OFFLINE=1` is set by `run_pipeline.py`; the web app binds to `127.0.0.1`).

---

## 1. Requirements

| Item | Requirement | Status |
|---|---|---|
| Operating system | macOS on Apple Silicon (tested: M2, macOS "Darwin 25") | **Tested** |
| | Linux / Windows | **NOT VERIFIED** (code uses only cross-platform libraries; the app's Sindhi fonts SF Arabic / Geeza Pro exist only on macOS — other systems fall back to "Noto Naskh Arabic" or "Segoe UI" if installed) |
| Python | 3.13.3 (the `.venv` was built with it) | Tested |
| RAM | 8 GB is enough for running the app and for every training script as configured | Tested |
| GPU | Optional. Apple GPU (MPS) is used if present (`cnn_transformer/train_utils.py::get_device("auto")`: CUDA → MPS → CPU). | MPS tested; CUDA and CPU-only **NOT VERIFIED** for speed |
| Disk | ~2.2 GB for the three models needed to run; ~9 GB for all checkpoints; raw data several GB | measured |
| Other software | git. Tesseract 5.5.2 (with `text2image`) only to regenerate the printed training renders or re-run Step 3 experiments — **not needed to run the app**. | — |

Python packages: `requirements.txt` (exact versions). SciPy is not listed by name; it is installed automatically as a
dependency of scikit-image (version 1.18.1 in the working `.venv`).

---

## 2. First-time setup on a new computer

```bash
# 1. go to the project folder
cd ~/Downloads/NEW

# 2. create and activate a virtual environment (Python 3.13)
python3.13 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 3. install the exact package versions
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4. check the installation
python -m pip check                  # expect: "No broken requirements found."
python -m pytest -q                  # expect: 79 passed
```

If you already have the working `.venv` in this folder, skip steps 2–3 and just run `source .venv/bin/activate`
(or call `.venv/bin/python …` directly, as below).

---

## 3. Models (must be present; they are NOT in git)

The app needs exactly these three items. Copy them from your backup into the same paths:

| Path | Size | What |
|---|---|---|
| `checkpoints/step2_recognizer_ft/best.pt` | 31 MB | word recognizer (with its character vocabulary) |
| `checkpoints/step4_xlmr_mlm/best/` | 1.1 GB | Sindhi-trained XLM-RoBERTa (masked LM) |
| `checkpoints/step4_xlmr_detector/best/` | 1.0 GB | "is this word wrong?" detector |

Also needed (small, in git): `data/preprocessed/step4/lexicon.tsv` (**not in git** — `data/preprocessed/` is
ignored; back it up or regenerate with `scripts/step4_prepare.py`), `results/step4/val_comparison.json`,
`results/step4/error_model.json`, `results/gates/*.json`, `configs/*.yaml`.

Quick check that everything needed is there:

```bash
ls -lh checkpoints/step2_recognizer_ft/best.pt checkpoints/step4_xlmr_mlm/best/model.safetensors \
       checkpoints/step4_xlmr_detector/best/model.safetensors data/preprocessed/step4/lexicon.tsv \
       results/step4/val_comparison.json results/step4/error_model.json
```

To run **without** the 2 GB correction models, use `--correct off` (command line) or "Off" (app): only the 31 MB
recognizer is then loaded.

---

## 4. Running

### Web app

```bash
cd ~/Downloads/NEW
.venv/bin/python app.py                 # then open http://127.0.0.1:7860
.venv/bin/python app.py --port 7861     # if 7860 is busy
```

Stop: press **Ctrl+C** in that terminal. From another terminal: `kill $(lsof -ti tcp:7860)`.

Using it:
1. **Screen 1** — drop a photo (JPG/PNG, upright, pen on plain paper) or press *Try an example page*; choose
   *Off*, *Auto-correct* (default) or *Suggest*; optionally open *Advanced settings* for word finding
   (*Joins* default, *Gaps* for clearly spaced writing); press **Read page** (about 2–60 s; measured on page A1:
   ~2 s with Off, ~9–10 s with Auto-correct).
2. **Screen 2** — final text with *Copy* and *Download text (.txt)*; *Pages* tab (cleaned page with boxes, your
   photo, zoom); *Compare* tab; in Suggest mode, review cards → **Apply choices**.
3. **← Read another page** to start again (clears everything).

Saving results: use *Download text (.txt)*; the cleaned/boxed page images have a download button.
Opt-in "Save my choices…" (Suggest mode) writes to `user_feedback/<timestamp>_<image>/`.

### Command line

```bash
.venv/bin/python run_pipeline.py page1.jpg page2.png --out pipeline_out/
.venv/bin/python run_pipeline.py page.jpg --correct off          # recognizer only
.venv/bin/python run_pipeline.py page.jpg --correct suggest      # no changes; candidates in the JSON
.venv/bin/python run_pipeline.py page.jpg --words gaps           # clearly spaced handwriting
.venv/bin/python run_pipeline.py page.jpg --erase-rules          # curved notebook rulings
```

Outputs: `pipeline_out/<name>.txt` and `pipeline_out/<name>.json`. A plausibility warning is printed when the page
was probably not read as text.

---

## 5. Retraining and evaluating (each stage checks the previous gate first)

| Goal | Command | Time on the M2 (from logs) |
|---|---|---|
| Fixed splits (once) | `.venv/bin/python scripts/prepare_splits.py` | seconds |
| Preprocess all datasets + Step 1 gate | `.venv/bin/python scripts/step1_preprocess.py` | ~1 min |
| Letter CNN experiment | `.venv/bin/python scripts/step2_train_cnn.py --run <name>`; promote: `--run <name> --eval --gate` | ~10–30 min per run |
| Printed word renders + word lists | `.venv/bin/python scripts/step2_make_synthetic.py` (needs the Tesseract program: it uses Tesseract's `text2image` renderer) | ~6 min |
| Real word crops (old pages) | `.venv/bin/python scripts/step2_extract_real_words.py` | short |
| Word recognizer from scratch | `.venv/bin/python scripts/step2_train_recognizer.py` | ~3 h 51 min (15 epochs; summed epoch times) |
| Fine-tune on batch-2 real words | `.venv/bin/python scripts/step2_train_recognizer.py --finetune` | ~2 h 7 min |
| Step 4 data, lexicon, error model | `.venv/bin/python scripts/step4_prepare.py` | minutes |
| Sindhi masked-LM training | `.venv/bin/python scripts/step4_train_mlm.py` | ~1 h 37 min (log timestamps) |
| Detector | `.venv/bin/python scripts/step4_train_detector.py` | ~21 min (trainer runtime 21 min 05 s) |
| Choose correction weights (validation) | `.venv/bin/python scripts/step4_compare.py` | ~1 h 20 min (scores every candidate with two XLM-R models, then a weight search) |
| VAL pages through the pipeline | `.venv/bin/python scripts/step5_eval.py` | ~7 min (three configurations) |
| Unit tests | `.venv/bin/python -m pytest -q` | ~7 s |

**Warning:** `scripts/step4_test.py`, `scripts/step5_eval.py --final` and `scripts/step6_final.py` are the
one-time test measurements. They already ran; `step5_eval.py --final` and `step6_final.py` refuse to run again
because the gate file exists. Re-running a test after changing a model would no longer be a clean test.

**To evaluate a new recognizer:** train it into a new folder (change `checkpoints` paths in
`configs/step2_cnn.yaml` or the `RECOGNIZER` constant in `scripts/step3_baseline.py`), then run
`scripts/step5_eval.py` (VAL pages, development only). For a fair test you need **new held-out pages** (FINAL_TEST
and the unseen-writer pages are used up).

**To change the model the app uses:** the recognizer path is `RECOGNIZER` in `scripts/step3_baseline.py`
(`load_recognizer()`); the correction models are `mlm.out` and `detector.out` in `configs/step4_xlmr.yaml`
(the app loads `<out>/best`); the correction weights come from `results/step4/val_comparison.json`.

---

## 6. Working on the code without an AI assistant

| Task | Where | How |
|---|---|---|
| Understand a function | open the file; every module starts with a docstring explaining it | `grep -n "def <name>" -r --include=*.py .` finds it |
| Change preprocessing | `configs/step1_preprocess.yaml` (thresholds), `preprocess/ops.py` (operations), `preprocess/pipeline.py` (order) | run `pytest tests/test_preprocess.py -q`, then `scripts/step1_preprocess.py` |
| Change line/word finding | `tesseract_ft/boxes.py`, `pipeline/layout.py`, `pipeline/page.py` | `pytest tests/test_step3.py tests/test_step5.py -q`; `scripts/step5_eval.py` (VAL) |
| Add training images | put copies in `data/raw/…`, add rows to the split CSVs in `data/splits/…`, record origin in `data/PROVENANCE.md` | re-run the matching `scripts/step2_extract_*` and training script |
| Modify the recognizer | `cnn_transformer/models.py::WordRecognizer`, settings in `configs/step2_cnn.yaml: recognizer` | train with `--small` first (quick end-to-end check) |
| Change the correction | `xlmr/correct.py` (candidates, `decide`), `xlmr/lm.py` | `pytest tests/test_step4.py tests/test_suggest.py -q`; `scripts/step4_compare.py` |
| Update the web app | `app.py` (layout, CSS, theme), `pipeline/present.py` (what is shown) | `pytest tests/test_app.py tests/test_present.py -q`; restart `app.py` |
| Run one test | `.venv/bin/python -m pytest tests/test_step5.py::test_plausibility_warning -q` | |
| See history | `git log --oneline`, `git show <commit>` | each commit message explains the step |
| Debug a prediction | run `run_pipeline.py page.jpg --correct off`, open `pipeline_out/page.json` (boxes + readings per word); in the app use the Pages tab (boxes) | see `14_TROUBLESHOOTING.md` |

Essential commands:

```bash
source .venv/bin/activate                 # use the project's Python
python -m pytest -q                       # run all tests
python run_pipeline.py IMG --out DIR      # read a page
python app.py                             # web app
git status / git diff / git log --oneline # what changed
git add -A && git commit -m "message"     # save a version
python -m pip install -r requirements.txt # (re)install packages
```
