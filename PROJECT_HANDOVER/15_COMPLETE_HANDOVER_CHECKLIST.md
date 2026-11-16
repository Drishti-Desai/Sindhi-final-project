# 15 — Complete handover checklist

State on 2026-11-16, commit `fbe4a5b`. `PROJECT_HANDOVER/` itself is **not committed yet** (waiting for your review).

## 1. Status of every part

| Part | Status | Evidence |
|---|---|---|
| Step 1 preprocessing | **Implemented and working**; gate PASS (synthetic benchmark) | `results/gates/step1_preprocessing.json`, 26 tests |
| 52-letter CNN | **Implemented and working**; 82.25 % (gate FAIL, accepted); used only as pre-training | `results/gates/step2_cnn.json` |
| Word recognizer | **Implemented and working**; unseen-writer word 1 − CER 0.912 (gate FAIL on composed words, accepted) | `results/gates/step2_cnn_transformer.json` |
| Box finding (own gap rule, Tesseract comparison) | **Implemented and working** (gap rule via `--words gaps`); Tesseract evaluated, not used | `results/gates/step3_tesseract.json` |
| XLM-R correction (auto + suggest) | **Implemented and working**; gate FAIL accepted | `results/gates/step4_xlmroberta.json`, tests |
| Pipeline (`run_pipeline.py`), joins, reading order | **Implemented and working**; gate FAIL accepted | `results/gates/step5_pipeline.json` |
| Final test | **Done once**; gate FAIL accepted | `results/gates/step6_final.json` |
| Web app (`app.py`) | **Implemented and working** (checked in a browser on a training page; 2 test files) | `tests/test_app.py`, `tests/test_present.py` |
| Plausibility warning | **Implemented**; designed after seeing FINAL_TEST, so **not independently tested** | `pipeline/page.py::plausibility` |
| Opt-in saving of reviewed words | **Implemented and tested**; folder `user_feedback/` does not exist yet (never used for real) | `tests/test_suggest.py` |
| Sindhi hard rules | Implemented, **not kept** (no gain) | `xlmr/rules.py` |
| Step 3 levers 1–2 | Implemented, kept as an option, **not used** | `tesseract_ft/cuts.py` |
| Linux / Windows / CPU-only / no-network run | **Not tested** | — |
| Beam search, line-level recognizer, TrOCR, text/non-text filter, rotation detection | **Not implemented** (future work) | `09_…` Part B |
| Literature review | **Partial** (web search only) | `06_…` §3 |

## 2. Answers to the handover questions

- **Completely implemented:** preprocessing, letter CNN, word recognizer, own box finder, correction (auto/suggest),
  pipeline CLI, web app, tests, evaluation of every step.
- **Partially implemented:** handling of difficult pages (only a warning); evaluation (small test sets, one unseen
  writer); dataset documentation (several licences unknown).
- **Not implemented:** see the last rows of the table above.
- **Currently working:** `python app.py`, `python run_pipeline.py`, `python -m pytest -q` (79 passed on 2026-11-16).
- **Still needs testing:** a new held-out set of new writers; pencil/ruled pages after any fix; other operating
  systems; offline operation with the network switched off (the code sets `HF_HUB_OFFLINE=1` and Gradio analytics
  off, but this was not tested with Wi-Fi disabled — **NOT VERIFIED**).
- **Saved models:** see `03_…`. Runtime: `checkpoints/step2_recognizer_ft/best.pt`, `checkpoints/step4_xlmr_mlm/best/`,
  `checkpoints/step4_xlmr_detector/best/`. Training inputs: `checkpoints/step2_cnn/best.pt`,
  `checkpoints/step2_recognizer/best.pt`, `checkpoints/xlmr_base/`.
- **Datasets required:** to run — none except `data/preprocessed/step4/lexicon.tsv` (and A1 for the example button);
  to retrain — letters52, handwritten, handwritten_batch2, pos_text, Wikipedia-sd, and **`~/Downloads/Sindhi_Legal_Dataset.csv`
  (outside the project folder!)**.
- **Dependencies required:** `requirements.txt` (Python 3.13.3); SciPy via scikit-image; Tesseract only for regenerating
  printed renders / Step 3 experiments.
- **Essential files to run:** code folders (`pipeline/`, `preprocess/`, `cnn_transformer/`, `tesseract_ft/`, `xlmr/`,
  `common/`, `scripts/step3_baseline.py`), `app.py`, `run_pipeline.py`, `configs/`, `results/gates/`,
  `results/step4/val_comparison.json`, `results/step4/error_model.json`, `data/preprocessed/step4/lexicon.tsv`, the 3
  runtime models.
- **Can be excluded from GitHub:** `checkpoints/`, `data/raw/`, `data/preprocessed/`, `data/synthetic/`, `logs/`,
  caches, debug figures, `user_feedback/`; and, until licences are clear, the tracked legal/POS-derived text files
  (`10_…` §1–2).
- **Claude-specific dependencies:** **none.** No Claude API, Claude Code or paid AI service is used by the code.
  `CLAUDE.md` is only a guidance file for AI tools; deleting it changes nothing.
- **Run without internet:** designed to (local models, `HF_HUB_OFFLINE=1`, app on 127.0.0.1); installing packages
  the first time needs internet.
- **Run without retraining:** yes, as long as the 3 runtime models and `lexicon.tsv` are present.

## 3. What exactly to back up (not in git!)

| Item | Size | Why |
|---|---|---|
| The whole project folder `~/Downloads/NEW` (incl. `.git/`) | ~11 GB with checkpoints | code, history, results |
| **Minimum** if space is short: `.git/` + working tree without `checkpoints/` and `data/synthetic/`, **plus** `checkpoints/step2_recognizer_ft/best.pt`, `checkpoints/step4_xlmr_mlm/best/`, `checkpoints/step4_xlmr_detector/best/` | ~2.2 GB | enough to run the app |
| `checkpoints/step2_cnn/best.pt`, `checkpoints/step2_recognizer/best.pt`, `checkpoints/xlmr_base/` | ~1.2 GB | needed to retrain |
| `data/raw/` | ~260 MB | original data copies (several cannot be re-downloaded) |
| `data/preprocessed/step4/lexicon.tsv` | 0.6 MB | needed at run time, not in git |
| **`~/Downloads/Sindhi_Legal_Dataset.csv`** | 0.2 MB | training text, lives **outside** the project |
| Original folders the data came from (`~/Downloads/untitled folder/SINDHI-PROJECT`, `~/Desktop/new data`) | — | provenance |
| `.venv/` | 1.7 GB | optional; can be recreated from `requirements.txt` |

The repository has **no remote**: until you push it somewhere, your backup is the only copy of the history.

## 4. Before closing Claude

1. Read `PROJECT_HANDOVER/00_EXECUTIVE_SUMMARY.md` and spot-check the other files.
2. Commit the handover when you approve it:
   `git add PROJECT_HANDOVER && git commit -m "Project handover documentation"`
3. Back up the items in section 3 (external drive or cloud storage).
4. Optionally create a private GitHub repository and push (follow `10_…` first).
5. Run the verification commands below once yourself, without any AI tool.

## 5. Verify the environment and launch (exact commands)

```bash
cd ~/Downloads/NEW
.venv/bin/python --version                 # Python 3.13.3
.venv/bin/python -m pip check              # No broken requirements found.
.venv/bin/python -m pytest -q              # 79 passed
ls -lh checkpoints/step2_recognizer_ft/best.pt \
       checkpoints/step4_xlmr_mlm/best/model.safetensors \
       checkpoints/step4_xlmr_detector/best/model.safetensors \
       data/preprocessed/step4/lexicon.tsv
.venv/bin/python run_pipeline.py data/raw/handwritten/images/A1.png --out /tmp/ocr_check
cat /tmp/ocr_check/A1.txt                  # three lines of Sindhi text
.venv/bin/python app.py                    # open http://127.0.0.1:7860 ; stop with Ctrl+C
```
