# 10 — GitHub publication guide

**Nothing has been pushed.** The repository has **no remote** (`git remote -v` is empty) and has 35 local commits.
This guide lists what to check, include and exclude before publishing. An improved public README draft is in
`README_DRAFT_FOR_GITHUB.md` (the existing `README.md` was not changed).

---

## 1. Security and privacy scan (2026-11-16)

Full evidence: `appendix/C_raw_models_references_safety.md`, Part C.

| Check | Finding | Action |
|---|---|---|
| API keys, tokens, passwords | **None found** in tracked files, untracked files or git history. `.env` does not exist (and never was committed). `.env.example` has two **empty** placeholders (`RF_API_KEY`, `HF_TOKEN`). | none |
| E-mail addresses | none in files; every commit's author is `drishti <drishti@example.com>` (placeholder) | set your real git identity if you want commits attributed (`git config user.email …`) |
| Absolute paths with your macOS username | `/Users/<your-username>/…` appears thousands of times in `results/step1/report_letters52.csv`, `report_roboflow_sindhi_ocrf.csv`, `report_handwritten.csv`, and once each in `results/step4/mlm_log.json`, `detector_log.json` | make the paths relative or remove those files before publishing |
| Names | "Drishti-Desai" in `data/splits/handwritten_batch2/manifest.csv` (folder names), `scripts/prepare_batch2.py`, `data/PROVENANCE.md`; third-party authors only as CC BY attribution; writers only as letter codes | decide whether to keep |
| Images of real handwriting (tracked) | `results/step2/real_words_review/*.png`, `results/step1/comparison_real_pages.png`, `comparison_real_letters.png`, `results/step2/cnn_errors.png`, `cnn_augment_samples.png` | get writers' consent or exclude |
| Text derived from the legal CSV (licence unknown), **tracked** | `data/splits/step4/legal_*.txt`, `corrupted_domain_*.jsonl`, `real_val.jsonl`, `data/splits/pos_handwritten/images.csv`, `data/raw/handwritten_batch2/text_corrected/*.txt` (verbatim rows), `results/step5/val_metrics.json`; also the page texts in `data/raw/handwritten/text_gt/` (news text copied by hand) | treat as non-redistributable until the licence is known |
| Text from `pos_text` (no licence; PROVENANCE says never redistribute), **tracked** | `data/splits/step4/pos_*.txt`, POS lines in `corrupted_domain_*.jsonl`, `images.csv` | remove from the public tree or get permission |
| Wikipedia-derived text, tracked | `wiki_val.txt`, `wiki_test.txt`, `corrupted_wiki_*.jsonl`, `data/splits/step2_words/*.txt` | keep only with CC BY-SA 3.0 attribution + share-alike |
| Large files | tracked tree **≈ 11 MB, 259 files**; largest 1.69 MB; no model weights tracked | fine for GitHub |
| Licence file | **no `LICENSE`** at the repository root | add one (section 4) |
| Dependency gap | SciPy used but not pinned (arrives via scikit-image) | optionally pin `scipy==1.18.1` |

---

## 2. What to include / exclude

| Item | Include? | Why |
|---|---|---|
| Source code: `app.py`, `run_pipeline.py`, `pipeline/`, `preprocess/`, `cnn_transformer/`, `tesseract_ft/`, `xlmr/`, `common/`, `scripts/`, `tests/` | **Yes** | your own code |
| `configs/`, `requirements.txt`, `pytest.ini`, `.gitignore`, `.env.example` | Yes | needed to run |
| `docs/`, `PLAN.md`, `CLAUDE.md`, `PROJECT_HANDOVER/` | Yes (review first) | documentation; `CLAUDE.md` is only guidance for an AI tool — keep or drop, it contains no secrets |
| `results/gates/`, `results/step*/` metrics (json/csv/png) | Yes | evidence for reported results |
| `results/step6/pages/*.txt|json`, `results/step4/test_real_changes.csv`, `results/step5/*_metrics.json` "readings" | **Check** | contain recognized/true text of test pages derived from the legal CSV (licence unknown) |
| `checkpoints/` (8.8 GB) | **No** (too large for git) | publish the three runtime models separately (GitHub Releases ≤ 2 GB per file, Hugging Face Hub, or Zenodo) **only after** the data-licence questions below are answered, because models trained on data carry that data's terms |
| `data/raw/` | **No** | third-party and personal handwriting; licences unknown for several sets |
| `data/preprocessed/`, `data/synthetic/` | No | derived from raw data; regenerable |
| `data/splits/letters52/`, `handwritten/`, `handwritten_batch2/` (manifests, file names, labels) | Check | contain file names (e.g. WhatsApp image names) and labels; no images |
| `data/splits/step4/*.txt`, `*.jsonl` (tracked) | **Check / likely exclude** | sentences from the legal CSV (**licence unknown**) and Wikipedia (**CC BY-SA 3.0: share-alike + attribution required if redistributed**) |
| `data/raw/handwritten/text_gt/*.txt`, `handwritten_batch2/text_corrected/` (tracked) | Check | page texts copied from the legal CSV |
| `data/raw/pos_text/` notebooks + CRF model | **No** | third-party, no licence |
| `logs/`, `results/*/cache/`, `results/*/debug/`, `__pycache__/`, `.DS_Store` | No | by-products (already git-ignored) |
| `user_feedback/` | **Never** | user photos of handwriting |

## 3. Unresolved permissions (must be answered before publishing data or models)

| Item | Question |
|---|---|
| `Sindhi_Legal_Dataset.csv` | Who created it? Under what licence? (Used for training text and as the source of the handwritten page texts.) |
| letters52 (Roboflow-export filenames) | Who photographed and labelled the letters? Is there a Roboflow project licence? |
| pos_text (`Sindhi-Pos-Tagging-main`) | Original repository URL and licence (README has a placeholder URL and no licence). |
| Handwritten pages (writers A, D, H, J, batch-2 writers, purple-pen writer) | Written consent from each writer to publish images of their handwriting. |
| Mendeley `pos_handwritten` | CC BY 4.0 allows reuse with attribution — re-check the licence on the Mendeley page before redistribution. |
| Wikipedia-derived text | CC BY-SA 3.0: any redistributed derived text must keep the same licence and attribution. |
| xlm-roberta-base | Released by Facebook AI under the MIT licence (as published on its Hugging Face model card — re-check the card). |

## 4. Licence recommendation for the code

- Code: **MIT** or **Apache-2.0** are suitable for your own code (all libraries used — PyTorch BSD-style,
  Transformers Apache-2.0, OpenCV Apache-2.0, scikit-image BSD, Gradio Apache-2.0, NumPy BSD, Pillow HPND/MIT-CMU —
  are compatible with both; verify each library's licence file in `.venv/lib/python3.13/site-packages/*/` if needed).
- Data and models: **do not attach a licence until section 3 is resolved.** If published later, documentation and
  text derived from Wikipedia must be CC BY-SA; the Mendeley data needs attribution.
- Add a `LICENSE` file (not created during this audit) and a "Data and model licences" section in the README.

## 5. Checklist before the first push

1. Resolve section 3; decide which data files stay out of git (`git rm --cached <file>` keeps the local copy but stops
   tracking it; history still contains it — if sensitive data was ever committed, the history must be rewritten or
   a fresh repository created).
2. Review the secrets/privacy findings in section 1 and remove absolute paths/usernames where you do not want them.
3. Add `LICENSE` and the README sections (description, features, architecture, install, usage, datasets, models,
   results, limitations, future scope, references, contributors, licence).
4. Fix portability issues before others clone it: `.PNG` vs `.png` names in `data/splits/handwritten_batch2/manifest.csv`
   (Linux is case-sensitive); the app's Sindhi fonts are macOS fonts.
5. Decide how models are distributed (Release asset / Hugging Face / Zenodo) and write the download step in the README.
6. Create the remote and push: `git remote add origin <url>` then `git push -u origin main` (do this yourself after
   the review).
7. Check the repository size: tracked files are listed in `appendix/A_raw_file_inventory.md`.

## 6. `.gitignore` and `requirements.txt`

Both already exist and are suitable. `.gitignore` excludes `data/raw/*` (except `handwritten/text_gt`), processed and
synthetic data, `checkpoints/`, `logs/`, caches, debug figures, `pipeline_out/`, `user_feedback/`, `.env`, `.venv/`.
`requirements.txt` pins every package with exact versions (Python 3.13.3). SciPy arrives through scikit-image;
Tesseract (system program) is only needed for Step 3 experiments; `pycrfsuite` only for the third-party POS
notebooks.
