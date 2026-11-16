# 00 — Executive summary

**Project:** Preserving Sindhi Heritage Through AI — Handwritten Sindhi OCR
**Location:** `~/Downloads/NEW` (git, branch `main`, 35 commits, last `fbe4a5b` on 2026-11-16, **no remote**)
**Audit:** 2026-11-16, evidence-based (source code, configs, checkpoints, result files, logs, datasets re-counted).
Nothing in the project was modified, trained or deleted during the audit; only `PROJECT_HANDOVER/` was created.

## What it is

An offline system that converts a **photo of a handwritten Sindhi page into Unicode text**:

1. **Preprocess** the photo (background flattening + Sauvola binarisation, deskew, line/speck removal) —
   `preprocess/`, `pipeline/page.py::clean_page`.
2. **Find lines and reading order** — `tesseract_ft/boxes.py::line_structures`, `pipeline/layout.py::order_lines`.
3. **Group ink pieces into words and read them** with a **CNN + Transformer word recognizer** (8.1 M parameters,
   trunk pre-trained as a 52-letter classifier) — `pipeline/page.py::read_page`, `cnn_transformer/models.py::WordRecognizer`.
4. **Correct** with a **Sindhi-further-trained XLM-RoBERTa** (candidate ranking) and a **wrong-word detector** —
   `xlmr/`, `pipeline/page.py::Corrector`. Modes: Off / Auto-correct / Suggest (user reviews each change).
5. **Output** text + JSON; local **web app** (`app.py`, Gradio, 127.0.0.1) and **command line** (`run_pipeline.py`).

No Claude, AI API or paid service is needed to run, train or evaluate it.

## Key results (measured, see `08_…`)

| Measure | Result | Gate 0.90 |
|---|---|---|
| Preprocessing on synthetic pages (F1 / dot recall) | 0.956 / 0.993 | PASS |
| 52-letter classification (unseen sheets) | 82.25 % | FAIL (accepted) |
| Word images, writer never seen in training | character accuracy 0.912; words fully right 71.5 % → **79.2 %** after correction | FAIL on composed words (accepted) |
| Word boxes, unseen writer (F1) | 0.809 (gap rule 0.862) | FAIL (accepted) |
| Correction on synthetic text | 0.890 word accuracy, 2.07 % correct words changed | FAIL (accepted) |
| Whole pages, unseen writer | CER 0.222 | FAIL (accepted) |
| Final test, 7 held-out pages | CER 0.354 on writers H + J; **1.65** overall (two pencil/dotted pages read as hundreds of false words) | FAIL (accepted) |

**Honest bottom line:** the system works end to end and is useful on clear pen writing, but accuracy is far below
90 %. The main causes are measured: few writers, dot-level letter ambiguity, word segmentation (perfect word crops
CER 0.12 vs whole pages 0.24–0.44), and no handling of rulings/pencil.

## Data (verified counts, see `04_…`)

4,241 unique letter images (52 balanced classes; 77 % of the original files were duplicates and removed);
22 + 129 handwritten page photos with 1,472 verified word crops in total; 4.1 M words of Sindhi Wikipedia; 518 POS
sentences; 1,060 legal Q&A rows; 998 Mendeley sentence images (976 unique, **not used** — no transcriptions).
No train/test leakage found. **Licences are unknown** for the legal CSV, the POS corpus and the letter images.

## Most important findings of this audit

1. The repository has **no remote** and the models (8.8 GB) are not in git → **back up** (see `15_…` §3).
2. The legal CSV used for training lives **outside** the project (`~/Downloads/Sindhi_Legal_Dataset.csv`) and is not
   recorded in `data/PROVENANCE.md`.
3. Before publishing: text derived from the legal CSV and the unlicensed POS corpus is tracked in git; username paths
   appear in some result files; handwriting images need writer consent (`10_…`).
4. Some older doc numbers were wrong and are corrected here (handwritten word count 402 not ~450; Mendeley near
   duplicates 2 not 1; detector training 21 min not 16; recognizer training ~3 h 51 min).
5. A byte-identical copy of FINAL_TEST page `D6.` sits (excluded, unused) in the batch-2 raw folder — a hazard for
   future code.

## Files in this folder

| File | Contents |
|---|---|
| `00_EXECUTIVE_SUMMARY.md` | this page |
| `01_COMPLETE_PROJECT_INVENTORY.md` | folders, entry points, every important file with purpose/status, unused items |
| `02_SYSTEM_ARCHITECTURE.md` | architecture, data flow, model and sequence diagrams (Mermaid), execution flow, technologies |
| `03_MODEL_AND_ALGORITHMS.md` | every model (layers, parameters, training settings, metrics, loading) and algorithm |
| `04_DATASET_AUDIT_AND_PROVENANCE.md` | verified dataset counts, duplicates, splits, leakage, provenance and licences |
| `05_RESEARCH_PAPERS_AND_REFERENCES.md` | references found in the repo, method papers, Sindhi literature, IEEE list |
| `06_PROJECT_UNIQUENESS_AND_INNOVATION.md` | difficulty, existing work, real contributions, ready-made explanations |
| `07_INSTALLATION_AND_RUN_GUIDE.md` | setup, running, retraining, evaluating, working without AI tools |
| `08_EVALUATION_AND_RESULTS.md` | every metric and result with its source file, reliability, what to test next |
| `09_LIMITATIONS_AND_FUTURE_SCOPE.md` | current limitations vs proposals (CRNN, CTC, TrOCR, ViT, …) |
| `10_GITHUB_PUBLICATION_GUIDE.md` | secrets/privacy scan, include/exclude, permissions, licence advice |
| `11_BTECH_REPORT_CONTENT.md` | report chapters (abstract … conclusion) |
| `12_PRESENTATION_GUIDE.md` | 10-slide plan |
| `13_VIVA_QUESTIONS_AND_ANSWERS.md` | 52 Q&A + the 20 most important functions |
| `14_TROUBLESHOOTING.md` | real errors and fixes |
| `15_COMPLETE_HANDOVER_CHECKLIST.md` | status, backup list, verification commands |
| `README_DRAFT_FOR_GITHUB.md` | proposed public README (existing `README.md` unchanged) |
| `appendix/A_raw_file_inventory.md` | full machine inventory (every folder/file/checkpoint/log/import) |
| `appendix/B_raw_dataset_audit.md` | full dataset recount with methods |
| `appendix/C_raw_models_references_safety.md` | models with file:line evidence, references found, safety scan |
