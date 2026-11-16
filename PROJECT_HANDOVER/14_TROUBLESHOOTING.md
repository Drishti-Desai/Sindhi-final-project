# 14 — Troubleshooting

Problems that actually occurred while building this project, plus the checks that diagnose them. Commands assume
you are in `~/Downloads/NEW`.

## Start-up and environment

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch'` (or gradio, transformers…) | Not using the project's Python | Use `.venv/bin/python …` or `source .venv/bin/activate` first. Re-install: `python -m pip install -r requirements.txt`; check: `python -m pip check`. |
| `.venv` stops working after a Python upgrade | The venv is tied to Python 3.13.3 | Delete and recreate it (section 2 of `07_…`). Ask yourself before deleting anything else. |
| `[gate] 'step…' has not been run yet … missing` | A `results/gates/*.json` file is missing (e.g. a partial copy of the project) | Restore `results/gates/` from git (`git checkout -- results/gates`) or run that step. |
| `[gate] WARNING: 'step…' is FAIL but was accepted by user` | Normal: Steps 2–6 failed their 0.90 gates and were accepted by the owner | Nothing to fix; it is printed on purpose. |
| `OSError: Cannot find empty port in range: 7860-7860` | The app is already running (maybe in another terminal) | Use that one, stop it (`kill $(lsof -ti tcp:7860)`), or start on another port: `python app.py --port 7861`. |
| App shows an old layout | An older copy of `app.py` is still running | Ctrl+C it and start `app.py` again. |
| `OSError … Can't load … checkpoints/step4_xlmr_mlm/best` / `FileNotFoundError: …best.pt` / `…lexicon.tsv` | Model or lexicon files missing (not in git) | Restore them from your backup (list in `07_…` §3) or retrain/regenerate (`scripts/step4_prepare.py` makes `lexicon.tsv`). Run with `--correct off` to skip the correction models. |
| `SystemExit: Step 4 final variant is …; Corrector implements xlmr_sindhi_det` | `results/step4/val_comparison.json` was regenerated with another winner | Restore it from git, or extend `pipeline/page.py::Corrector` for the new variant. |
| Hugging Face tries to download something | `HF_HUB_OFFLINE` not set (only when importing `xlmr` modules outside `run_pipeline.py`/`app.py`) | `export HF_HUB_OFFLINE=1` or import `run_pipeline` first. |

## Memory (8 GB Mac)

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: MPS backend out of memory (… max allowed: 3.73 GiB)` during training | Batch too large for the cap set by `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | Lower `batch_size` and raise `grad_accum` in `configs/step2_cnn.yaml` / `configs/step4_xlmr.yaml` (this is how the detector was fixed: 16×2 → 8×4), keep gradient checkpointing on, close other apps. Raising the ratio risks heavy swapping. |
| Computer very slow, swapping | Two big jobs at once (e.g. training while the app runs) | Run one model job at a time. |

## Library version changes

| Symptom | Cause | Fix |
|---|---|---|
| `TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'warmup_ratio'` | transformers 5 removed `warmup_ratio` | Already fixed: `warmup_steps=0.06` (a float means a ratio). Keep the pinned versions in `requirements.txt`. |
| Gradio errors after upgrading Gradio | API differences between major versions | Re-install `gradio==6.32.0`. |

## Reading results

| Symptom | Likely cause | What to try |
|---|---|---|
| Red/amber banner "may not have been read reliably" | Ruled or dotted paper, pencil, speckle or a poor photo: rulings and fragments are read as words (`pipeline/page.py::plausibility`) | Pen on plain paper, sharp upright photo; `--erase-rules` for curved notebook rulings. Pencil and dotted paper are **not supported**. |
| Words glued together | Joins mode over-joins writers who leave clear gaps (Step 5 finding) | `--words gaps` / Advanced settings → Gaps. |
| Words split into pieces | Cramped writing; gap rule cuts inside words | Use the default `--words joins`. |
| Lines in the wrong order | Page not upright, or a strongly sloped line found as several pieces | Rotate the photo upright; check the boxes in the Pages tab / `page.json`. |
| Correction changed a correct word | The corrector changes ~2 % of correct legal-domain words and ~4 % on other topics | Use **Suggest** mode and keep the original. |
| "No handwriting was found" | Wrong side of the page, very faint, or tiny image | Retake the photo. |
| "This image is too small to read" / "not an image we can open" | Shorter side < 120 px, or unsupported/corrupt file | Use a JPG/PNG photo of normal size. |
| Sindhi letters show as boxes in the app (non-Mac) | No font with Sindhi letters installed | Install a Naskh font such as Noto Naskh Arabic (OFL) on that computer. |

## Debugging a single prediction

```bash
.venv/bin/python run_pipeline.py page.jpg --correct off --out debug_out/
python3 -c "import json; j=json.load(open('debug_out/page.json')); [print(w['box'], w['text'], w['logp']) for l in j['lines'] for w in l['words']]"
```
Each word shows its box and the recognizer's mean log-probability (closer to 0 = more confident). Compare with
`--correct auto` to see what the correction changed (`actions` in the JSON).

## Tests and measurements

| Symptom | Cause | Fix |
|---|---|---|
| `[step5] the final measurement has already been made` / `[step6] …` | Test runs are allowed once (protocol) | Do not delete the gate file to re-run; measure on new held-out pages instead. |
| A test fails after a code change | Behaviour changed | `python -m pytest -q -x` stops at the first failure; read the test to see what it expects. |
