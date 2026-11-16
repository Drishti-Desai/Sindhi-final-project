# Step 1 — Preprocessing

Goal: whatever the background, output **dark text on pure white**, saved as a grayscale PNG
(values only 0 and 255), with thin strokes and diacritic dots kept.

## 1. Where the data goes

```
data/
  raw/                                   input (paths in configs/paths.yaml)
    letters52/<old_split>/<class_id>/*.jpg     class_id = 1..52, see configs/letter_map.yaml
    handwritten/images/<stem>.png              e.g. A1.png, D1..png
    handwritten/text_original/<stem>.txt       source text (UTF-8), one file per image
    handwritten/text_gt/<stem>.txt             draft ground truth (line breaks follow the image)
    roboflow_sindhi_ocrf/{train,valid,test}/*.jpg + _annotations.coco.json
  splits/
    letters52/manifest.csv   one row per UNIQUE image (hash, class_id, relpath, sheet, batch, n_copies)
    letters52/split.csv      relpath, class_id, sheet, batch, split (train/val/test, grouped by sheet)
    handwritten/pairs.csv    image, text files, writer, story_group, split (TESS_TRAIN/TESS_TEST/FINAL_TEST)
  preprocessed/                          output
    letters52/<class_id>/<original stem>.png   + index.csv (raw_relpath -> relpath, class_id, split)
    handwritten/<stem>.png
    roboflow_sindhi_ocrf/<split>/<stem>.png    + _annotations.coco.json (file names -> .png, boxes unchanged)
```

Labels:
- **Letters:** the class is the folder name (`1`..`52`); `configs/letter_map.yaml` maps it to the glyph.
- **Handwritten:** the text file has the same stem as the image.
- **Roboflow:** labels are COCO word boxes. Geometry is kept identical (no deskew, no resize), so the boxes stay valid.

Accepted inputs: any format Pillow reads (PNG/JPG/TIFF/…), including RGBA (transparency is flattened onto white).

## 2. Commands

```bash
source .venv/bin/activate
python scripts/prepare_splits.py              # once: dedup manifest + fixed splits (seed 42)
python scripts/step1_compare_methods.py       # evidence: method benchmark + real-image grids (~5 min)
python scripts/step1_preprocess.py            # all datasets + gate (~1 min on 7 cores)
python scripts/step1_preprocess.py --small    # quick check: 20 images per dataset, no gate
python -m pytest tests/test_preprocess.py -q  # unit tests (26)
```

In code:
```python
from preprocess import preprocess_image
result = preprocess_image("page.jpg")        # profile auto: glyph if longer side <= 160 px
result.image                                 # uint8, text 0, background 255
result.report                                # stats, steps applied, deskew angle, timing
```

## 3. What happens to an image

Every operation is a separate function in `preprocess/ops.py`; `preprocess/pipeline.py`
decides which ones to run from measured statistics:

| Step | Runs when | Function |
|---|---|---|
| Flatten transparency onto white | image has alpha | `flatten_alpha` |
| Grayscale: best-separating channel | always (`gray_mode: auto`) | `to_gray` |
| Upscale ×(128/h) | glyph profile | — |
| Downscale to ≤ 2400 px | page longer than that | `resize_max_side` |
| Invert | text lighter than its surroundings (page profile) | `is_dark_background`, `invert` |
| Keep only the paper sheet | page photo shows a darker surface around a sharp-edged sheet (`remove_desk`) | `find_paper_mask` |
| Denoise (non-local means) | noise σ > 8 | `noise_sigma`, `denoise` |
| Illumination flattening | always (harmless when even) | `estimate_background`, `flatten_illumination` |
| CLAHE | contrast (p99−p1)/255 < 0.40, or always for glyphs | `apply_clahe` |
| Binarise | pages: Sauvola · glyphs: normalise + fixed threshold | `binarize`, `normalize_contrast` |
| Deskew | pages, \|angle\| ≥ 0.4° | `estimate_skew`, `rotate` |
| Remove ruled lines / frames | pages | `remove_lines`, `remove_line_fragments` |
| Remove edge blobs (desk, fingers, binders) | pages | `remove_border_artifacts` |
| Remove paper-edge bars | glyphs | `find_edge_bars` |
| Remove specks smaller than a dot | always | `remove_small_components` |

All parameters are in `configs/step1_preprocess.yaml`.

## 4. How the method was chosen (evidence)

Real photos have no pixel-level ground truth, so methods were compared on synthetic Sindhi
text with a **known ink mask** (`preprocess/synthetic.py`). The 12 page conditions are:
clean, transparent, coloured, gradient, shadow, texture, low contrast, dark, noise, ruled
lines, faint pencil, and a page photographed on a desk. The 60 letter crops were generated to match statistics **measured on
400 real crops**: 22–77 px high, ink only 10–45% darker than the paper. Each condition was also
checked visually on real images (`results/step1/comparison_real_*.png`).

Summary (full table: `results/step1/method_comparison.md`):

| method | page mean F1 | page worst F1 | page dots | glyph F1 | glyph dots |
|---|---|---|---|---|---|
| otsu_raw (old project) | 0.805 | 0.214 (shadow) | 0.999 | 0.170 | 1.000 |
| **rembg_otsu (option b)** | 0.158 | 0.039 | 0.393 | 0.231 | 0.119 |
| flatten_otsu | 0.970 | 0.822 (pencil) | 1.000 | 0.608 | 1.000 |
| flatten_adaptive | 0.886 | 0.453 (pencil) | 1.000 | 0.726 | 0.889 |
| **flatten_sauvola** (pages) | 0.956 | **0.931** | 0.993 | 0.307 | 0.267 |
| **flatten_normalized_fixed** (glyphs) | 0.957 | 0.899 | 0.989 | **0.770** | 0.933 |
| oracle (best global threshold, chosen with GT) | | | | 0.791 | |

- **rembg is unsuitable.** It is trained to cut out a salient object (people, products) and treats text on paper as background: it loses 60–90% of the dots. Option (a) wins clearly.
- **Pages: Sauvola** after flattening has the best *worst case* (0.931). Selection is by worst-condition F1 because a new image can have any background; Otsu has a slightly higher mean but drops to 0.82 on faint pencil, which is common in this data (writers D and H).
- **Letter crops** need their own recipe: a background kernel about half the crop size, per-crop contrast normalisation, and a fixed threshold. It reaches 97% of the oracle ceiling. The ceiling is only 0.79 because the crops are tiny and blurry: the lost information cannot be recovered by any threshold.
- Parameters were tuned on synthetic seeds 100+; the benchmark and gate use seed 7 (unseen samples).

## 5. Accuracy definition and pass criteria (gate `results/gates/step1_preprocessing.json`)

The brief sets no accuracy target for Step 1, so I defined these checks:

| check | definition | pass |
|---|---|---|
| page_mean_f1 | text-pixel F1 vs ground-truth mask, mean over 12 page conditions | ≥ 0.90 |
| page_dot_recall | share of ground-truth dots with ≥ 50% of their pixels kept | ≥ 0.90 |
| glyph_f1_vs_oracle | glyph F1 ÷ oracle F1 (60 crops) | ≥ 0.95 |
| glyph_dot_recall | as above, letter crops | ≥ 0.90 |
| deskew error | mean \|estimated − true\| on tilted pages (2–7°) | ≤ 1.0° |
| configured = chosen | config uses the methods the benchmark selects | yes |
| completeness | every image processed, 0 failures | yes |
| unit tests | `tests/test_preprocess.py` | all pass |

Why glyph F1 is measured relative to the oracle: an absolute 0.90 is impossible for *any*
method on crops this blurry (the oracle itself is 0.79). This is a definition, not a relaxed
threshold; the absolute value (0.770) is also stored in the gate file.

Step 2 calls `require_gate("step1_preprocessing")` and refuses to start unless it is PASS.

## 6. Outputs, logs, results

| what | where |
|---|---|
| cleaned images | `data/preprocessed/<dataset>/` |
| per-image stats + steps applied | `results/step1/report_<dataset>.csv` |
| before/after samples | `results/step1/debug/<dataset>/*_before_after.png` (not in git) |
| method benchmark | `results/step1/method_comparison.{md,json}`, `comparison_real_{pages,letters}.png` |
| run summary | `results/step1/preprocess_summary.json` |
| logs | `logs/step1/` |
| gate | `results/gates/step1_preprocessing.json` |

## 7. Known limitations (honest list)

- **D2 (FINAL_TEST):** replaced by the user with a clearer photo on 2026-10-17. The desk around the sheet is now removed by `find_paper_mask`, which was validated on the synthetic `desk` condition (F1 0.95) and checked not to fire on shadows. A bright desk reflection at the top edge survives as speckle, and the faint pencil of lines 1–2 is still thin and broken. Nothing was tuned on D2's text.
- `find_paper_mask` decides "sheet vs shadow" by edge sharpness: synthetic sheets score 0.47–0.50, shadows at most 0.34, and the threshold is 0.40. If it ever wrongly fired on a strongly shadowed page, it would blank text in the shadow; `remove_desk: false` turns it off.
- Some line remnants stay: faint dashed ruled lines on D4, a paper-fold band on J4 (FINAL_TEST), wavy hand-drawn frames on H images, and dark wedges at the left of A5. These rules are deliberately conservative, because removing a real stroke is worse than leaving a line fragment.
- A5's top edge contains a cut-off line of other writing. It is real ink and stays, but it is not transcribed.
- About 1 in 200 letter crops is extremely faint (≈10% ink contrast) and comes out fragmented.
- Letter crops are always treated as dark ink on paper (`polarity: dark_ink`). Auto-detection wrongly inverted 3 of 4,241 crops, and every crop source in this project is dark-on-light. Pages keep automatic dark/light detection.
- The synthetic benchmark renders letters without Arabic joining (Pillow lacks libraqm). This is fine for measuring strokes and dots, but it is not real cursive.
