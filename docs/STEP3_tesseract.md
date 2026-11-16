# Step 3 — Tesseract 5.5.2: word and line boxes

**Status (2026-11-14): gate FAIL, accepted by the user** (final method chosen on TRAIN+VAL: lever 2, unseen-writer
word-box F1 **0.809** < 0.90; section 8). Step 5 uses our own gap-rule box finder (section 9). Nothing was
downloaded or fine-tuned.

## 1. What Tesseract is for here

The Step 2 recognizer reads single words well **when it is given correct word boxes** (unseen writer: 1 − CER
0.912). On a real page nothing in the project finds those boxes yet: the batch 2 crops were cut with the known
text. So Step 3 is gated on **word-box quality** (the user's choice), with line boxes and text CER/WER reported.

## 2. Data (no new labelling)

| use | pages | ground truth |
|---|---|---|
| tuning (choose settings only) | 110 batch-2 TRAIN + VAL pages | 1,175 kept word crops (QC-passed, ~88–92% correct); flagged crops = "don't care" |
| **gate test: unseen writer** | 7 purple-pen pages | 144 word boxes checked by eye |
| also reported (writers seen in training) | TESS_TEST H1, J2, J3 | 56 verified word boxes (recomputed and checked pixel-for-pixel against the saved crops) |

Every page is rebuilt exactly as it was cut (`tesseract_ft/pages.py`): turned upright, Step 1, circled numbers
removed. Box finders see only the kept ink. FINAL_TEST pages are never read.

## 3. Commands

```bash
.venv/bin/python scripts/step3_baseline.py     # ~35 min on the M2 (4 Tesseract processes); no install, no training
.venv/bin/python scripts/build_lexicon.py      # word list for lever 2 (test-overlapping sentences removed)
.venv/bin/python scripts/step3_levers.py       # levers 1-2, ~1 h; measures the tests ONCE (use --dry to check code)
.venv/bin/python -m pytest tests/test_step3.py -q
```
Settings: `configs/step3_tesseract.yaml`. Outputs in `results/step3/`: `baseline_metrics.json`,
`baseline_tuning_grid.csv` (all 416 settings), `baseline_per_page.csv`, `baseline_text.csv` (reference and
hypothesis per page), `debug/<method>/*.png` (green = ground truth, blue = matched, red = false positive).

## 4. How boxes are scored (`tesseract_ft/boxes.py`)

- Every predicted box is tightened to the ink inside it (empty boxes are dropped).
- One-to-one matching at **IoU ≥ 0.5**. Unmatched predicted boxes are false positives, unless most of their ink
  is "don't care" ink (crossed-out words, excluded crops, ink no ground-truth word covers).
- Precision, recall, F1; also split/merge counts and words with no box anywhere near them ("not found").
- Text: page-level CER/WER of Tesseract's own text, and of the Step 2 recognizer reading the words inside each
  method's boxes (box crops, reading order: lines top to bottom, words right to left).

## 5. Methods compared

| method | what | settings chosen on tuning pages |
|---|---|---|
| Tesseract, whole page | `tesseract --psm N`, word boxes from its TSV | `ara`, psm 11 (sparse), binary, letters scaled to ~40 px |
| Tesseract inside our lines | our line finder; each line alone to Tesseract (psm 13, raw line) | `ara`, psm 13, grayscale, ×1.5 |
| our own finder (no Tesseract) | line finder + word-gap rule: cut where gap > max(0.3 × letter height, 0.75 × the line's median gap) | slope-tracking lines, 0.3, 0.75 |

Grid: 4 installed models (`snd` is the 2017 "fast" model; `urd`, `ara`, `script/Arabic`) × psm × binary/gray ×
scale (fixed 1/1.5/2 or letters normalised to 20/30/40 px); our finder: 2 line finders × 11 × 8 gap rules.

## 6. Results (2026-11-14) — gate FAIL

**Unseen writer (gate), 144 words:**

| method | word P | word R | **word F1** | line F1 | not found | recognizer CER on these boxes | Tesseract text CER |
|---|---|---|---|---|---|---|---|
| Tesseract, whole page | 0.892 | 0.688 | 0.777 | 0.629 | 30 | 0.413 | 0.487 |
| Tesseract inside our lines | 0.837 | 0.819 | **0.828** (gated) | 0.944 | 2 | 0.249 | 0.314 |
| our own finder | 0.856 | 0.868 | **0.862** | 0.944 | 0 | 0.212 | – |
| ground-truth boxes (reference) | | | | | | **0.081** | |

**TESS_TEST (H1, J2, J3; writers seen in training), 56 words:** word F1 Tesseract 0.635, Tesseract inside our
lines 0.719, our finder 0.775. Recognizer CER: 0.482 / 0.451 / 0.336; with ground-truth boxes 0.227.

**Tuning pages (110 pages, other writers):** word F1 0.550 / 0.660 / 0.662. The purple-pen writer is neat; most
other writers' pages are much harder (and their ground truth is ~10% noisy).

### What the errors are
- **Whole-page Tesseract loses whole lines** (line F1 0.63; 30 of 144 words get no box at all). Its layout analysis
  also silently drops text it judges too large (at ×1.5–2 a whole line can vanish; psm 13 on the same line reads it).
- **Inside our lines Tesseract cuts words about right, but places the boxes loosely.** At IoU ≥ 0.3 instead of 0.5
  its F1 rises 0.828 → 0.940 on the test and 0.660 → 0.810 on the tuning pages; ours rises 0.862 → 0.903 and
  0.662 → 0.793. Tesseract's LSTM boxes are estimated from where it emitted each character, not from the ink.
  (Diagnostic only; the gate stays at IoU 0.5.)
- **Box errors cost far more than they look:** with F1 0.86 the recognizer's page CER is 0.21, against 0.08 with
  correct boxes. A merged or split word is lost entirely, and words shift.
- The two methods fail on different words: 17 words only ours got, 10 only Tesseract-in-lines got, 9 neither.
  Our finder minus Tesseract-in-lines, recall difference 95% bootstrap interval [−0.02, +0.12]: not clearly different.

### Earlier runs (kept for the record)
- `results/step3/run1_val_only/`: settings chosen on the 12 VAL pages only (mostly tiny one-line images). The choice
  did not transfer to full pages (Tesseract 0.620, ours 0.761 on the test). After seeing that, the tuning set
  was widened to all TRAIN + VAL pages: **this change was prompted by test results**, though no choice uses test pages.
- `results/step3/run2_small_grid/`: wider tuning set, smaller grid (no psm 13, no letter-height scaling, no
  Tesseract inside our lines). Tesseract 0.661, ours 0.862.

## 7. Will fine-tuning Tesseract help word boxes? (open decision)

- Fine-tuning changes the LSTM (what characters and spaces it emits), **not** the layout analysis that drops whole
  lines. Whole-page Tesseract would keep its main failure.
- Inside our lines, the cuts are already mostly right; what fails IoU 0.5 is box placement. Fine-tuning on 165
  lines might sharpen that somewhat, but nothing guarantees it, and Tesseract's text (CER 0.31 inside lines) would
  still be far behind the Step 2 recognizer (0.08 on correct boxes), so its text is not needed either.
- Cheaper, more direct levers (no download, no training): snap Tesseract's cuts to the ink (assign each ink group to
  the Tesseract word it overlaps most and box the whole group), and/or combine its cuts with our gap rule, or
  let the recognizer score alternative cuts within a line. These would be chosen on the tuning pages and measured
  once on the test. Note: the snapping idea came from a diagnostic on the test pages.

## 8. Levers 1 and 2 (2026-11-14) — gate FAIL, stopped as agreed

Rules set by the user: tune on TRAIN+VAL only; measure the unseen writer ONCE with the final method (it had been
looked at in the baseline); if still below 0.90, stop. `scripts/step3_levers.py`, `results/step3/levers_*`.
Lines are always our slope-tracking lines.

- **Lever 1** (`tesseract_ft/cuts.py: snap_cuts`): Tesseract reads each line; each ink span goes to the
  Tesseract word it overlaps most, so the cuts come from Tesseract and the boxes from the ink. Tesseract setting
  re-chosen among the 12 best of the baseline: `script/Arabic`, psm 13, grayscale, ×1.5.
- **Lever 2** (`best_cuts`): every group of up to 5 consecutive spans (99% of true words) is read by the Step 2
  recognizer; dynamic programming maximises a·(mean log-prob) + b·[reading in the word list] − c per word, plus
  d·gap/height + e·(Tesseract agrees) per cut. 600+ weight settings searched; chosen a 3.0, b 2.75, c 2.79,
  d 0.0, e 0.67. Word list: 48,357 words (`scripts/build_lexicon.py`: Wikipedia-sd, POS sentences, batch 2 TRAIN+VAL
  texts, TESS_TRAIN texts, legal CSV; sentences overlapping any test text removed).

**Choice on the tuning pages** (word-box F1, in-sample): own gap rule 0.662, Tesseract in lines 0.660, lever 1 0.743,
**lever 2 0.928** (without Tesseract 0.918) → final = lever 2.

**Measured once:**

| | word P | word R | **word F1** | merged / split | recognizer CER on found words (n) | words correct | page CER |
|---|---|---|---|---|---|---|---|
| **unseen writer** (gate) | 0.859 | 0.764 | **0.809** | 16 / 4 | 0.095 (110) | 70.9% | 0.216 |
| TESS_TEST (H, J; seen in training) | 0.714 | 0.625 | 0.667 | 9 / 3 | 0.190 (35) | 57.1% | 0.313 |

For comparison, the baseline own finder measured in section 6 (before this round): unseen writer 0.862,
TESS_TEST 0.775. **Lever 2 is worse on both independent tests, although it won clearly on the tuning pages.**

**Typical handwriting (tuning pages, in-sample: they chose the weights, and the recognizer was fine-tuned on
the TRAIN ones):**

| writer | pages | words | own gap rule F1 | lever 1 F1 | lever 2 F1 | lever 2: CER on found words | words correct |
|---|---|---|---|---|---|---|---|
| A | 50 | 297 | 0.749 | 0.831 | 0.956 | 0.032 | 92.9% |
| H | 33 | 240 | 0.594 | 0.712 | 0.916 | 0.049 | 92.2% |
| J | 16 | 131 | 0.554 | 0.673 | 0.959 | 0.010 | 98.4% |
| neat pen (350–370) | 5 | 338 | 0.889 | 0.889 | 0.918 | 0.022 | 95.4% |
| spaced-letters pencil (305–316) | 6 | 169 | 0.360 | 0.472 | 0.893 | 0.079 | 80.6% |
| VAL pages only (recognizer never trained on them) | 12 | 71 | 0.442 | 0.578 | 0.845 | 0.154 | |

### Why lever 2 failed on the tests
- **It trusts the recognizer, and the recognizer is overconfident on writing it was trained on.** On TRAIN pages a
  correct cut reads back almost perfectly (CER 0.01–0.05), so the search learned to ignore gap size entirely (d = 0)
  and lean on recognizer confidence. On a new writer the readings are less certain, and a merged pair of words
  can score as well as two separate words: 16 merges on the unseen writer, against 9 for the gap rule.
  This bias was foreseeable; the user's rule fixed TRAIN+VAL as the tuning set.
- **Line finding errors feed it.** On 385 the line finder merged the two lines in places; lever 2 then grouped ink
  from both lines into one "word" (F1 0.42 on that page vs 0.74 for the gap rule). See `debug/final/`.
- Lever 1 helped over the plain Tesseract boxes on the tuning pages (0.743 vs 0.660) but was not chosen.

### For Step 5 (the user's decision)
Two candidate box finders, with conflicting evidence:
- **own gap rule** (baseline): better on both independent tests (0.862 / 0.775), no dependence on the recognizer;
- **lever 2**: far better on tuning pages and on VAL pages the recognizer never saw (0.845 vs 0.442; mostly tiny
  one-line images), worse on full multi-line pages.
Choosing between them with the test numbers in hand uses the test; Step 6 (FINAL_TEST, never touched) will give
the honest end-to-end number either way.

## 9. Decision (2026-11-14) — Step 3 accepted with the gate FAIL

- The user accepted Step 3 as it is. `results/gates/step3_tesseract.json` still says **FAIL** (0.809), with the
  decision stored next to it (`user_decision`) and the Step 5 box finder under `extra.step5_box_finder`.
- **Step 5 box finder: our own gap rule** (`configs/step3_tesseract.yaml: box_finder`; slope-tracking lines,
  cut where the gap > max(0.3 × letter height, 0.75 × the line's median gap)). Unseen writer 0.862, TESS_TEST 0.775.
  This was chosen with the test numbers in hand; Step 6 (FINAL_TEST, untouched) gives the honest end-to-end number.
- **Lever 2 stays in the code as an option** (`tesseract_ft/cuts.py`, `scripts/step3_levers.py`). Before using it,
  re-tune its weights on pages from writers the recognizer was **not** trained on (for example, newly transcribed
  pages), since tuning on training writers made it over-trust the recognizer.
- Tesseract itself is not used by the pipeline: neither its boxes nor its text beat the alternatives.
