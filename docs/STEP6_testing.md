# Step 6 — final test on FINAL_TEST

**Status (2026-11-14): gate FAIL, accepted by the user (section 6).** FINAL_TEST, whole page, measured once:
**1 − CER −0.650** (CER 1.650 > 1, because the pipeline read far more words than are on the page). The prediction
(CER 0.28–0.40) was **wrong because of writer D's two pages**, where the pipeline read 306 words for 42 true ones.
On writers H and J (5 pages) the CER is **0.354**, inside the predicted range.

## 1. Protocol (agreed with the user, 2026-11-14)

- **Set:** FINAL_TEST, 7 handwritten pages (D2, D6., H3, H5, J1, J4, J5; writers D, H, J; about 150 words).
  The writers were seen in training (H and J through batch 2, D through the old TESS_TRAIN pages); the texts are
  new (their sentences were removed from all training text). So Step 6 measures **seen writers, new text**. The
  Step 5 unseen-writer result stays the reference for new writers.
- **What touched FINAL_TEST before Step 6:** Step 1 preprocessed the 7 images, and its doc comments on their image
  quality (D2 re-photographed, D2's faint pencil, J4's fold band). The texts were read only to remove their sentences
  from training text, plus one word-coverage statistic in Step 2. During Step 5 planning the non-empty lines of the
  text files were counted (23). No recognizer, box finder or corrector has run on a FINAL_TEST page. No FINAL_TEST
  page is in any training, validation or tuning list (checked in the manifests). The command line's image loader
  gives the same pixels as the earlier loader on all 15 TESS images.
- **Method:** `run_pipeline.py` defaults, exactly as on the command line: raw image → Step 1 → lines → reading order →
  recognizer joins → Step 4 correction (Sindhi-trained XLM-R + detector). No `--erase-rules` / `--remove-circled`
  (as the TESS pages were cleaned). Run once by `scripts/step6_final.py`, which refuses to run twice.
- **Metrics:** page text against the reference text (normalised as for TESS_TEST), **whole page**: 1 − CER pooled
  over the 7 pages (the gate), word accuracy (1 − WER), before and after the correction, correct words changed by the
  correction; per page and per writer; a 95% bootstrap interval over pages; lines and words found vs the reference.
- **Reference only, same procedure:** the 3 TESS_TEST pages (H1, J2, J3), whole page. Steps 3–5 scored only the
  verified part of each page, so this shows how much the whole-page scoring changes the number.
- **Gate:** `step6_final`, 1 − CER ≥ 0.90.
- After the run: error analysis only, no changes.

## 2. Prediction, recorded before the run

**CER 0.28–0.40 (1 − CER 0.60–0.72): the gate will FAIL.** Basis: the H/J TESS_TEST pages through the pipeline gave
CER 0.303 (verified part only); whole pages add frames and stray ink; writer D's faint pencil (D2) will likely be
the worst page.

## 3. Results (measured once, 2026-11-14; `results/step6/final_metrics.json`)

| set | pages | words (read / true) | lines (found / true) | CER before → after correction | words correct before → after |
|---|---|---|---|---|---|
| **FINAL_TEST (gate)** | 7 | 396 / 147 | 119 / 23 | 1.709 → **1.650** | −1.605 → −1.510 |
| — writer D | 2 | 306 / 42 | 99 / 6 | 4.950 → 4.724 | −6.17 → −6.17 |
| — writer H | 2 | 15 / 19 | 3 / 3 | 0.187 → **0.173** | 0.368 → 0.737 |
| — writer J | 3 | 75 / 86 | 17 / 14 | 0.373 → 0.388 | 0.186 → 0.267 |
| — writers H + J | 5 | 90 / 105 | 20 / 17 | 0.343 → **0.354** | |
| reference: TESS_TEST, whole page | 3 | 55 / 61 | 12 / 11 | 0.307 → 0.291 | 0.246 → 0.377 |

Per page, CER after correction: D2 6.26, D6. 3.58, H3 0.289, H5 **0.054**, J1 0.345, J4 0.540, J5 0.310.
95% bootstrap interval of the CER over pages: 0.32–3.53 (7 pages; it shows how much two pages decide the result).
The correction changed 1 of 31 correct words (on J5). Gate `step6_final`: **FAIL**.

CER above 1 is possible because inserted words count as errors: on D2 the pipeline produced 167 words for 18.
Word accuracy below 0 has the same cause.

## 4. Why writer D's pages failed (looked at after the measurement; nothing changed)

Page images with the found lines drawn on them: `results/step6/debug/` (git-ignored).

- **D2: faint pencil on a photo of the sheet on a desk.** Step 1 turns the thin pencil strokes into many broken
  fragments; a desk reflection at the top leaves a cloud of speckle; remnants of the ruled lines and the dark
  paper edge on the left survive. The line finder builds **74 overlapping lines** from these fragments (true: 4),
  and every fragment is read as a "word" (167 for 18). This page's problems were already listed in the Step 1 doc
  (faint pencil of lines 1–2, desk reflection).
- **D6.: writing on dotted ruled lines.** The dotted and solid rulings survive Step 1 as rows of dots that touch the
  writing; the line finder finds **25 lines** (true: 2) and the dots are read as words.
- Writer D also writes many letters in their separate (isolated) forms, so words consist of many pieces.
- **No page like these was ever used to choose anything**: the Step 3–5 tuning and test pages were batch 2 (A, H,
  J, numbered writers) and TESS_TEST (H, J); D's pages appeared only as TESS_TRAIN word crops. Step 1's gate was
  measured on synthetic pages, which had no dotted rulings and no pencil this faint. The pipeline has no check
  that rejects ink that is not text (rulings, speckle, edges).
- Whether `--erase-rules` (curved rulings) or other clean-ups would help these pages was **not tried on
  FINAL_TEST**, so that the test stays a test.

Writers H and J behave as the Step 5 reference predicted: H5 is read almost perfectly (CER 0.054), J4 is the worst
of them (0.540; Step 1 noted a paper-fold band on J4). On the H + J pages the correction raises word accuracy but
not CER (0.343 → 0.354), as on the VAL pages.

## 5. The project's results (all gates)

| step | gate | result | status |
|---|---|---|---|
| 1 preprocessing | page F1, dot recall, glyph F1, deskew (synthetic pages) | F1 0.956, dot recall 0.993 | **PASS** |
| 2 letter CNN | letter test accuracy ≥ 0.90 | 0.8225 | FAIL, accepted |
| 2 word recognizer | composed words 1 − CER ≥ 0.90 | 0.866 (unseen writer's real words: 0.912) | FAIL, accepted |
| 3 box finding | unseen writer word-box F1 ≥ 0.90 | 0.809 (gap rule 0.862, used) | FAIL, accepted |
| 4 XLM-R correction | domain test word accuracy ≥ 0.90, changes ≤ 2% | 0.890, 2.07% (unseen writer words 71.5% → 79.2%) | FAIL, accepted |
| 5 pipeline | unseen writer's pages 1 − CER ≥ 0.90 | 0.778 | FAIL, accepted |
| **6 final test** | FINAL_TEST pages 1 − CER ≥ 0.90 | **−0.650** (writers H + J: 0.646) | **FAIL** |

What the system can do today: on clean pen pages by writers it has seen, about 2 in 3 characters right overall and
some pages almost perfect (H5: CER 0.05); on a new writer with clear spacing about 4 in 5 (Step 5). It **cannot**
read faint pencil photos or pages written on dotted rulings: it reads the rulings and the noise as text.

The biggest levers, in order: (1) a text/non-text filter and ruling removal before line finding (D pages);
(2) more transcribed lines from more writers, which every later step needs (better joins, option 2's honest tuning,
option 3's line recognizer); (3) combining gap width with the recognizer score for joins (Step 5 section 5).

## 6. Decision (2026-11-14) — Step 6 accepted with the gate FAIL

The user accepted Step 6 with the gate recorded as FAIL plus this override (`results/gates/step6_final.json`,
`user_decision`); the project closes as it stands. Added afterwards, **as usability measures only** (designed after
seeing FINAL_TEST, so they are not measured improvements):

- README "How to get good results": pen on plain paper, upright photo; pencil and ruled/dotted paper not supported.
- A plausibility warning in `run_pipeline.py` (`pipeline/page.py`, `plausibility`): it fires when found line pieces
  lie on top of each other ≥ 2.5 deep on average (or ≥ 20 pieces ≥ 2.0 deep), or when ≥ 30% of ≥ 30 words read are
  single letters. On our pages it fires on D2, D6. and the TESS_TRAIN page D1.. (which fails the same way: 33 lines,
  259 words) and on none of the other 130 pages (TESS_TRAIN, TESS_TEST, batch 2).
- A local web frontend, `app.py` (Gradio 6.32, 127.0.0.1 only, no share link), using the same pipeline code as
  `run_pipeline.py`; tested once on the clear pen page A1 (TESS_TRAIN): 3 lines, 17 words, about 10 s, no warning.
  Start/stop instructions: README, "Web frontend".
