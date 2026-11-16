# Step 5 — `run_pipeline.py`: from a page image to corrected text

**Status (2026-11-14): gate FAIL, accepted by the user (section 6).** Unseen writer's pages through `run_pipeline.py`:
**1 − CER 0.778** (CER 0.222) < 0.90. The prediction recorded before the run (CER 0.12–0.15) was **wrong, in the bad
direction**: on this writer the pipeline is worse than the Step 4 gap-rule reading (CER 0.193), although it is much
better on the VAL pages (0.462 → 0.246) and a little better on H/J (0.321 → 0.303). Section 4 explains why.

## 1. Plan (agreed with the user, 2026-11-14)

Step 4 showed that whole pages read through the Step 3 gap-rule boxes are much worse than word crops (VAL pages:
CER 0.44 vs 0.12), and that **word splitting** is the main cause (docs/STEP4_xlmroberta.md section 8). Step 5:

1. **Reading order**: join line pieces that continue one sloped line before sorting (lines top to bottom, words
   right to left). Chosen by the share of word pairs read out of order on the TRAIN + VAL pages (boxes only, no
   recognizer).
2. **Joins (option 1)**: every line is cut into single ink spans; the Step 2 recognizer reads every run of up to 6
   neighbouring spans (only their own ink), and dynamic programming keeps the grouping with the best total
   recognizer log-probability. **No tuned settings.**
3. **`run_pipeline.py`**: Step 1 preprocessing → lines → reading order → joins → recognizer → Step 4 correction
   (Sindhi-trained XLM-R + detector) → text and JSON.
4. Every choice on the **VAL pages only**; then **one** measurement on the unseen writer's pages (the gate) and the
   H/J TESS_TEST pages (reported). **FINAL_TEST is not read** (Step 6).

**Gate** (user, 2026-11-14): 1 − CER ≥ 0.90 on the unseen writer's 7 pages (the verified part of each page, 144
words) through `run_pipeline.py`, measured once.

### Planning probe (VAL only, nothing trained or tuned; `scripts/step5_probe_joins.py`)

| reading of the VAL pages (71 words) | CER, our order | CER, true order |
|---|---|---|
| gap-rule pieces as now (Step 4) | 0.446 | 0.408 |
| join pieces, word-list rule | 0.376 | 0.338 |
| join pieces, recognizer rule | 0.295 | 0.249 |
| **join spans, recognizer rule** (chosen) | **0.284** | 0.241 |
| best possible joins of pieces (oracle) | 0.232 | 0.192 |
| best possible joins of spans (oracle) | 0.195 | 0.154 |
| true word crops (floor) | – | 0.122 |

### Prediction, recorded before the measurement

Unseen writer's pages through the pipeline: **CER 0.12–0.15 (1 − CER 0.85–0.88), so the gate will most likely
FAIL.** Today (gap-rule boxes + Step 4 correction) the same pages give CER 0.193.

## 2. Options not taken (future levers)

- **Option 2: re-tune lever 2** (Step 3; weights over recognizer score, word list, gap size). It overfit in Step 3
  because the recognizer had memorised the TRAIN writers. The only honest way to tune it with our data: fine-tune the
  recognizer 4 times, each time leaving one writer group out (A, H, J, numbered pages), so every TRAIN + VAL page is
  scored by a recognizer that never saw its writer; tune at most 2 weights by leave-one-writer-out. About 1 day of
  work + ~8 h of training; expected gain over option 1 0.00–0.05 on VAL (ceiling: oracle 0.195), possibly none.
- **Option 3: a line-level recognizer** (no word cutting; CNN encoder + CTC, ~8M parameters, fits 8 GB). Data now:
  about 290 real lines (batch-2 TRAIN 234, VAL 23, old TESS_TRAIN 31), unlimited printed lines, composed lines.
  About 1.5–2 days + 8–12 h training; expected VAL CER 0.25–0.45, because ~290 real lines are too few to bridge the
  printed-to-handwritten gap (the word recognizer needed 1,104 real crops). **Worth trying once more real lines
  are transcribed** (for example the Mendeley POS images, data/raw/pos_handwritten).

## 3. What was built (choices on TRAIN + VAL only)

- `pipeline/layout.py`: **reading order**. Line pieces whose heights overlap by ≥ 60% of the lower piece and whose
  widths overlap by ≤ 75% of the narrower piece form one row; rows top to bottom, pieces right to left. Chosen in
  `scripts/step5_order.py` on the 97 TRAIN + VAL pages from boxes only: word pairs read out of order 1.48% (Step 3/4
  order) → **1.07%** (`results/step5/order_choice.json`).
- `pipeline/page.py`: **joins** (span-level recognizer rule, no settings) and the Step 4 corrector (Sindhi-trained
  XLM-R + detector, validation weights; the two models are loaded one at a time).
- `run_pipeline.py IMAGE ... [--out DIR] [--no-correct] [--erase-rules] [--remove-circled]`: Step 1 preprocessing
  → the above → `<name>.txt` (one text line per found line) and `<name>.json` (boxes, readings, corrections).
- `scripts/step5_eval.py`: the evaluation goes through `run_pipeline.run`, the function the command line uses, on
  the same cleaned pages as Steps 3–4, restricted to the verified part of each page.

### VAL pages (71 words; `results/step5/val_metrics.json`)

| reading | CER before → after correction | words correct (1 − WER) before → after | correct words changed |
|---|---|---|---|
| Step 4: gap-rule boxes | 0.441 → 0.462 | −0.254 → −0.141 | 0 of 17 |
| joins, old reading order | 0.284 → 0.278 | 0.282 → 0.408 | 0 of 22 |
| **joins + new reading order (the pipeline)** | **0.240 → 0.246** | 0.296 → 0.451 | 0 of 23 |

The new order reaches the "true order" bound of the probe (0.241). The correction fixes many words (ڪائونسل,
قانون ٺاهيا ويندا, استعمال) but also cuts a few merged tokens short (وينداهن → ويندا), so CER moves slightly the wrong way
(+0.006, within the noise of 71 words) while word accuracy rises.

## 4. Test: the single measurement (2026-11-14; `results/step5/test_metrics.json`)

| set | reading | CER before → after correction | words correct before → after | correct words changed |
|---|---|---|---|---|
| **unseen writer, 7 pages (144 words) — gate** | Step 4 gap rule | 0.208 → 0.193 | 0.424 → 0.514 | 1 of 74 |
| | **Step 5 pipeline** | 0.237 → **0.222** | 0.347 → 0.486 | 0 of 50 |
| H/J TESS_TEST, 3 pages (56 words) | Step 4 gap rule | 0.321 → 0.321 | 0.071 → 0.196 | 0 of 21 |
| | Step 5 pipeline | 0.314 → 0.303 | 0.232 → 0.375 | 0 of 14 |

Gate `step5_pipeline`: 1 − CER **0.778 < 0.90 → FAIL**. Predicted before the run: CER 0.12–0.15, likely FAIL.

### Why the prediction was wrong

- **The join rule joins too often, for every writer.** It reads 15% fewer words than there are: VAL 60 of 71,
  unseen writer 121 of 144, H/J 51 of 56 (on VAL it made 87 joins where 73 were right). The recognizer, trained on
  single words, tends to prefer one long reading over two short ones, and the rule ignores how wide the gap is.
- **On the VAL writers that cost less than it gained**, because the gap rule split their cramped writing very badly
  (65 of 101 pieces were parts of words). **The unseen writer leaves clear gaps**, so the gap rule was already good
  there (Step 3 word-box F1 0.862), and the extra joins mostly glue true words together: ووٽنگجو, صيحڪان,
  جوطرقيڪار, لڊبڪس.
- **Reading order is still wrong on one page** (385: a line piece is read after the next line), and some words on
  385 are missing.
- My estimate scaled the VAL gain to the unseen writer's pages; it did not allow for the gap rule already working
  well for this writer. The VAL pages (4 writer groups, cramped writing) did not represent this writer's spacing.

Nothing was changed after the measurement.

## 5. Future levers (not tried)

- A gap safeguard: never join across a gap wider than k × letter height (one setting). VAL alone cannot choose k
  for a writer with wider spacing; it would need pages from more writers.
- **Combining gap width with the recognizer score** when deciding joins (for example: join score = recognizer
  log-probability gain − w × gap / letter height). Needs tuning on writers outside training, which we do not have
  labelled apart from the purple-pen test writer.
- Option 2 (lever 2 with out-of-fold recognizers, ≤ 2 weights) — section 2.
- Option 3 (line recognizer) once more real lines are transcribed — section 2.

## 6. Decision (2026-11-14) — Step 5 accepted with the gate FAIL

The user chose option 3: Step 5 is accepted with the gate recorded as FAIL plus this override
(`results/gates/step5_pipeline.json`, `user_decision`). **Joins stay the default**, as chosen on VAL.
`run_pipeline.py --words gaps` reads each line with the Step 3 gap rule instead (no joins), for writers who leave
clear spaces between words; on the VAL pages it gives CER 0.405 → 0.432 after correction (joins: 0.240 → 0.246),
on the unseen writer the gap-rule reading was the better one (section 4). **Step 6 is measured with the default
(joins) only.**
