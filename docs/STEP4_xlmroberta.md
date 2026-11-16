# Step 4 — XLM-RoBERTa: correcting the recognizer's words

**Status (2026-11-14): gate FAIL, accepted by the user (section 10).** On the held-out legal + POS test text with
20% of words corrupted, the final variant (Sindhi-trained XLM-R plus a wrong-word detector, chosen on validation)
reaches word accuracy **0.890** (< 0.90) and changes **2.07%** of correct words (> 2%). On **real handwriting** (the
main result) it helps: the unseen writer goes from **71.5% → 79.2%** words correct (CER 0.071 → 0.051), with **3 of
103** correct words wrongly changed; writers H and J go from 42.9% → 60.7%.

## 1. What Step 4 is for

The Step 2 recognizer reads single words; its errors are mostly look-alike letters (dots, small marks) and missing
letters. Step 4 reads the recognizer's word sequence and fixes words using a Sindhi word list and XLM-RoBERTa's
judgement of what fits the sentence. It never sees the image.

## 2. Data

| source | train | val | test | notes |
|---|---|---|---|---|
| legal CSV (`~/Downloads/Sindhi_Legal_Dataset.csv`, read only) | 1,722 sentences | 215 | 169 | rows sharing a question/answer kept together; rows overlapping batch-2 VAL pages forced to val, TRAIN pages to train |
| POS sentences (cleaned, 518) | 414 | 52 | 52 | `data/preprocessed/pos_text` |
| Wikipedia (public corpus from earlier) | 243,553 | 1,001 | 1,037 | split by article; val/test sentences of 4–40 words |

Sentences from the test pages, FINAL_TEST and the unseen writer are removed everywhere (`tesseract_ft/texts.py`,
4-gram overlap or containment). Word list: 47,422 words from the TRAIN split only (Wikipedia words need 3+
occurrences).

**Error model** (`xlmr/errors.py`): the fine-tuned recognizer's readings of validation words it never trained on
(Step 2 composed_val, 600 words; the 71 batch-2 VAL crops), smoothed with a table of look-alike letters; word
merges (7.6%) and splits (21.5%) from the Step 3 gap rule on the tuning pages. Top confusions: ن→م, و→ڻ, ه→و;
ا is the letter most often dropped.

**Frozen synthetic sets**: exactly 20% of words made wrong with the error model. Domain = legal + POS (val 3,090
words, test 2,419), Wikipedia (val 16,497, test 17,040). **Real sets**: the recognizer's readings of verified crops
(VAL: 71 words; unseen writer: 144; H/J TESS_TEST: 56), and the same pages through the Step 5 gap-rule boxes.

## 3. Models (`checkpoints/`, git-ignored)

- `xlmr_base`: xlm-roberta-base copied from the user's Hugging Face cache (SHA-256 checked against the published
  value). Its vocabulary has every Sindhi letter.
- `step4_xlmr_mlm/best`: further masked-LM training on the Step 4 TRAIN text (legal + POS oversampled ×20, whole-word
  masking 15%, 1,300 steps). Validation loss 3.60 → 3.11 (perplexity 36.8 → 22.5). To fit the 8 GB M2: embeddings
  frozen and held in bfloat16 during training (exact fp32 weights restored before saving), lower 6 layers frozen,
  output layer applied at masked positions only, gradient checkpointing.
- `step4_xlmr_detector/best`: "is this word wrong?" token classifier on top of the Sindhi model, trained on 30,000
  corrupted TRAIN sentences (legal + POS ×3, Wikipedia). Validation loss 0.135. Same memory measures.

## 4. How a word is corrected (`xlmr/correct.py`, `xlmr/lm.py`)

1. Candidates for each word: word-list entries within a weighted edit distance (look-alike letters and the
   recognizer's frequent confusions cost less), plus splits into two words and merges with the next word.
2. XLM-R scores each candidate in context: the word is replaced by k `<mask>` tokens and the candidate's pieces'
   log-probabilities are summed.
3. score = −α·edit cost + β·log word frequency − η·unknown word + γ·XLM-R score. An unknown word is replaced when
   the best candidate beats keeping it by `tau_oov`; a known word only by `tau_known` (and, with the detector, only
   if the detector flags it).

Weights for each variant: random search (300) + coordinate refinement on the synthetic **domain val** set, maximising
word accuracy with at most 2% of correct words changed.

## 5. Sindhi hard rules (flags only; not kept)

Each rule was checked on the 4.0M correct training words; a rule that correct words break more than 1 in 2,000 times
was dropped. Rules may only flag a word for correction, never change it.

| rule | correct words breaking it | used |
|---|---|---|
| R1 ۾ only as a whole word | 0.026% | yes |
| R2 ۽ only as a whole word | 0.020% | yes |
| R3–R5 no word starts with ڻ / ڙ / ڱ | ≤ 0.001% | yes |
| R6 no longer word starts with ء | 0.0002% | yes |
| R7 no letter three times in a row | 0.002% | yes |
| R8 no digits mixed with letters | dropped (years like 1947ع) | no |
| R9 no punctuation between letters | 0.20% (ق.م, آهي.هن) | no |
| R10 one-letter words only from the allowed list | 0.007% | yes |

Allowed one-letter words (seen 20+ times): ء آ ا ب ت ج د ر س ص ع ف ق م ن ه و ي ٻ پ ڪ گ ۔ ۽ ۾. Rules about ھ are
impossible because the normalisation maps ھ to ه. Known flaws: curly quotes and brackets are not stripped before the
check (`۾“`, `(۽`), and ٬ and ۔ count as letters. **On validation the rules changed nothing on domain text, slightly
raised wrong changes on Wikipedia, and flagged 0 of the 71 real crop words, so they are not used.**

## 6. Validation (chosen here; `results/step4/val_comparison.json`)

Word accuracy before → after, with the share of correct words that were changed.

| variant | domain val | Wikipedia val | real crops (71) |
|---|---|---|---|
| no correction | 0.777 | 0.776 | 0.549 |
| word list only | 0.852 · 1.3% | 0.824 · 3.4% | 0.606 · 0 |
| XLM-R as downloaded | 0.884 · 1.1% | 0.856 · 2.8% | 0.718 · 0 |
| XLM-R Sindhi-trained | 0.893 · 1.4% | 0.865 · 3.0% | 0.704 · 0 |
| **Sindhi-trained + detector** (final) | **0.903 · 1.9%** | 0.864 · 4.1% | 0.718 · 0 |
| + rules | 0.903 · 1.9% | 0.864 · 4.3% | 0.718 · 0 |

The real VAL crops also fed the error model, so their numbers are somewhat optimistic.

## 7. Test (measured once, 2026-11-14; `results/step4/test_metrics.json`)

### Main result: real handwriting

| set | words | before | after (final) | CER before → after | correct words wrongly changed |
|---|---|---|---|---|---|
| **unseen writer, crops** | 144 | 0.715 | **0.792** | 0.071 → 0.051 | **3 of 103** |
| unseen writer, gap-rule pages | 144 | 0.424 | 0.514 | 0.208 → 0.193 | 1 of 74 |
| H/J (TESS_TEST), crops | 56 | 0.429 | 0.607 | 0.185 → 0.178 | 1 of 24 |
| — writer H | 9 | 0.889 | 0.889 | 0.024 → 0.073 | 0 of 8 |
| — writer J | 47 | 0.340 | 0.553 | 0.211 → 0.195 | 1 of 16 |
| H/J, gap-rule pages | 56 | 0.071 | 0.196 | 0.321 → 0.321 | 0 of 21 |

"Before" equals the Step 2 reference exactly (unseen writer 71.5%, H/J 42.9%). CER here is over the joined page
text, so it differs from Step 2's per-word mean. Word accuracy is 1 − WER (an inserted extra word counts as an error).

On the unseen writer's crops the corrector changed 27 words: 15 became right, 3 were right and became wrong
(لڊ→لڏ twice, اڇي→جي), 9 stayed wrong (4 of those now have the right word and differ only in punctuation the
recognizer misread, e.g. ء for ،). Every change is listed in `results/step4/test_real_changes.csv`.

### Synthetic text (the gate)

| variant | domain test: accuracy · changed | Wikipedia test |
|---|---|---|
| no correction | 0.775 | 0.772 |
| word list only | 0.843 · 1.3% | 0.808 · 3.7% |
| **final (Sindhi-trained + detector)** | **0.890 · 2.07%** → gate FAIL | 0.858 · 4.3% |
| reference only: XLM-R as downloaded | 0.874 · 1.6% | 0.851 · 3.0% |

The XLM-R as downloaded is reported for reference only (it matched the final variant on the real VAL crops); it was
not re-selected. On the test it is close behind: unseen crops 0.785 (1 correct word changed), H/J crops 0.554.

### Known limitation: non-legal text

The weights were tuned on legal + POS text. On Wikipedia they change too many correct words (validation 4.1%,
test 4.3%, limit 2%). Use on other kinds of text needs re-tuning; a stricter setting (at most 1% of correct words
changed, chosen on validation) is a future option, at some cost in accuracy.

## 8. Why the gap-rule pages read badly (for Step 5; `scripts/step4_page_diagnosis.py`)

On the VAL pages (71 verified words) the CER was measured four ways, each removing one source of error:

| reading | CER |
|---|---|
| a. saved word crops (only the word's ink), true order | 0.122 |
| b. true word boxes cut from the page, true order | 0.122 |
| c. gap-rule boxes, sorted into the true order | 0.400 |
| d. gap-rule boxes in our reading order (the pipeline) | 0.441 |

- **Word splitting is the main cause (0.12 → 0.40).** Of the 101 gap-rule boxes on these pages only 25 hold one
  whole word; 65 hold part of a word (TRAIN + VAL: 565 of 1,426). Sindhi words contain letters that do not join the
  next one, so a gap inside a word often looks like a gap between words.
- **What looked like duplicated words are word pieces.** The recognizer was trained on whole words, so a piece is
  often read as a whole-looking word: سائيٽ cut 33% / 67% is read "سا" + "سائيٿ"; an 11-pixel mark is read "اهان".
  The corrector cannot undo this, because the pieces no longer look like a split word.
- **Merges** (one box over two words of the same line): 10 of 101 (TRAIN + VAL: 113 of 1,426).
- **Reading order** costs 0.04 overall (1.6% of word pairs out of order) but is severe on single pages: H111 goes
  from 0.57 to 0.88 when a sloped line is found as two pieces and they are sorted wrongly.
- **Not the cause:** other ink inside the box rectangle (a = b), boxes mixing two text lines (1 of 101), and found
  lines holding two true lines (4 of 202 on TRAIN + VAL).

The unseen writer leaves clearer gaps, so their pages fare better (test CER 0.208 vs 0.071 on crops), but the
cause is the same.

## 9. Commands

```bash
.venv/bin/python scripts/step4_prepare.py          # splits, word list, error model, frozen sets, real val readings
.venv/bin/python scripts/step4_train_mlm.py        # ~1.5 h on the M2
.venv/bin/python scripts/step4_train_detector.py   # ~16 min
.venv/bin/python scripts/step4_compare.py          # validation only: variants, weights, rules
.venv/bin/python scripts/step4_test.py             # test, once; writes the gate
.venv/bin/python scripts/step4_page_diagnosis.py   # section 8
.venv/bin/python -m pytest tests/test_step4.py -q
```

## 10. Decision (2026-11-14) — Step 4 accepted with the gate FAIL

The user accepted Step 4 with the gate recorded as FAIL plus this override (`results/gates/step4_xlmroberta.json`,
`user_decision`), as for Steps 2 and 3. **The pipeline uses the final variant: Sindhi-trained XLM-R + detector**
(`checkpoints/step4_xlmr_mlm/best`, `checkpoints/step4_xlmr_detector/best`, weights in
`results/step4/val_comparison.json`). The Wikipedia limitation (section 7) stands.
