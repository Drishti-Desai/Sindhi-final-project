"""Unit tests for Step 4 correction pieces (xlmr/correct.py, errors.py, rules.py). No model is loaded.

Run: .venv/bin/python -m pytest tests/test_step4.py -q
"""
from xlmr.correct import Lexicon, build_positions, clean_tokens, decide, evaluate, split_tail, wdist
from xlmr.errors import ErrorModel, align_ops
from xlmr.rules import rules

LEX = Lexicon({"ڪهڙا": 50, "ماڻهو": 80, "کاڌو": 20, "وٺي": 30, "سگهن": 40, "ٿا": 60, "۾": 500, "بنيادي": 10,
               "ضابطا": 5})
W = {"alpha": 1.0, "beta": 0.3, "eta": 2.0, "gamma": 0.0, "tau_oov": 0.0, "tau_known": 99.0}


def test_split_tail_and_weighted_distance():
    assert split_tail("آهن؟") == ("آهن", "؟")
    assert wdist("ٿا", "تا") < 1.0          # look-alike family: cheap
    assert wdist("ٿا", "ما") == 1.0         # unrelated letters: full cost
    assert wdist("ماڻهو", "ماڻهو") == 0.0


def test_oov_word_is_corrected_and_known_words_kept():
    tokens = ["ڪهڙا", "ماڻهو", "کاڌو", "وٺي", "سگهن", "تا؟"]
    out, act = decide(build_positions(tokens, LEX), W, flag_known=False)
    assert out == ["ڪهڙا", "ماڻهو", "کاڌو", "وٺي", "سگهن", "ٿا؟"]
    assert act == ["keep"] * 5 + ["replace"]


def test_merge_and_split_candidates():
    out, act = decide(build_positions(["بنياديضابطا"], LEX), W, flag_known=False)
    assert out == ["بنيادي", "ضابطا"] and act == ["split"]
    out, act = decide(build_positions(["ماڻ", "هو"], LEX), W, flag_known=False)
    assert out == ["ماڻهو"] and act == ["merge", "merged"]


def test_false_correction_counting():
    items = [{"ref": ["ماڻهو", "وٺي"], "input": ["ماڻهو", "رٺي"], "origin": [[0], [1]]}]
    e = evaluate(items, [["ماڻهو", "وٺي"]], [["keep", "replace"]])
    assert e["word_accuracy_before"] == 0.5 and e["word_accuracy"] == 1.0 and e["false_correction_rate"] == 0.0
    e = evaluate(items, [["ماڻهن", "وٺي"]], [["replace", "replace"]])
    assert e["clean_changed"] == 1
    # without an origin map the correct tokens are found by alignment
    assert clean_tokens(["الف", "ب", "ج"], ["الف", "x", "ج"], None) == [True, False, True]


def test_error_model_corrupts_exactly_the_rate():
    em = ErrorModel([("ماڻهو", "ماڻمو"), ("کاڌو", "کاڌو"), ("وٺي", "وي")], merge_rate=0.05, split_rate=0.05)
    assert align_ops("وٺي", "وي") == [("del", "ٺ", "")]
    sents = [["ڪهڙا", "ماڻهو", "کاڌو", "وٺي", "سگهن"]] * 20
    toks, origin = em.corrupt(sents, 0.2, seed=1)
    wrong = sum(len({i for t, o in zip(st, so) if len(o) > 1 or t != sr[o[0]] or so.count(o) > 1 for i in o})
                for st, so, sr in zip(toks, origin, sents))
    assert wrong == 20                      # 20% of 100 words


def test_rules_flag_only_impossible_forms():
    r = rules({"۾", "۽", "ء"})
    assert r["R1 ۾ only as a whole word"]("وچ۾") and not r["R1 ۾ only as a whole word"]("۾")
    assert r["R3 no word starts with ڻ"]("ڻو") and not r["R3 no word starts with ڻ"]("ماڻهو")
    assert r["R7 no letter three times in a row"]("ممم") and not r["R7 no letter three times in a row"]("2000")
    assert r["R10 one-letter words only from the training list"]("ز")
