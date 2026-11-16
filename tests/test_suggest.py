"""Correction modes and saved choices (xlmr/correct.py suggest / apply_choices, pipeline/page.py, pipeline/feedback.py).
No model is loaded.

Run: .venv/bin/python -m pytest tests/test_suggest.py -q
"""
import json
from datetime import datetime

import numpy as np
import pytest

from pipeline.feedback import save_feedback
from pipeline.page import add_suggestion_boxes, apply_page_choices, correct_mode, finish_page
from xlmr.correct import Lexicon, apply_choices, build_positions, decide, suggest

LEX = Lexicon({"ڪهڙا": 50, "ماڻهو": 80, "کاڌو": 20, "وٺي": 30, "سگهن": 40, "ٿا": 60, "تا": 1, "بنيادي": 10,
               "ضابطا": 5})
W = {"alpha": 1.0, "beta": 0.3, "eta": 2.0, "gamma": 0.0, "tau_oov": 0.0, "tau_known": 99.0}
TOKENS = ["ڪهڙا", "ماڻهو", "کاڌو", "وٺي", "سگهن", "ٽا؟"]


def test_suggest_changes_nothing_and_lists_what_auto_would_change():
    P = build_positions(TOKENS, LEX)
    auto, acts = decide(P, W, flag_known=False)
    sg = suggest(TOKENS, P, W, flag_known=False)
    changed = [i for i, a in enumerate(acts) if a not in ("keep", "merged")]
    assert [s["index"] for s in sg] == changed == [5]
    s = sg[0]
    assert s["reading"] == "ٽا؟" and 1 <= len(s["candidates"]) <= 3
    assert s["candidates"][0]["text"] == auto[5]          # the first candidate is what auto-correct picks
    assert all(c["text"] != s["reading"] for c in s["candidates"])
    assert s["candidates"][0]["words"][-1].endswith("؟")  # punctuation put back


def test_apply_choices_keep_replace_merge_split():
    tokens = ["الف", "ب", "ج", "دال"]
    sugg = [{"index": 1, "reading": "ب", "candidates": [
                {"text": "بج", "words": ["بج"], "kind": "merge", "consumes": 2},
                {"text": "پ", "words": ["پ"], "kind": "replace", "consumes": 1}]},
            {"index": 2, "reading": "ج", "candidates": [{"text": "چ", "words": ["چ"], "kind": "replace", "consumes": 1}]},
            {"index": 3, "reading": "دال", "candidates": [{"text": "دا ل", "words": ["دا", "ل"], "kind": "split",
                                                          "consumes": 1}]}]
    assert apply_choices(tokens, sugg, {}) == (tokens, ["keep"] * 4)
    assert apply_choices(tokens, sugg, {1: 2, 2: 1}) == (["الف", "پ", "چ", "دال"], ["keep", "replace", "replace", "keep"])
    # a merge takes the next token; that token's own choice is ignored
    assert apply_choices(tokens, sugg, {1: 1, 2: 1}) == (["الف", "بج", "دال"], ["keep", "merge", "merged", "keep"])
    assert apply_choices(tokens, sugg, {3: 1}) == (["الف", "ب", "ج", "دا", "ل"], ["keep", "keep", "keep", "split"])
    assert apply_choices(tokens, sugg, {1: 9}) == (tokens, ["keep"] * 4)   # an out-of-range pick keeps the word


def page_with_suggestion():
    page = {"lines": [{"box": [0, 0, 100, 20], "words": [{"box": [60, 2, 90, 18], "text": "ڪهڙا"},
                                                         {"box": [10, 2, 50, 18], "text": "ٽا؟"}]}],
            "tokens": ["ڪهڙا", "ٽا؟"], "correct_mode": "suggest", "words_mode": "joins",
            "suggestions": [{"index": 1, "reading": "ٽا؟",
                             "candidates": [{"text": "ٿا؟", "words": ["ٿا؟"], "kind": "replace", "consumes": 1},
                                            {"text": "تا؟", "words": ["تا؟"], "kind": "replace", "consumes": 1}]}]}
    add_suggestion_boxes(page)
    return finish_page(page, list(page["tokens"]), ["keep", "keep"])


def test_page_modes_and_choices():
    assert correct_mode(True) == "auto" and correct_mode(False) == "off" and correct_mode("suggest") == "suggest"
    with pytest.raises(ValueError):
        correct_mode("on")
    page = page_with_suggestion()
    assert page["suggestions"][0]["box"] == [10, 2, 50, 18] and page["text"] == "ڪهڙا ٽا؟"
    apply_page_choices(page, {1: 2})
    assert page["text"] == "ڪهڙا تا؟" and page["actions"] == ["keep", "replace"]
    apply_page_choices(page, {1: 0})
    assert page["text"] == "ڪهڙا ٽا؟"


def test_save_feedback_writes_crops_and_records(tmp_path):
    page = page_with_suggestion()
    binary = np.full((20, 100), 255, np.uint8)
    binary[5:15, 20:40] = 0
    folder = save_feedback(binary, "my page.png", page, {1: 1}, now=datetime(2026, 11, 15, 14, 32, 5), root=tmp_path)
    assert folder.name == "20261115-143205_my_page"
    assert (folder / "word_001.png").exists()
    rec = [json.loads(l) for l in open(folder / "choices.jsonl", encoding="utf-8")]
    assert rec == [{"time": "2026-11-15T14:32:05", "image": "my page.png", "word": 1, "crop": "word_001.png",
                    "box": [10, 2, 50, 18], "recognizer_reading": "ٽا؟", "next_reading": None,
                    "candidates": ["ٿا؟", "تا؟"], "chosen": "ٿا؟", "kept_original": False, "words_mode": "joins"}]
    again = save_feedback(binary, "my page.png", page, {}, now=datetime(2026, 11, 15, 14, 32, 5), root=tmp_path)
    assert again != folder and json.loads(open(again / "choices.jsonl", encoding="utf-8").readline())["kept_original"]
