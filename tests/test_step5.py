"""Unit tests for the Step 5 pipeline pieces that need no model (pipeline/layout.py, pipeline/page.py).

Run: .venv/bin/python -m pytest tests/test_step5.py -q
"""
from pipeline.layout import order_by_centre, order_lines
from pipeline.page import best_grouping, corrected_lines


def test_sloped_line_pieces_are_read_right_to_left():
    # page H111: the first text line was found as two pieces; the right-hand piece sits lower on the page
    boxes = [(42, 75, 932, 163), (812, 113, 1060, 194), (578, 241, 1060, 310)]
    assert order_by_centre(boxes) == [0, 1, 2]          # the old order reads the left piece first
    assert order_lines(boxes) == [1, 0, 2]


def test_lines_on_top_of_each_other_are_not_joined():
    boxes = [(278, 1237, 616, 1354), (194, 1233, 2315, 1447)]   # a small piece inside a long line (page 313)
    assert len(set(order_lines(boxes))) == 2
    assert order_lines([(0, 0, 100, 40), (0, 50, 100, 90)]) == [0, 1]


def test_best_grouping_joins_when_the_joined_reading_scores_better():
    reads = {(0, 0): ("ا", -2.0), (1, 1): ("ب", -2.0), (0, 1): ("اب", -0.5)}
    assert best_grouping(2, reads) == [(0, 1)]
    reads[(0, 1)] = ("اب", -3.0)
    assert best_grouping(2, reads) == [(0, 0), (1, 1)]


def test_corrected_tokens_go_back_to_their_lines():
    lines = [{"words": [{}, {}]}, {"words": [{}]}]
    assert corrected_lines(lines, ["a", "b", "c"], ["keep", "replace", "keep"]) == [["a", "b"], ["c"]]
    assert corrected_lines(lines, ["a", "x", "y", "c"], ["keep", "split", "keep"]) == [["a", "x", "y"], ["c"]]
    assert corrected_lines(lines, ["a", "bc"], ["keep", "merge", "merged"]) == [["a", "bc"], []]


def test_words_gaps_cuts_only_at_wide_gaps():
    from pipeline.page import line_units
    ln = {"spans": [None] * 4, "gaps": [5, 40, 6], "h": 30.0}   # gap rule: cut where gap > max(0.3 h, 0.75 median)
    assert line_units(ln, "joins") == [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert line_units(ln, "gaps") == [(0, 1), (2, 3)]                 # threshold max(9, 4.5): only the 40 gap


def test_plausibility_warning():
    from pipeline.page import line_stacking, plausibility
    normal = [(0, 100 * i, 1000, 100 * i + 80) for i in range(5)]               # 5 lines, one under the other
    assert line_stacking(normal) == 1.0 and plausibility(normal, ["کتاب"] * 30) == []
    stacked = [(0, 0, 1000, 100)] * 4                                             # 4 found lines on top of each other
    assert plausibility(stacked, ["کتاب"]) != []
    assert plausibility(normal, ["ه"] * 20 + ["کتاب"] * 20) != []               # half the words are single letters
