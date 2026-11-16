"""Unit tests for Step 3 box scoring and our own box finder (tesseract_ft/boxes.py).

Run: .venv/bin/python -m pytest tests/test_step3.py -q
"""
import numpy as np

from tesseract_ft.boxes import iou_matrix, own_layout, prf, resolve_scale, score_boxes, split_merge, tighten


def page(words):
    """White page; each word is a list of (x, y, w, h) ink blocks."""
    img = np.full((200, 600), 255, np.uint8)
    for word in words:
        for x, y, w, h in word:
            img[y:y + h, x:x + w] = 0
    return img


def test_tighten_and_empty_box():
    img = page([[(100, 50, 20, 30)]])
    assert tighten((90, 40, 140, 100), img < 128) == (100, 50, 120, 80)
    assert tighten((300, 40, 340, 100), img < 128) is None


def test_iou():
    m = iou_matrix([(0, 0, 10, 10)], [(0, 0, 10, 10), (5, 0, 15, 10)])
    assert np.allclose(m, [[1.0, 1 / 3]])


def test_match_fp_and_dont_care():
    img = page([[(100, 50, 40, 30)], [(200, 50, 40, 30)], [(400, 50, 40, 30)]])
    gt = [(100, 50, 140, 80), (200, 50, 240, 80)]
    dc = [(400, 50, 440, 80)]                       # e.g. a crossed-out word
    pred = [(98, 45, 142, 85),                      # matches word 1 after tightening
            (195, 45, 245, 85),                     # matches word 2
            (398, 45, 442, 85),                     # on don't-care ink: ignored, not a false positive
            (150, 45, 190, 85)]                     # no ink: dropped
    r = score_boxes(pred, gt, dc, img)
    assert (r["tp"], r["fp"], r["fn"], r["ignored"]) == (2, 0, 0, 1)


def test_merged_box_is_a_false_positive():
    img = page([[(100, 50, 40, 30)], [(150, 50, 40, 30)]])
    gt = [(100, 50, 140, 80), (150, 50, 190, 80)]
    r = score_boxes([(100, 50, 190, 80)], gt, [], img)
    assert (r["tp"], r["fp"], r["fn"]) == (0, 1, 2)
    assert split_merge(r["pred"], gt)["pred_boxes_merging_2plus"] == 1
    assert prf(r["tp"], r["fp"], r["fn"])["f1"] == 0.0


def test_own_layout_cuts_at_wide_gaps_only():
    # one line, three words; letters inside a word 6 px apart, words 40 px apart
    words = [[(x0 + k * 26, 80, 20, 30) for k in range(3)] for x0 in (40, 200, 360)]
    lines = own_layout(page(words), "simple", gap_h=0.5, gap_med=0.0)
    assert len(lines) == 1 and len(lines[0]) == 3
    assert lines[0][0][0] > lines[0][-1][0]          # reading order: right to left


def test_resolve_scale():
    img = page([[(40 + k * 30, 80, 20, 20)] for k in range(5)])
    assert resolve_scale(img, 1.5) == 1.5
    assert abs(resolve_scale(img, "h40") - 2.0) < 1e-6


def test_snap_cuts_follow_tesseract_words_not_its_box_edges():
    from tesseract_ft.boxes import line_structures
    from tesseract_ft.cuts import snap_cuts
    # 4 spans (right to left: x=400, 340, 200, 140); Tesseract says 2 words but its boxes are loose and offset
    img = page([[(400, 80, 20, 30)], [(340, 80, 20, 30)], [(200, 80, 20, 30)], [(140, 80, 20, 30)]])
    (line,), _ = line_structures(img)
    assert snap_cuts(line, [(330, 70, 410, 120), (150, 70, 260, 120)]) == {1}
    assert snap_cuts(line, []) is None


def test_best_cuts_prefers_known_words_and_wide_gaps():
    from tesseract_ft.boxes import line_structures
    from tesseract_ft.cuts import best_cuts, candidate_groups
    img = page([[(400, 80, 20, 30)], [(372, 80, 20, 30)], [(200, 80, 20, 30)]])   # gaps 8 px and 152 px
    (line,), _ = line_structures(img)
    groups = candidate_groups(line, 3)
    known = {(0, 1), (2, 2)}                                                      # the recognizer reads real words here
    feats = [(-0.1 if g in known else -2.0, 1.0 if g in known else 0.0) for g in groups]
    assert best_cuts(line, groups, feats, [0, 0], a=1, b=1, c=0.5, d=0.0, e=0) == {1}
    # with only the gap term, the same answer comes from the wide gap
    flat = [(0.0, 0.0)] * len(groups)
    assert best_cuts(line, groups, flat, [0, 0], a=0, b=0, c=1.0, d=0.5, e=0) == {1}
