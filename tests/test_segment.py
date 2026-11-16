"""Unit tests for word segmentation helpers (cnn_transformer/segment.py).

Run: .venv/bin/python -m pytest tests/test_segment.py -q
"""
import numpy as np

from cnn_transformer.segment import (align_page, cluster_lines, cluster_lines_simple, components,
                                     remove_circled_numbers)


def blob(img, x, y, w=14, h=28):
    img[y:y + h, x:x + w] = 0


def page_with_lines(slope=0.0, gap=40):
    """Two text lines of 'letters' (14x28 blocks); the second line drifts by `slope` per pixel."""
    img = np.full((260, 900), 255, np.uint8)
    for k in range(12):
        x = 860 - k * (gap + 14)
        blob(img, x, 40)                                  # line 1, straight
        blob(img, x, int(140 - slope * (860 - x)))        # line 2, rising to the left if slope > 0
    return img


def test_simple_clusters_two_straight_lines():
    comps, _ = components(page_with_lines())
    assert len(cluster_lines_simple(comps)) == 2


def test_tracking_follows_a_sloping_line():
    comps, _ = components(page_with_lines(slope=0.07))    # 2nd line climbs ~43 px across the page
    lines = cluster_lines(comps)
    assert len(lines) == 2
    assert sorted(len(ln) for ln in lines) == [12, 12]


def test_circled_number_and_its_digit_are_removed():
    img = np.hstack([page_with_lines(), np.full((260, 100), 255, np.uint8)])  # free right margin
    yy, xx = np.ogrid[:260, :1000]
    ring = (np.abs(np.hypot(xx - 950, yy - 54) - 24) < 2)  # circle in the margin, clear of the letters
    img[ring] = 0
    img[48:60, 947:953] = 0                                 # a digit inside it
    comps, _ = components(img)
    kept, n = remove_circled_numbers(comps)
    assert n == 1 and len(kept) == len(comps) - 2


def test_align_page_respects_reading_order_and_line_breaks():
    img = np.full((200, 600), 255, np.uint8)
    # line 1: two words (2 letters each) right-to-left; line 2: one word of 3 letters
    for x in (540, 520):
        blob(img, x, 20)
    for x in (400, 380):
        blob(img, x, 20)
    for x in (540, 520, 500):
        blob(img, x, 120)
    comps, _ = components(img)
    lines = cluster_lines_simple(comps)
    groups, line_of = align_page(lines, ["aa", "bb", "ccc"])
    assert line_of == [0, 0, 1]
    xs = [min(c["x"] for c in g) for g in groups]
    assert xs[0] > xs[1]                       # first word is the rightmost on line 1
    assert [len(g) for g in groups] == [2, 2, 3]
