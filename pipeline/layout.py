"""Lines and reading order (Step 5).

Lines come from our tracked line finder (tesseract_ft.boxes.line_structures): merged ink spans in reading order,
right to left. A sloped handwritten line is sometimes found as two pieces side by side; sorting pieces by their
height on the page can then put the right-hand piece (read first) after the left one.

order_lines: pieces whose vertical ranges overlap by at least `min_v` of the lower piece's height, and whose
horizontal ranges overlap by at most `max_x` of the narrower piece's width, are the same row. Rows go top to bottom
(by mean centre), pieces in a row right to left (by right edge).
"""
from __future__ import annotations

import numpy as np

from tesseract_ft.boxes import group_box

ORDER = {"min_v": 0.6, "max_x": 0.75}   # chosen in scripts/step5_order.py on TRAIN + VAL (boxes only)


def line_bbox(ln: dict) -> tuple:
    return group_box(ln, 0, len(ln["spans"]) - 1)


def order_lines(boxes: list[tuple], min_v: float = ORDER["min_v"], max_x: float = ORDER["max_x"]) -> list[int]:
    """Reading order of line pieces given their boxes (x0, y0, x1, y1). Returns piece indices."""
    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = boxes[i], boxes[j]
            v = min(a[3], b[3]) - max(a[1], b[1])
            x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
            if v >= min_v * min(a[3] - a[1], b[3] - b[1]) and x <= max_x * min(a[2] - a[0], b[2] - b[0]):
                parent[find(i)] = find(j)
    rows: dict[int, list[int]] = {}
    for i in range(n):
        rows.setdefault(find(i), []).append(i)
    ordered = sorted(rows.values(), key=lambda r: np.mean([(boxes[i][1] + boxes[i][3]) / 2 for i in r]))
    return [i for r in ordered for i in sorted(r, key=lambda i: -boxes[i][2])]


def order_by_centre(boxes: list[tuple]) -> list[int]:
    """The Step 3/4 order: pieces by mean centre height only."""
    return sorted(range(len(boxes)), key=lambda i: (boxes[i][1] + boxes[i][3]) / 2)
