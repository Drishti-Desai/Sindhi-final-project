"""Word/line boxes from Tesseract and from our own text-free finder, and how they are scored.

Scoring (the Step 3 gate uses the word numbers):
  - every predicted box is first TIGHTENED to the kept ink inside it, and dropped if it holds no ink
    (the pipeline crops ink the same way; ground-truth boxes are tight around their ink too);
  - one-to-one matching at IoU >= 0.5 (Hungarian assignment on IoU);
  - an unmatched predicted box counts as a false positive unless most of its ink is "don't care" ink
    (ink of a flagged/excluded crop, or ink no ground-truth word covers: crossed-out words, stray marks);
  - recall = matched / ground-truth boxes; precision = matched / (matched + false positives); F1.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from cnn_transformer.align import line_spans
from cnn_transformer.segment import cluster_lines, cluster_lines_simple, components

IOU = 0.5


# ----------------------------------------------------------------------------- Tesseract

def resolve_scale(binary: np.ndarray, scale) -> float:
    """A number is used as is. "h30" means: scale so the median letter-body height becomes 30 px
    (Tesseract silently drops text it finds too big or too small)."""
    if not isinstance(scale, str):
        return float(scale)
    comps, _ = components(binary)
    if not comps:
        return 1.0
    a = np.array([c["a"] for c in comps])
    h = float(np.median([c["h"] for c in comps if c["a"] >= 0.25 * np.median(a[a >= np.median(a)])]))
    return float(np.clip(float(scale[1:]) / max(h, 1.0), 0.5, 4.0))


def run_tesseract(image: np.ndarray, lang: str, psm: int, scale: float = 1.0) -> list[dict]:
    """Word boxes (in the input frame) with Tesseract's block/paragraph/line ids, its text and confidence."""
    img = image if scale == 1.0 else cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "page.png"
        cv2.imwrite(str(src), img)
        out = subprocess.run(["tesseract", str(src), "stdout", "-l", lang, "--psm", str(psm), "tsv"],
                             capture_output=True, text=True, env={**os.environ, "OMP_THREAD_LIMIT": "1"})
    words = []
    for row in out.stdout.splitlines()[1:]:
        f = row.split("\t")
        if len(f) < 12 or f[0] != "5":
            continue
        x, y, w, h = (int(v) / scale for v in f[6:10])
        words.append({"block": int(f[2]), "par": int(f[3]), "line": int(f[4]), "text": f[11],
                      "conf": float(f[10]), "box": (int(x), int(y), int(round(x + w)), int(round(y + h)))})
    return words


def tesseract_layout(words: list[dict]) -> tuple[list[list[tuple]], list[list[str]]]:
    """Group Tesseract's words into its lines: (word boxes per line, word texts per line), each line
    right-to-left, lines in Tesseract's own order."""
    lines: dict[tuple, list[dict]] = {}
    for w in words:
        lines.setdefault((w["block"], w["par"], w["line"]), []).append(w)
    ordered = [sorted(ws, key=lambda w: -w["box"][2]) for ws in lines.values()]
    return [[w["box"] for w in ws] for ws in ordered], [[w["text"] for w in ws] for ws in ordered]


# ----------------------------------------------------------------------------- our own finder

def line_structures(binary: np.ndarray, finder: str = "simple"):
    """Lines found by one of our two line finders, each as: merged ink spans in reading order (right to left),
    the small components (dots, diacritics), the letter-body height, the gaps between consecutive spans and the
    component ids of the whole line. Returns (lines, label image)."""
    comps, labels = components(binary)
    if not comps:
        return [], labels
    lines = cluster_lines_simple(comps) if finder == "simple" else cluster_lines(comps)
    out = []
    for spans, smalls in line_spans(lines):
        if not spans:
            continue
        out.append({"spans": spans, "smalls": smalls,
                    "h": float(np.median([c["h"] for s in spans for c in s[2]])),   # letter-body height on this line
                    "gaps": [spans[i][0] - spans[i + 1][1] for i in range(len(spans) - 1)],
                    "ids": [c["i"] for s in spans for c in s[2]] + [c["i"] for c in smalls]})
    return out, labels


def group_comps(line: dict, i0: int, i1: int) -> list[dict]:
    """Components of spans i0..i1 (reading order) plus the dots/diacritics within their x-range."""
    ws = line["spans"][i0:i1 + 1]
    x0, x1 = min(s[0] for s in ws), max(s[1] for s in ws)
    return [c for s in ws for c in s[2]] + [c for c in line["smalls"] if x0 <= c["x"] + c["w"] / 2 <= x1]


def group_box(line: dict, i0: int, i1: int) -> tuple:
    cs = group_comps(line, i0, i1)
    return (min(c["x"] for c in cs), min(c["y"] for c in cs),
            max(c["x"] + c["w"] for c in cs), max(c["y"] + c["h"] for c in cs))


def groups_from_cuts(n: int, cuts: set[int]) -> list[tuple[int, int]]:
    """Spans 0..n-1 cut after every span index in `cuts` -> (first, last) span of each word."""
    out, start = [], 0
    for k in range(n - 1):
        if k in cuts:
            out.append((start, k))
            start = k + 1
    return out + [(start, n - 1)]


def own_layout(binary: np.ndarray, finder: str = "simple", gap_h: float = 0.5, gap_med: float = 0.0,
               with_comps: bool = False):
    """Text-free lines and words. Lines: one of our two line finders. Words: inside a line, merged ink spans
    in reading order; a cut is made where the gap between two spans is wider than
    max(gap_h x median letter-body height of the line, gap_med x the line's median gap).
    Dots and diacritics join the word whose x-range they fall in. Returns word boxes per line
    (with_comps=True: also the component ids of each line and the label image)."""
    lines, labels = line_structures(binary, finder)
    out = []
    for ln in lines:
        gaps = ln["gaps"]
        thr = max(gap_h * ln["h"], gap_med * float(np.median(gaps)) if gaps else 0.0)
        cuts = {k for k, g in enumerate(gaps) if g > thr}
        out.append([group_box(ln, i0, i1) for i0, i1 in groups_from_cuts(len(ln["spans"]), cuts)])
    return (out, [ln["ids"] for ln in lines], labels) if with_comps else out


def tesseract_in_lines(binary: np.ndarray, gray: np.ndarray, line_ids: list[list[int]], labels: np.ndarray,
                       lang: str, use_gray: bool, scale, psm: int = 7, pad: int = 12) -> tuple[list, list]:
    """Tesseract reads each line on its own (--psm 7 one text line, or 13 raw line), with only that line's ink.
    This tests Tesseract's word splitting alone, not its layout analysis. Returns word boxes and texts per line."""
    out_boxes, out_texts = [], []
    H, W = binary.shape
    for ids in line_ids:
        mask = np.isin(labels, ids)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            out_boxes.append([])
            out_texts.append([])
            continue
        X0, Y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
        X1, Y1 = min(W, xs.max() + 1 + pad), min(H, ys.max() + 1 + pad)
        m = mask[Y0:Y1, X0:X1]
        if use_gray:
            m = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            crop = np.where(m, gray[Y0:Y1, X0:X1], 255).astype(np.uint8)
        else:
            crop = np.where(m, 0, 255).astype(np.uint8)
        sc = resolve_scale(np.where(mask[Y0:Y1, X0:X1], 0, 255).astype(np.uint8), scale)
        words = sorted(run_tesseract(crop, lang, psm, sc), key=lambda w: -w["box"][2])
        out_boxes.append([(b[0] + X0, b[1] + Y0, b[2] + X0, b[3] + Y0) for b in (w["box"] for w in words)])
        out_texts.append([w["text"] for w in words])
    return out_boxes, out_texts


# ----------------------------------------------------------------------------- scoring

def tighten(box, ink: np.ndarray):
    x0, y0, x1, y1 = (max(0, v) for v in box)
    sub = ink[y0:y1, x0:x1]
    ys, xs = np.nonzero(sub)
    if len(xs) == 0:
        return None
    return (x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1)


def iou_matrix(a: list, b: list) -> np.ndarray:
    if not a or not b:
        return np.zeros((len(a), len(b)))
    A, B = np.array(a, float)[:, None], np.array(b, float)[None]
    iw = np.clip(np.minimum(A[..., 2], B[..., 2]) - np.maximum(A[..., 0], B[..., 0]), 0, None)
    ih = np.clip(np.minimum(A[..., 3], B[..., 3]) - np.maximum(A[..., 1], B[..., 1]), 0, None)
    inter = iw * ih
    area = lambda X: (X[..., 2] - X[..., 0]) * (X[..., 3] - X[..., 1])  # noqa: E731
    return inter / np.maximum(area(A) + area(B) - inter, 1e-9)


def score_boxes(pred: list, gt: list, dc: list, binary: np.ndarray) -> dict:
    """Counts for one page (see the module docstring). Returns tp, fp, fn, ignored and the matched pairs."""
    ink = binary < 128
    gt_ink = np.zeros_like(ink)
    for x0, y0, x1, y1 in gt:
        gt_ink[y0:y1, x0:x1] = True
    gt_ink &= ink
    pred = [b for b in (tighten(b, ink) for b in pred) if b is not None]
    m = iou_matrix(pred, gt)
    matched = []
    if m.size:
        r, c = linear_sum_assignment(-m)
        matched = [(i, j) for i, j in zip(r, c) if m[i, j] >= IOU]
    used = {i for i, _ in matched}
    fp = ignored = 0
    for i, (x0, y0, x1, y1) in enumerate(pred):
        if i in used:
            continue
        n_gt = int(gt_ink[y0:y1, x0:x1].sum())
        n_dc = int(ink[y0:y1, x0:x1].sum()) - n_gt
        if n_dc > n_gt:
            ignored += 1
        else:
            fp += 1
    return {"tp": len(matched), "fp": fp, "fn": len(gt) - len(matched), "ignored": ignored, "pred": pred,
            "matched": matched}


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
            "tp": tp, "fp": fp, "fn": fn}


def split_merge(pred: list, gt: list) -> dict:
    """Why ground-truth words were missed: a word covered by 2+ predicted boxes was SPLIT; a predicted box
    covering 2+ ground-truth words MERGED them (coverage = 50% of the smaller box's area)."""
    def cover(a, b):  # share of a's area inside b
        iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        return iw * ih / max(1, (a[2] - a[0]) * (a[3] - a[1]))
    split = sum(sum(cover(p, g) >= 0.5 for p in pred) >= 2 for g in gt)
    merged = sum(sum(cover(g, p) >= 0.5 for g in gt) >= 2 for p in pred)
    untouched = sum(not any(cover(g, p) > 0.1 for p in pred) for g in gt)   # no box anywhere near it
    return {"gt_words_split": int(split), "pred_boxes_merging_2plus": int(merged), "gt_words_not_found": int(untouched)}
