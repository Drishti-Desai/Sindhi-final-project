"""Where to cut a line into words, without knowing its text (Step 3 levers 1 and 2).

Both work on our own lines (tesseract_ft.boxes.line_structures): merged ink spans in reading order. A cut sits in
the gap after span k; a word is a run of consecutive spans plus the dots/diacritics in its x-range.

Lever 1, snap_cuts: Tesseract (reading the line alone) says roughly where its words are, but its boxes are
  estimated from its character output, not from the ink. Each ink span goes to the Tesseract word box it overlaps
  most horizontally (a span overlapping none goes to the nearest box); a cut is placed between consecutive spans
  that went to different boxes. So the cuts come from Tesseract, the boxes from the ink.
Lever 2, best_cuts: every way of grouping up to `max_spans` consecutive spans is read by the Step 2 recognizer.
  Dynamic programming picks the grouping with the best total of
      per word:  a x (mean log-probability of the recognizer's reading) + b x [reading is a known word] - c
      per cut:   d x (gap / letter height) + e x (+1 if Tesseract also cuts there, -1 if it does not)
  The weights are chosen on the tuning pages only.
"""
from __future__ import annotations

import numpy as np
import torch

from cnn_transformer.words import BOS, EOS, collate, ink_crop, to_ink, to_tensor_image
from common.text import strip_punct
from tesseract_ft.boxes import group_comps


def snap_cuts(line: dict, tess_boxes: list[tuple]) -> set[int] | None:
    """Lever 1. None when Tesseract found no word on this line (the caller falls back to the gap rule)."""
    if not tess_boxes:
        return None
    owner = []
    for x0, x1, _ in line["spans"]:
        ov = [max(0, min(x1, b[2]) - max(x0, b[0])) for b in tess_boxes]
        if max(ov) > 0:
            owner.append(int(np.argmax(ov)))
        else:
            mid = (x0 + x1) / 2
            owner.append(int(np.argmin([min(abs(mid - b[0]), abs(mid - b[2])) for b in tess_boxes])))
    return {k for k in range(len(owner) - 1) if owner[k] != owner[k + 1]}


def tess_cut_flags(line: dict, tess_boxes: list[tuple]) -> list[int]:
    """Per gap: +1 if snapping Tesseract's words to the ink cuts there, -1 if not, 0 if Tesseract found nothing."""
    cuts = snap_cuts(line, tess_boxes)
    if cuts is None:
        return [0] * len(line["gaps"])
    return [1 if k in cuts else -1 for k in range(len(line["gaps"]))]


def candidate_groups(line: dict, max_spans: int) -> list[tuple[int, int]]:
    n = len(line["spans"])
    return [(i, j) for i in range(n) for j in range(i, min(n, i + max_spans))]


@torch.no_grad()
def read_groups(rec, labels: np.ndarray, line: dict, groups: list[tuple[int, int]], device,
                batch: int = 64) -> list[tuple[str, float]]:
    """The recognizer reads each candidate group (only that group's ink): (text, mean log-probability per step,
    end-of-word included)."""
    model, vocab, rc = rec
    items = []
    for i0, i1 in groups:
        cs = group_comps(line, i0, i1)
        x0, y0 = min(c["x"] for c in cs), min(c["y"] for c in cs)
        x1, y1 = max(c["x"] + c["w"] for c in cs), max(c["y"] + c["h"] for c in cs)
        sub = labels[y0:y1, x0:x1]
        crop = np.where(np.isin(sub, [c["i"] for c in cs]), 0, 255).astype(np.uint8)
        crop = np.pad(crop, 4, constant_values=255)
        items.append((to_tensor_image(ink_crop(to_ink(crop)), rc["height"], rc["max_width"]), ""))
    out = []
    for i in range(0, len(items), batch):
        x, w, _, _ = collate(items[i:i + batch], vocab)
        seq, probs = model.greedy(x.to(device), w.to(device), BOS, EOS, 26)
        for s, p in zip(seq.cpu(), probs.cpu()):
            text = vocab.decode(s)
            n = min(len(text) + 1, len(p))
            out.append((text, float(p[:n].clamp_min(1e-6).log().mean()) if n else -10.0))
    return out


def word_features(readings: list[tuple[str, float]], lexicon: set[str]) -> list[tuple[float, float]]:
    """(mean log-probability, 1 if the reading without punctuation is a known word)."""
    return [(lp, 1.0 if strip_punct(t) in lexicon else 0.0) for t, lp in readings]


def best_cuts(line: dict, groups: list[tuple[int, int]], feats: list[tuple[float, float]], tess_flags: list[int],
              a: float, b: float, c: float, d: float, e: float) -> set[int]:
    """Lever 2: dynamic programming over the candidate groups (see the module docstring)."""
    n = len(line["spans"])
    h = max(line["h"], 1.0)
    word = {g: a * lp + b * lex - c for g, (lp, lex) in zip(groups, feats)}
    cut = [d * gap / h + e * f for gap, f in zip(line["gaps"], tess_flags)]
    best = np.full(n + 1, -np.inf)
    best[0] = 0.0
    back = [0] * (n + 1)
    for j in range(1, n + 1):
        for i in range(j - 1, -1, -1):
            s = word.get((i, j - 1))
            if s is None:
                break
            v = best[i] + s + (cut[i - 1] if i > 0 else 0.0)
            if v > best[j]:
                best[j], back[j] = v, i
    cuts, j = set(), n
    while j > 0:
        i = back[j]
        if i > 0:
            cuts.add(i - 1)
        j = i
    return cuts
