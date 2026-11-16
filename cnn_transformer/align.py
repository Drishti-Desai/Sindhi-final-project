"""Model-guided ("forced") alignment of a known transcription to handwritten ink.

Width-only alignment (segment.align_page) shifts labels whenever a writer's word gaps are not clearly
bigger than the gaps inside words. Here the recognizer decides instead: for every candidate group of
consecutive ink spans on a line and every plausible word for it, the recognizer scores how likely the
KNOWN label is (teacher-forced log-probability, averaged per character). Dynamic programming then picks
the grouping with the best total score. Rules:
  - a word never crosses a line;
  - a span may be left out (crossed-out words, stray marks, missed numbers) for a fixed penalty;
  - only words within +-band of the width-only guess are considered for a span (keeps it fast).
The model never sees anything it could learn from here: it only ranks cut positions for a given text.
"""
from __future__ import annotations

import numpy as np
import torch

from cnn_transformer.segment import align_page, body_threshold
from cnn_transformer.words import BOS, EOS, PAD, Vocab, collate, ink_crop, to_tensor_image

SKIP_PENALTY = 2.5   # nats; leaving a span out must be clearly better than any reading of it


def line_spans(lines: list[list[dict]]):
    """Per line: merged body spans in READING order (right-to-left) and the small components."""
    out = []
    for line in lines:
        bmin = body_threshold(line) if line else 0
        bodies = [c for c in line if c["a"] >= bmin]
        smalls = [c for c in line if c["a"] < bmin]
        spans = sorted([[c["x"], c["x"] + c["w"], [c]] for c in bodies], key=lambda s: s[0])
        merged = []
        for s in spans:
            if merged and s[0] <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], s[1])
                merged[-1][2] += s[2]
            else:
                merged.append(s)
        out.append((merged[::-1], smalls))
    return out


def segment_comps(spans, smalls, i0: int, i1: int) -> list[dict]:
    """Components of spans i0..i1 (reading order) plus the dots/diacritics lying within their x-range."""
    comps = [c for s in spans[i0:i1 + 1] for c in s[2]]
    x0 = min(s[0] for s in spans[i0:i1 + 1])
    x1 = max(s[1] for s in spans[i0:i1 + 1])
    return comps + [c for c in smalls if x0 <= c["x"] + c["w"] / 2 <= x1]


def crop_ink(labels: np.ndarray, comps: list[dict]) -> np.ndarray:
    x0, y0 = min(c["x"] for c in comps), min(c["y"] for c in comps)
    x1, y1 = max(c["x"] + c["w"] for c in comps), max(c["y"] + c["h"] for c in comps)
    sub = labels[y0:y1, x0:x1]
    return ink_crop(np.isin(sub, [c["i"] for c in comps]).astype(np.uint8) * 255)


@torch.no_grad()
def score_pairs(model, vocab: Vocab, rc: dict, device, crops: list[np.ndarray], pairs: list[tuple[int, str]],
                batch: int = 48) -> np.ndarray:
    """Mean log-probability per character of word w given crop c, for every (crop index, word) pair."""
    model.eval()
    images = [to_tensor_image(c, rc["height"], rc["max_width"]) for c in crops]
    mem_cache = {}
    for i in range(0, len(images), batch):  # encode every crop once
        x, w, _, _ = collate([(im, "") for im in images[i:i + batch]], vocab)
        memory, mask = model.encode(x.to(device), w.to(device))
        for k in range(memory.size(0)):
            valid = int((~mask[k]).sum()) if mask is not None else memory.size(1)
            mem_cache[i + k] = memory[k, :valid].cpu()
    scores = np.empty(len(pairs), dtype=np.float32)
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        mems = [mem_cache[ci] for ci, _ in chunk]
        T = max(m.size(0) for m in mems)
        memory = torch.zeros(len(chunk), T, mems[0].size(1))
        pad = torch.ones(len(chunk), T, dtype=torch.bool)
        for k, m in enumerate(mems):
            memory[k, :m.size(0)] = m
            pad[k, :m.size(0)] = False
        enc = [vocab.encode(wd) for _, wd in chunk]
        L = max(len(e) for e in enc)
        y = torch.full((len(chunk), L), PAD, dtype=torch.long)
        for k, e in enumerate(enc):
            y[k, :len(e)] = torch.tensor(e)
        y = y.to(device)
        logits = model.decode(memory.to(device), pad.to(device), y[:, :-1])
        logp = logits.log_softmax(-1).gather(-1, y[:, 1:, None]).squeeze(-1)
        keep = (y[:, 1:] != PAD).float()
        scores[i:i + len(chunk)] = ((logp * keep).sum(1) / keep.sum(1)).cpu().numpy()
    return scores


def forced_align(binary_labels: np.ndarray, lines: list[list[dict]], words: list[str], model, vocab: Vocab,
                 rc: dict, device, max_spans: int = 12, band: int | None = None):
    """Returns (groups of components per word in reading order, line index per word, score per word,
    number of skipped spans) or None when there are more words than ink spans."""
    per_line = line_spans(lines)
    flat = [(li, si) for li, (spans, _) in enumerate(per_line) for si in range(len(spans))]
    n, m = len(words), len(flat)
    if m < n or n == 0:
        return None
    band = band if band is not None else max(5, round(0.12 * n))  # long pages can drift further
    # width-only guess -> which word each span probably belongs to (for the search band)
    guess = align_page(lines, words)
    k0 = np.zeros(m, dtype=int)
    if guess is not None:
        groups, _ = guess
        owner = {id(c): k for k, g in enumerate(groups) for c in g}
        for f, (li, si) in enumerate(flat):
            k0[f] = owner.get(id(per_line[li][0][si][2][0]), int(f * n / m))
    else:
        k0 = (np.arange(m) * n / m).astype(int)
    # candidate segments within a line, and the words allowed for each
    seg_index, crops, pairs, pair_seg = {}, [], [], []
    for f0, (li, si) in enumerate(flat):
        spans, smalls = per_line[li]
        for L in range(1, max_spans + 1):
            if si + L > len(spans):
                break
            f1 = f0 + L - 1
            seg_index[(f0, f1)] = len(crops)
            crops.append(crop_ink(binary_labels, segment_comps(spans, smalls, si, si + L - 1)))
            for k in range(max(0, k0[f0] - band), min(n, k0[f0] + band + 1)):
                pairs.append((seg_index[(f0, f1)], words[k]))
                pair_seg.append((f0, f1, k))
    scores = score_pairs(model, vocab, rc, device, crops, pairs)
    S = {key: s for key, s in zip(pair_seg, scores)}
    # DP over flat spans: best[k][j] = best score with words 0..k-1 placed and spans 0..j-1 consumed
    NEG = -1e9
    best = np.full((n + 1, m + 1), NEG)
    back = {}
    best[0][0] = 0.0
    for j in range(1, m + 1):
        for k in range(0, n + 1):
            # leave span j-1 out
            if best[k][j - 1] > NEG and best[k][j - 1] - SKIP_PENALTY > best[k][j]:
                best[k][j] = best[k][j - 1] - SKIP_PENALTY
                back[(k, j)] = ("skip", j - 1)
            if k == 0:
                continue
            # word k-1 = spans i..j-1 (same line)
            li = flat[j - 1][0]
            for L in range(1, max_spans + 1):
                i = j - L
                if i < 0 or flat[i][0] != li:
                    break
                s = S.get((i, j - 1, k - 1))
                if s is None or best[k - 1][i] <= NEG:
                    continue
                v = best[k - 1][i] + s
                if v > best[k][j]:
                    best[k][j] = v
                    back[(k, j)] = ("word", i)
    if best[n][m] <= NEG:
        return None
    # trace back
    assign, skipped, k, j = [None] * n, 0, n, m
    while j > 0:
        kind, i = back[(k, j)]
        if kind == "skip":
            skipped += 1
            j = i
        else:
            assign[k - 1] = (i, j - 1, float(S[(i, j - 1, k - 1)]))
            k, j = k - 1, i
    groups, line_of, word_scores = [], [], []
    for (f0, f1, s) in assign:
        li, si0 = flat[f0]
        _, si1 = flat[f1]
        spans, smalls = per_line[li]
        groups.append(segment_comps(spans, smalls, si0, si1))
        line_of.append(li)
        word_scores.append(s)
    return groups, line_of, word_scores, skipped
