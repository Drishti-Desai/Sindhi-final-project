"""Word segmentation helpers shared by the real-word extraction scripts (Step 2).

Handwritten lines are cut into words by aligning the known transcription to the ink:
connected components -> lines -> horizontal "spans" (letter bodies that overlap in x) -> the n-1
cuts between spans that best fit both the gap sizes and each word's expected width (dynamic
programming). Small components (dots, diacritics, commas) are attached to the nearest word afterwards.
"""
from __future__ import annotations

import cv2
import numpy as np


def components(binary: np.ndarray):
    n, lab, st, cen = cv2.connectedComponentsWithStats((binary < 128).astype(np.uint8), 8)
    return [dict(i=i, x=st[i, 0], y=st[i, 1], w=st[i, 2], h=st[i, 3], a=st[i, 4], cy=cen[i, 1])
            for i in range(1, n)], lab


def assign_lines(comps: list[dict], n_lines: int) -> list[list[dict]]:
    """1-D k-means on the vertical centres of the bigger components."""
    if not comps:
        return [[] for _ in range(n_lines)]
    areas = np.array([c["a"] for c in comps])
    big = [c for c in comps if c["a"] >= np.median(areas)]
    ys = np.array([c["cy"] for c in big])
    centers = np.quantile(ys, (np.arange(n_lines) + 0.5) / n_lines)
    for _ in range(50):
        lab = np.argmin(np.abs(ys[:, None] - centers[None, :]), 1)
        new = np.array([ys[lab == k].mean() if (lab == k).any() else centers[k] for k in range(n_lines)])
        if np.allclose(new, centers):
            break
        centers = new
    order = np.argsort(centers)
    centers = centers[order]
    lines = [[] for _ in range(n_lines)]
    for c in comps:
        lines[int(np.argmin(np.abs(centers - c["cy"])))].append(c)
    return lines


NARROW = set("ارڙزژدڌڏڊڍذوء۽0123456789")
PUNCT_CH = set("،.,:()_-")


def expected_width(word: str) -> float:
    """Rough relative ink width of a word from its letters (narrow letters count less)."""
    return sum(0.3 if ch in PUNCT_CH else 0.55 if ch in NARROW else 1.0 for ch in word) + 0.4


def choose_cuts(merged: list, words_visual: list[str], alpha: float = 25.0) -> list[int]:
    """Dynamic programming: choose n-1 cut positions (cut after span i) maximising
    sum(log(gap)) - alpha * sum((segment width share - expected width share)^2)."""
    m, n = len(merged), len(words_visual)
    left = np.array([s[0] for s in merged], float)
    right = np.array([s[1] for s in merged], float)
    total_w = right[-1] - left[0]
    exp = np.array([expected_width(w) for w in words_visual])
    exp = exp / exp.sum()
    gap = np.array([max(1.0, left[i + 1] - right[i]) for i in range(m - 1)])
    reward = np.log(gap)

    def seg_cost(i0, i1, k):  # spans i0..i1 form word k (visual order)
        return alpha * ((right[i1] - left[i0]) / total_w - exp[k]) ** 2

    NEG = -1e18
    # dp[k][j]: best score with words 0..k placed and word k ending at span j
    dp = np.full((n, m), NEG)
    back = np.zeros((n, m), int)
    for j in range(m):
        dp[0][j] = -seg_cost(0, j, 0)
    for k in range(1, n):
        for j in range(k, m):
            best, arg = NEG, -1
            for i in range(k - 1, j):  # previous word ends at span i, cut after i
                v = dp[k - 1][i] + reward[i] - seg_cost(i + 1, j, k)
                if v > best:
                    best, arg = v, i
            dp[k][j], back[k][j] = best, arg
    cuts, j = [], m - 1
    for k in range(n - 1, 0, -1):
        i = back[k][j]
        cuts.append(i)
        j = i
    return sorted(cuts)


def split_words(line: list[dict], n_words: int, words: list[str] | None = None):
    """Return word groups right-to-left and a confidence for the chosen cuts.

    Only letter BODIES decide where words are cut. Small components (dots, diacritics, commas)
    are attached afterwards to the body group nearest in x, so a lone comma or dot can never
    become a "word" and shift all following labels.
    """
    if not line:
        return [], 0.0
    areas = np.array([c["a"] for c in line])
    body_min = 0.25 * np.median(areas[areas >= np.median(areas)])
    bodies = [c for c in line if c["a"] >= body_min]
    smalls = [c for c in line if c["a"] < body_min]
    # merge bodies whose x-ranges overlap (stacked parts of one letter group)
    spans = sorted([[c["x"], c["x"] + c["w"], [c]] for c in bodies], key=lambda s: s[0])
    merged = [spans[0]]
    for s in spans[1:]:
        if s[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], s[1])
            merged[-1][2] += s[2]
        else:
            merged.append(s)
    gaps = [(merged[i + 1][0] - merged[i][1], i) for i in range(len(merged) - 1)]
    if len(merged) < n_words:
        return [], 0.0
    if words is not None and n_words > 1:
        cut_after = set(choose_cuts(merged, words[::-1]))  # spans run left-to-right = words reversed
        chosen = [g for g in gaps if g[1] in cut_after]
    else:
        chosen = sorted(gaps, reverse=True)[: n_words - 1]
        cut_after = {i for _, i in chosen}
    unchosen = [g for g, i in gaps if i not in cut_after]
    conf = (min(g for g, _ in chosen) / max(1, max(unchosen))) if chosen and unchosen else 9.9
    words, cur = [], []
    for i, s in enumerate(merged):
        cur += s[2]
        if i in cut_after:
            words.append(cur)
            cur = []
    words.append(cur)
    ranges = [(min(c["x"] for c in g), max(c["x"] + c["w"] for c in g)) for g in words]
    for c in smalls:
        cx = c["x"] + c["w"] / 2
        dist = [0 if a0 <= cx <= a1 else min(abs(cx - a0), abs(cx - a1)) for a0, a1 in ranges]
        words[int(np.argmin(dist))].append(c)
    return words[::-1], conf  # right-to-left = reading order


# ----------------------------------------------------------------------------- multi-line pages
# (batch 2: the transcription gives the word sequence but not where the visual lines break)

def body_threshold(comps: list[dict]) -> float:
    """Components at least this big are letter bodies; smaller ones are dots/diacritics/specks."""
    areas = np.array([c["a"] for c in comps])
    return 0.25 * float(np.median(areas[areas >= np.median(areas)]))


def page_bodies(comps: list[dict], height_frac: float = 0.35) -> float:
    """Height a component needs to count as a letter body on a whole page: height_frac x the 90th
    percentile height. Ruled-line dashes, paper specks and dots are far shorter than real letters."""
    return height_frac * float(np.percentile([c["h"] for c in comps], 90))


def cluster_lines_simple(comps: list[dict], gap_factor: float = 0.6, min_line_share: float = 0.15) -> list[list[dict]]:
    """Lines by vertical position only: letter bodies sorted by centre height start a new line where a
    centre lies more than gap_factor x the median body height below the current line's mean. Good for
    straight lines; splits strongly sloping ones (then cluster_lines, which tracks slope, does better)."""
    if not comps:
        return []
    hmin = page_bodies(comps)
    bodies = sorted([c for c in comps if c["h"] >= hmin], key=lambda c: c["cy"])
    h_med = float(np.median([c["h"] for c in bodies]))
    lines, cur = [], [bodies[0]]
    for c in bodies[1:]:
        if c["cy"] - np.mean([b["cy"] for b in cur]) > gap_factor * h_med:
            lines.append(cur)
            cur = [c]
        else:
            cur.append(c)
    lines.append(cur)
    ink = [sum(c["a"] for c in ln) for ln in lines]
    keep = [ln for ln, a in zip(lines, ink) if a >= min_line_share * np.percentile(ink, 75)]
    centers = np.array([np.mean([c["cy"] for c in ln]) for ln in keep])
    heights = np.array([np.median([c["h"] for c in ln]) for ln in keep])
    kept_ids = {id(c) for ln in keep for c in ln}
    for c in comps:
        if id(c) in kept_ids:
            continue
        k = int(np.argmin(np.abs(centers - c["cy"])))
        if abs(centers[k] - c["cy"]) <= 1.5 * heights[k]:
            keep[k].append(c)
    return keep


def cluster_lines(comps: list[dict], tol_factor: float = 0.7, min_line_share: float = 0.15) -> list[list[dict]]:
    """Group components into visual lines, top to bottom, without knowing how many lines there are.

    Lines are TRACKED from right to left (Sindhi reading order): each letter body (tall enough, see
    page_bodies) joins the line whose members nearest to it in x sit at a similar height (within
    tol_factor x the median body height); otherwise it starts a new line. Following nearby members
    lets a gently sloping line stay one line instead of being cut in two. Lines with less than
    min_line_share of the ink of the 75th-percentile line (page numbers, stray marks) are dropped.
    Everything else joins the nearest kept line, unless it lies far from all of them.
    """
    if not comps:
        return []
    hmin = page_bodies(comps)
    bodies = sorted([c for c in comps if c["h"] >= hmin], key=lambda c: -(c["x"] + c["w"] / 2))
    h_med = float(np.median([c["h"] for c in bodies]))
    # line spacing from a quick first pass (split sorted centres at gaps > 0.6 x body height);
    # closely written lines need a tighter tolerance than 0.7 x body height
    ys = np.sort([c["cy"] for c in bodies])
    starts = [ys[0]] + [y for y, d in zip(ys[1:], np.diff(ys)) if d > 0.6 * h_med]
    tol = tol_factor * h_med
    if len(starts) >= 3:
        tol = min(tol, 0.45 * float(np.median(np.diff(starts))))
    lines: list[list[dict]] = []

    def local_y(line: list[dict], c: dict) -> float:
        """Where this line is expected to be at c's x: a straight-line fit through its 4 members
        nearest in x (so a sloping line is followed), or their mean height if they are too close."""
        cx = c["x"] + c["w"] / 2
        near = sorted(line, key=lambda d: abs(d["x"] + d["w"] / 2 - cx))[:4]
        xs = np.array([d["x"] + d["w"] / 2 for d in near])
        ys = np.array([d["cy"] for d in near])
        if len(near) >= 2 and xs.max() - xs.min() > 2 * h_med:
            slope = float(np.clip(np.polyfit(xs, ys, 1)[0], -0.3, 0.3))
            return float(ys.mean() + slope * (cx - xs.mean()))
        return float(ys.mean())

    for c in bodies:
        dists = [abs(local_y(ln, c) - c["cy"]) for ln in lines]
        k = int(np.argmin(dists)) if dists else -1
        if k >= 0 and dists[k] <= tol:
            lines[k].append(c)
        else:
            lines.append([c])
    # merge pieces of one line that tracking split (a gap, a tall alif or a low tail can start a new
    # "line"): a smaller line whose bodies lie, on median, within tol of a bigger line's path joins it
    merged = True
    while merged and len(lines) > 1:
        merged = False
        lines.sort(key=lambda ln: -len(ln))
        for i in range(len(lines) - 1, 0, -1):
            small = lines[i]
            for j in range(i):
                if np.median([abs(local_y(lines[j], c) - c["cy"]) for c in small]) <= tol:
                    lines[j] += small
                    del lines[i]
                    merged = True
                    break
            if merged:
                break
    lines.sort(key=lambda ln: np.mean([c["cy"] for c in ln]))
    ink = [sum(c["a"] for c in ln) for ln in lines]
    keep = [ln for ln, a in zip(lines, ink) if a >= min_line_share * np.percentile(ink, 75)]
    kept_ids = {id(c) for ln in keep for c in ln}
    heights = [float(np.median([c["h"] for c in ln])) for ln in keep]
    for c in comps:
        if id(c) in kept_ids:
            continue
        d = [abs(local_y(ln, c) - c["cy"]) for ln in keep]
        k = int(np.argmin(d))
        if d[k] <= 1.5 * heights[k]:  # far-away specks (margins, the other page) are dropped
            keep[k].append(c)
    return keep


def remove_circled_numbers(comps: list[dict]) -> tuple[list[dict], int]:
    """Drop hand-drawn circles round question numbers, and the digits inside them.

    A ring: roughly square, clearly taller than typical letter bodies (>= 1.3x), ink covering under 25%
    of its box (a thin loop, not a filled letter), centred in the outer 20% of the page width (numbers
    sit in the margin), and containing only small components (digits).
    """
    if not comps:
        return comps, 0
    hmin = page_bodies(comps)
    h_med = float(np.median([c["h"] for c in comps if c["h"] >= hmin]))
    x_lo = min(c["x"] for c in comps)
    x_hi = max(c["x"] + c["w"] for c in comps)
    width = max(1, x_hi - x_lo)
    rings = []
    for c in comps:
        cx = (c["x"] + c["w"] / 2 - x_lo) / width
        square = 0.6 <= c["w"] / max(1, c["h"]) <= 1.6
        if not (square and c["h"] >= 1.3 * h_med and c["a"] < 0.25 * c["w"] * c["h"] and (cx < 0.2 or cx > 0.8)):
            continue
        inside = [d for d in comps if d is not c and d["x"] >= c["x"] and d["y"] >= c["y"]
                  and d["x"] + d["w"] <= c["x"] + c["w"] and d["y"] + d["h"] <= c["y"] + c["h"]]
        if inside and all(d["h"] < 0.8 * c["h"] for d in inside):
            rings.append((c, inside))
    drop = {id(d) for c, ins in rings for d in [c, *ins]}
    return [c for c in comps if id(c) not in drop], len(rings)


def align_page(lines: list[list[dict]], words: list[str], line_gap_factor: float = 3.0,
               alpha: float = 25.0) -> tuple[list[list[dict]], list[int]] | None:
    """Assign the page's words (reading order) to ink across all lines at once.

    Every line's spans are laid out right-to-left on one virtual axis, line after line, with a gap of
    line_gap_factor x the median inter-span gap between lines, then choose_cuts() places the n-1 word
    boundaries. Returns (word groups in reading order, line index of each word) or None if there are
    fewer spans than words.
    """
    seq, smalls_by_line, gaps = [], [], []
    for li, line in enumerate(lines):
        bmin = body_threshold(line) if line else 0
        bodies = [c for c in line if c["a"] >= bmin]
        smalls_by_line.append([c for c in line if c["a"] < bmin])
        spans = sorted([[c["x"], c["x"] + c["w"], [c]] for c in bodies], key=lambda s: s[0])
        merged = []
        for s in spans:
            if merged and s[0] <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], s[1])
                merged[-1][2] += s[2]
            else:
                merged.append(s)
        gaps += [merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1)]
        smalls_by_line[-1] = (smalls_by_line[-1], merged)
    G = line_gap_factor * (float(np.median(gaps)) if gaps else 20.0) + 1.0
    offset = 0.0
    for li, (_, merged) in enumerate(smalls_by_line):
        if not merged:
            continue
        right_edge = merged[-1][1]
        for s in reversed(merged):  # right-to-left = reading order
            seq.append([offset + right_edge - s[1], offset + right_edge - s[0], list(s[2]), li])
        offset = seq[-1][1] + G
    if len(seq) < len(words):
        return None
    cuts = set(choose_cuts(seq, words, alpha)) if len(words) > 1 else set()
    groups, line_of, cur, cur_lines = [], [], [], []
    for i, s in enumerate(seq):
        cur += s[2]
        cur_lines.append(s[3])
        if i in cuts:
            groups.append(cur)
            line_of.append(cur_lines[0])
            cur, cur_lines = [], []
    groups.append(cur)
    line_of.append(cur_lines[0])
    # attach dots/diacritics to the nearest word on the same line
    for li, (smalls, _) in enumerate(smalls_by_line):
        cand = [k for k, g in enumerate(groups) if any(c in g for c in lines[li])]
        for c in smalls:
            if not cand:
                continue
            cx = c["x"] + c["w"] / 2

            def dist(k):
                xs = [(d["x"], d["x"] + d["w"]) for d in groups[k] if d in lines[li]]
                return min(0 if a <= cx <= b else min(abs(cx - a), abs(cx - b)) for a, b in xs) if xs else 1e9
            groups[min(cand, key=dist)].append(c)
    return groups, line_of


def erase_ruled_lines(binary: np.ndarray, min_width_frac: float = 0.3, max_fill: float = 0.12) -> tuple[np.ndarray, int]:
    """Remove curved ruled lines (bent notebook pages) that Step 1's straight-line removal misses.

    A ruling = a component wider than min_width_frac of the page whose ink fills under max_fill of its
    box. For each, a quadratic curve is fitted through the median ink row of every column, and only a
    thin band around that curve (the line's own thickness + 1 px) is erased, so letters touching the
    ruling lose a sliver rather than being deleted with it. Returns (cleaned binary, rulings removed).
    """
    ink = binary < 128
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    W = binary.shape[1]
    out = binary.copy()
    removed = 0
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w < min_width_frac * W or a > max_fill * w * h:
            continue
        ys, xs = np.nonzero(lab[y:y + h, x:x + w] == i)
        cols = np.unique(xs)
        med = np.array([np.median(ys[xs == c]) for c in cols])
        coef = np.polyfit(cols, med, 2)
        resid = np.abs(ys - np.polyval(coef, xs))
        thick = max(2.0, float(np.percentile(np.bincount(xs)[cols], 50)))  # typical ink rows per column
        band = resid <= thick / 2 + 1
        out[ys[band] + y, xs[band] + x] = 255
        removed += 1
    return out, removed
