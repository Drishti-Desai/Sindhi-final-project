"""Reading a page: lines, reading order, recognizer joins (Step 5, option 1), and the Step 4 correction.

Joins: each line is cut into single ink spans (merged ink runs, with their dots and diacritics). The Step 2
recognizer reads every run of up to MAX_SPANS neighbouring spans (only their own ink); dynamic programming keeps the
grouping with the best total recognizer log-probability (mean log-probability per output step x number of steps).
There are no tuned settings.

Gaps (`words="gaps"`, for writers who leave clear spaces between words): each line is cut with the Step 3 gap
rule (tesseract_ft.boxes.own_layout settings in configs/step3_tesseract.yaml) and each piece is read as one word,
without joins. On the unseen writer this was the better reading (docs/STEP5_pipeline.md section 4); the default
stays "joins", as chosen on VAL.
"""
from __future__ import annotations

import gc
import json

import numpy as np
import torch

from common.paths import REPO_ROOT, load_config
from pipeline.layout import line_bbox, order_lines
from tesseract_ft.boxes import group_box, groups_from_cuts, line_structures
from tesseract_ft.cuts import read_groups
from tesseract_ft.pages import filter_comps, keep_only
from xlmr.correct import Lexicon, apply_choices, build_positions, decide, set_confusions, suggest

MAX_SPANS = 6
BOX_CFG = load_config("step3")["box_finder"]
LINE_FINDER = BOX_CFG["line_finder"]
WORDS = ("joins", "gaps")
CORRECT_MODES = ("off", "auto", "suggest")   # off: recognizer only; auto: Step 4 correction; suggest: candidates only


def clean_page(rgb: np.ndarray, erase_rules: bool = False, remove_circled: bool = False) -> np.ndarray:
    """Step 1 preprocessing (page profile), then the clean-ups the box finders expect: specks and edge remnants
    dropped; optionally curved ruled lines and circled question numbers (as on the batch-2 pages)."""
    from cnn_transformer.segment import components, erase_ruled_lines, remove_circled_numbers
    from preprocess import preprocess_image
    res = preprocess_image(rgb, load_config("step1"), profile="page")
    binary = res.image
    if erase_rules:
        binary, _ = erase_ruled_lines(binary)
    comps, labels = components(binary)
    comps = filter_comps(comps, binary.shape)
    if remove_circled:
        comps, _ = remove_circled_numbers(comps)
    return keep_only(binary, res.gray, labels, comps)[0]


def best_grouping(n: int, reads: dict, max_join: int = MAX_SPANS) -> list[tuple[int, int]]:
    """Units 0..n-1 -> runs of <= max_join with the best total log-probability."""
    best = [(0.0, [])] + [None] * n
    for j in range(1, n + 1):
        for i in range(max(0, j - max_join), j):
            text, lp = reads[(i, j - 1)]
            tot = best[i][0] + lp * (len(text) + 1)
            if best[j] is None or tot > best[j][0]:
                best[j] = (tot, best[i][1] + [(i, j - 1)])
    return best[n][1]


def line_units(ln: dict, words: str) -> list[tuple[int, int]]:
    """The units a line is read in: single spans (joins) or the gap rule's pieces (gaps), as span ranges."""
    n = len(ln["spans"])
    if words == "joins":
        return [(k, k) for k in range(n)]
    gaps = ln["gaps"]
    thr = max(BOX_CFG["gap_h"] * ln["h"], BOX_CFG["gap_med"] * float(np.median(gaps)) if gaps else 0.0)
    return groups_from_cuts(n, {k for k, g in enumerate(gaps) if g > thr})


def read_page(binary: np.ndarray, rec, device, region: np.ndarray | None = None, order=order_lines,
              words: str = "joins", stats: dict | None = None) -> list[dict]:
    """Lines in reading order, each {'box', 'words': [{'box', 'text', 'logp'}]}. `region` (evaluation only): a mask;
    units whose ink lies mostly outside it are left out (the unverified parts of a page). `order`: line-piece boxes
    -> reading order (pipeline.layout). `words`: "joins" (default) or "gaps" (see the module docstring). `stats`, if
    given, receives the boxes of every found line piece (for `plausibility`)."""
    if words not in WORDS:
        raise ValueError(f"words must be one of {WORDS}")
    max_join = MAX_SPANS if words == "joins" else 1
    lines, labels = line_structures(binary, LINE_FINDER)
    if stats is not None:
        stats["line_boxes"] = [tuple(int(v) for v in line_bbox(ln)) for ln in lines]
    ink = binary < 128
    out = []
    for li in order([line_bbox(ln) for ln in lines]):
        ln = lines[li]
        units = line_units(ln, words)
        keep = []
        for u0, u1 in units:
            x0, y0, x1, y1 = group_box(ln, u0, u1)
            m = ink[y0:y1, x0:x1]
            keep.append(bool(m.any()) and (region is None or region[y0:y1, x0:x1][m].mean() > 0.5))
        runs, cur = [], []
        for k, ok in enumerate(keep):
            if ok:
                cur.append(k)
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        found = []
        for run in runs:
            n = len(run)
            groups = [(i, j) for i in range(n) for j in range(i, min(n, i + max_join))]
            spans = [(units[run[i]][0], units[run[j]][1]) for i, j in groups]
            reads = dict(zip(groups, read_groups(rec, labels, ln, spans, device)))
            for i, j in best_grouping(n, reads, max_join):
                text, lp = reads[(i, j)]
                if text:
                    found.append({"box": [int(v) for v in group_box(ln, units[run[i]][0], units[run[j]][1])],
                                  "text": text, "logp": round(lp, 4)})
        if found:
            out.append({"box": [int(v) for v in line_bbox(ln)], "words": found})
    return out


# Plausibility warning (usability only, NOT a measured improvement): the thresholds were set by hand after seeing
# the two FINAL_TEST pages of writer D (faint pencil, dotted rulings) that were read as hundreds of false words
# (docs/STEP6_testing.md). Checked on the 132 other pages we have (TESS_TRAIN, TESS_TEST, batch 2): it fires only
# on D1.. (TESS_TRAIN), which fails the same way (33 lines, 259 words), and on no other page.
MAX_STACK = 2.5          # found line pieces lying on top of each other, averaged over the text's height
MANY_LINES, MANY_LINES_STACK = 20, 2.0
MIN_WORDS_SHORT, MAX_SHORT_SHARE = 30, 0.30


def line_stacking(boxes: list[tuple]) -> float:
    """How many found line boxes cover each pixel row, averaged over the rows covered by any (1 = no overlap)."""
    if not boxes:
        return 0.0
    y0, y1 = min(b[1] for b in boxes), max(b[3] for b in boxes)
    cov = np.zeros(y1 - y0 + 1)
    for b in boxes:
        cov[b[1] - y0:b[3] - y0] += 1
    return float(cov[cov > 0].mean())


def plausibility(line_boxes: list[tuple], tokens: list[str]) -> list[str]:
    """Warnings when the page probably was not read as text (rulings, speckle or pencil fragments read as words)."""
    warn = []
    stack = line_stacking(line_boxes)
    if stack >= MAX_STACK or (len(line_boxes) >= MANY_LINES and stack >= MANY_LINES_STACK):
        warn.append(f"{len(line_boxes)} text-line pieces were found, lying on top of each other {stack:.1f} deep on "
                    f"average: ruled or dotted lines, speckle or broken pencil strokes are probably being read as text")
    letters = [sum("\u0620" <= c <= "\u06ff" and c not in "،؛؟۔٬" for c in t) for t in tokens]
    short = float(np.mean([n <= 1 for n in letters])) if letters else 0.0
    if len(tokens) >= MIN_WORDS_SHORT and short >= MAX_SHORT_SHARE:
        warn.append(f"{short:.0%} of the {len(tokens)} words read are single letters or marks: much of the ink is "
                    f"probably not text")
    if warn:
        warn.append("The output is probably unreliable. Best results: pen on plain paper, an upright photo, no pencil, "
                    "no ruled or dotted paper (README, 'How to get good results').")
    return warn


class Corrector:
    """The Step 4 final variant (Sindhi-trained XLM-R + detector) with its validation weights. The two models are
    loaded one at a time to stay within the memory cap."""

    def __init__(self, device):
        cfg = load_config("step4")
        res = REPO_ROOT / "results" / "step4"
        val = json.loads((res / "val_comparison.json").read_text(encoding="utf-8"))
        self.variant = val["final_variant"]
        if self.variant != "xlmr_sindhi_det":
            raise SystemExit(f"Step 4 final variant is {self.variant}; Corrector implements xlmr_sindhi_det")
        self.w = val["weights"][self.variant]
        counts = {l.split("\t")[0]: int(l.split("\t")[1])
                  for l in open(REPO_ROOT / "data" / "preprocessed" / "step4" / "lexicon.tsv", encoding="utf-8")}
        self.lex = Lexicon(counts)
        set_confusions(json.loads((res / "error_model.json").read_text(encoding="utf-8")))
        self.mlm = REPO_ROOT / cfg["mlm"]["out"] / "best"
        self.det = REPO_ROOT / cfg["detector"]["out"] / "best"
        self.device = device

    def correct(self, pages: list[list[str]], mode: str = "auto") -> list[tuple[list[str], list[str], list[dict]]]:
        """Token lists (one per page) -> (output tokens, action per input token, suggestions). "auto" changes the
        words; "suggest" changes nothing and lists, for each word "auto" would change, its top-3 candidates
        (xlmr.correct.suggest)."""
        if mode not in ("auto", "suggest"):
            raise ValueError(f"mode must be 'auto' or 'suggest', not {mode!r}")
        from xlmr.lm import Detector, MaskedScorer
        P = [build_positions(t, self.lex) for t in pages]
        scorer = MaskedScorer(self.mlm, self.device)
        for t, p in zip(pages, P):
            if t:
                scorer.score(t, p)
        del scorer
        free(self.device)
        det = Detector(self.det, self.device)
        probs = det.probs([t for t in pages if t])
        del det
        free(self.device)
        it = iter(probs)
        out = []
        for t, p in zip(pages, P):
            if not t:
                out.append(([], [], []))
                continue
            for pos, q in zip(p, next(it)):
                pos.flags = {"det": q > self.w["theta"]}
            if mode == "auto":
                out.append((*decide(p, self.w, flag_known=True), []))
            else:
                out.append((list(t), ["keep"] * len(t), suggest(t, p, self.w, flag_known=True)))
        return out


def free(device) -> None:
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()


def corrected_lines(lines: list[dict], out: list[str], actions: list[str]) -> list[list[str]]:
    """Put the corrected tokens back on their lines (a merge across two lines goes to the first)."""
    line_of = [li for li, ln in enumerate(lines) for _ in ln["words"]]
    res: list[list[str]] = [[] for _ in lines]
    k = 0
    for i, act in enumerate(actions):
        if act == "merged":
            continue
        n = 2 if act == "split" else 1
        res[line_of[i]] += out[k:k + n]
        k += n
    return res


def correct_mode(value) -> str:
    """True / False (the old boolean switch) or a mode name -> "off" | "auto" | "suggest"."""
    if value is True:
        return "auto"
    if value is False or value is None:
        return "off"
    if value not in CORRECT_MODES:
        raise ValueError(f"correction mode must be one of {CORRECT_MODES}, not {value!r}")
    return value


def finish_page(page: dict, out: list[str], actions: list[str]) -> dict:
    """Set a page's output words, actions, per-line words and text."""
    page["corrected"], page["actions"] = out, actions
    for ln, words in zip(page["lines"], corrected_lines(page["lines"], out, actions)):
        ln["corrected"] = words
    page["text"] = "\n".join(" ".join(ln["corrected"]) for ln in page["lines"])
    return page


def add_suggestion_boxes(page: dict) -> None:
    """Give each suggestion the box of its word (joined with the next word's box if a candidate merges them)."""
    boxes = [w["box"] for ln in page["lines"] for w in ln["words"]]
    for sg in page["suggestions"]:
        i = sg["index"]
        b = list(boxes[i])
        if any(c["consumes"] == 2 for c in sg["candidates"]) and i + 1 < len(boxes):
            n = boxes[i + 1]
            b = [min(b[0], n[0]), min(b[1], n[1]), max(b[2], n[2]), max(b[3], n[3])]
            sg["next_reading"] = page["tokens"][i + 1]
        sg["box"] = b


def apply_page_choices(page: dict, choices: dict[int, int]) -> dict:
    """Suggest mode: apply the user's picks (token index -> 0 keep, n = n-th candidate) to a page."""
    page["choices"] = {int(k): int(v) for k, v in choices.items()}
    return finish_page(page, *apply_choices(page["tokens"], page["suggestions"], page["choices"]))


def word_crop(binary: np.ndarray, box, pad: int = 6) -> np.ndarray:
    x0, y0, x1, y1 = box
    return binary[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad].copy()


def load_rgb(path) -> np.ndarray:
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGBA")
    im = Image.alpha_composite(Image.new("RGBA", im.size, "white"), im).convert("RGB")
    return np.asarray(im)

