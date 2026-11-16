"""Evaluation pages with ground-truth word and line boxes, in the exact frame the boxes were cut in.

Two sources:
  batch 2 (VAL, TEST_UNSEEN_WRITER): turned upright, cleaned with Step 1, circled numbers removed, exactly as
      scripts/step2_extract_batch2_words.py did; boxes come from data/splits/handwritten_batch2/words.csv.
      Kept crops (`verified` = 1: QC-passed or checked by eye) are ground truth; flagged or excluded crops are "don't care".
  TESS_TEST (H1, J2, J3): the Step 1 pages in data/preprocessed/handwritten/; boxes are recomputed with the
      same deterministic split as scripts/step2_extract_real_words.py and checked pixel-for-pixel against the
      saved crops; crops not verified by eye are "don't care".
A line is ground truth only if every word on it is ground truth; other lines are "don't care".
FINAL_TEST pages are never read.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageOps

from cnn_transformer.segment import (assign_lines, components, erase_ruled_lines, remove_circled_numbers,
                                     split_words)
from cnn_transformer.words import label_norm
from common.paths import load_config, load_paths
from common.text import normalize
from preprocess import preprocess_image

MIN_HEIGHT = 150   # same as the batch 2 extraction: tiny line images are upscaled before cleaning

Box = tuple[int, int, int, int]   # x0, y0, x1, y1 (exclusive)


@dataclass
class EvalPage:
    name: str
    split: str
    writer: str
    binary: np.ndarray            # kept ink only (0 = ink), the frame of every box
    gray: np.ndarray              # Step 1 grayscale, same frame, removed components painted white
    words: list[str]              # reference text, reading order
    gt_words: list[Box] = field(default_factory=list)
    dc_words: list[Box] = field(default_factory=list)
    gt_lines: list[Box] = field(default_factory=list)
    dc_lines: list[Box] = field(default_factory=list)
    gt_labels: list[str] = field(default_factory=list)   # label of each gt_words box
    all_words: list[Box] = field(default_factory=list)   # every cut word box (gt and don't care), reading order


def label_norm_words(text: str) -> list[str]:
    return [label_norm(w) for w in text.split() if label_norm(w)]


def load_upright(path, rotate_ccw: str) -> np.ndarray:
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGBA")
    im = Image.alpha_composite(Image.new("RGBA", im.size, "white"), im).convert("RGB")
    if rotate_ccw:
        im = im.rotate(int(rotate_ccw), expand=True, fillcolor="white")
    rgb = np.asarray(im)
    if rgb.shape[0] < MIN_HEIGHT:
        s = MIN_HEIGHT / rgb.shape[0]
        rgb = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    return rgb


def keep_only(binary: np.ndarray, gray: np.ndarray, labels: np.ndarray, kept: list[dict]):
    """Pages as the box finders see them: only the kept components (specks, edge remnants, circled numbers
    and erased rulings gone), and the grayscale with the same pixels painted white."""
    keep = np.isin(labels, [c["i"] for c in kept])
    out_b = np.where(keep, 0, 255).astype(np.uint8)
    out_g = gray.copy()
    out_g[(binary < 128) & ~keep] = 255
    return out_b, out_g


def union(boxes: list[Box]) -> Box:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def filter_comps(comps: list[dict], shape) -> list[dict]:
    H, W = shape
    return [c for c in comps if c["a"] >= 6 and c["x"] > 1 and c["y"] > 1
            and c["x"] + c["w"] < W - 1 and c["y"] + c["h"] < H - 1]


def load_batch2_pages(splits: tuple[str, ...]) -> list[EvalPage]:
    paths, cfg1 = load_paths(), load_config("step1")
    root = paths["raw"]["handwritten_batch2"]
    corrected = paths["raw"]["handwritten_batch2_text_corrected"]
    sp = paths["splits"]["handwritten_batch2"]
    manifest = {m["name"]: m for m in csv.DictReader(open(sp / "manifest.csv", encoding="utf-8"))}
    rows = list(csv.DictReader(open(sp / "words.csv", encoding="utf-8")))
    pages = []
    for name in dict.fromkeys(r["name"] for r in rows if r["split"] in splits):
        m = manifest[name]
        text_path = (corrected / f"{name}.txt" if m["status"] in ("text_mismatch", "ok_text_corrected")
                     else root / m["text"])
        res = preprocess_image(load_upright(root / m["image"], m["rotate_ccw"]), cfg1, profile="page")
        binary, gray = res.image, res.gray
        if m["group"] == "top":
            binary, _ = erase_ruled_lines(binary)
        comps, labels = components(binary)
        comps = filter_comps(comps, binary.shape)
        if m["group"] == "top":
            comps, _ = remove_circled_numbers(comps)
        b, g = keep_only(binary, gray, labels, comps)
        page = EvalPage(name, m["split"], m["writer"], b, g, label_norm_words(text_path.read_text(encoding="utf-8")))
        by_line: dict[str, list] = {}
        for r in (r for r in rows if r["name"] == name):
            box = tuple(map(int, r["box"].split(",")))
            ok = r["verified"] == "1"   # QC-passed or checked by eye; words_review.csv is already applied
            (page.gt_words if ok else page.dc_words).append(box)
            page.all_words.append(box)
            if ok:
                page.gt_labels.append(r["label"])
            by_line.setdefault(r["line"], []).append((box, ok))
        for items in by_line.values():
            (page.gt_lines if all(ok for _, ok in items) else page.dc_lines).append(union([b for b, _ in items]))
        pages.append(page)
    return pages


def load_tess_test_pages() -> list[EvalPage]:
    """H1, J2, J3. The word boxes are recomputed and must reproduce the saved crops exactly."""
    paths, cfg1 = load_paths(), load_config("step1")
    pairs = [p for p in csv.DictReader(open(paths["splits"]["handwritten"] / "pairs.csv", encoding="utf-8"))
             if p["split"] == "TESS_TEST"]
    verified = {r["file"]: r["verified"] == "1" for r in
                csv.DictReader(open(paths["splits"]["handwritten"].parent / "step2_real_words" / "verified.csv",
                                    encoding="utf-8"))}
    crops_root = paths["preprocessed"]["handwritten"].parent / "handwritten_words"
    pages = []
    for p in pairs:
        stem = p["image"][:-4]
        binary = cv2.imread(str(paths["preprocessed"]["handwritten"] / p["image"]), cv2.IMREAD_GRAYSCALE)
        res = preprocess_image(cv2.cvtColor(cv2.imread(str(paths["raw"]["handwritten_images"] / p["image"])),
                                            cv2.COLOR_BGR2RGB), cfg1, profile="page")
        if res.image.shape != binary.shape or np.mean(res.image != binary) > 0.001:
            raise SystemExit(f"{stem}: Step 1 output no longer reproduces data/preprocessed/handwritten")
        gt_text = normalize((paths["raw"]["handwritten_text_gt"] / p["text_gt"]).read_text(encoding="utf-8"),
                            diacritics=False).splitlines()
        comps, labels = components(binary)
        comps = filter_comps(comps, binary.shape)
        b, g = keep_only(binary, res.gray, labels, comps)
        page = EvalPage(stem, "TESS_TEST", p["writer"], b, g,
                        label_norm_words(" ".join(gt_text)))
        for li, (line, text) in enumerate(zip(assign_lines(comps, len(gt_text)), gt_text)):
            words = text.split()
            groups, _ = split_words(line, len(words), words)
            if len(groups) != len(words):
                continue
            items = []
            for wi, (grp, label) in enumerate(zip(groups, words)):
                box = (min(c["x"] for c in grp), min(c["y"] for c in grp),
                       max(c["x"] + c["w"] for c in grp), max(c["y"] + c["h"] for c in grp))
                fn = f"TESS_TEST/{stem}_l{li + 1}_w{wi + 1}.png"
                mask = np.where(np.isin(labels, [c["i"] for c in grp]), 0, 255).astype(np.uint8)
                saved = cv2.imread(str(crops_root / fn), cv2.IMREAD_GRAYSCALE)
                crop = mask[max(0, box[1] - 4):box[3] + 4, max(0, box[0] - 4):box[2] + 4]
                if saved is None or saved.shape != crop.shape or (saved != crop).any():
                    raise SystemExit(f"{fn}: recomputed box does not reproduce the saved crop")
                ok = verified.get(fn, False)
                (page.gt_words if ok else page.dc_words).append(box)
                page.all_words.append(box)
                if ok:
                    page.gt_labels.append(label_norm(label))
                items.append((box, ok))
            (page.gt_lines if all(ok for _, ok in items) else page.dc_lines).append(union([bx for bx, _ in items]))
        pages.append(page)
    return pages
