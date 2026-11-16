"""52-letter dataset (preprocessed binary crops from Step 1) and augmentation.

Every crop is cut to its ink bounding box and fitted, keeping its aspect ratio, into a 64x64
canvas (so a tall alif and a wide ب keep their shapes). Tensors use 1 = ink, 0 = paper.

Augmentations imitate handwriting variation: small rotation/shear/scale/shift, pen width, and
slight elastic wobble. There is deliberately NO random erasing or cropping: removing a dot would
turn ت into ب and silently corrupt the label.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def fit_to_square(binary: np.ndarray, size: int = 64, margin: int = 4, gray: bool = False) -> np.ndarray:
    """binary: uint8 0 = ink / 255 = paper. Returns uint8 ink map (255 = ink) of size x size.
    gray=True: the input is the pre-threshold grayscale; ink intensity is kept (soft faint dots)."""
    ink = (255 - binary) if gray else (binary < 128).astype(np.uint8) * 255
    ys, xs = np.nonzero(ink > 64)
    out = np.zeros((size, size), np.uint8)
    if len(ys) == 0:
        return out
    crop = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    s = (size - 2 * margin) / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = crop
    return out


def augment(ink: np.ndarray, rng: np.random.Generator, a: dict) -> np.ndarray:
    """Random handwriting-like distortion of a 255-ink map (same size in and out)."""
    h, w = ink.shape
    ang = rng.uniform(-a["rotate_deg"], a["rotate_deg"])
    sc = rng.uniform(*a["scale"])
    sh = np.tan(np.radians(rng.uniform(-a["shear_deg"], a["shear_deg"])))
    m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    m[0, 1] += sh
    m[0, 2] += rng.uniform(-a["translate"], a["translate"]) * w - sh * h / 2
    m[1, 2] += rng.uniform(-a["translate"], a["translate"]) * h
    out = cv2.warpAffine(ink, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    if rng.random() < a["elastic_p"]:
        dx = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), 6) * 40
        dy = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), 6) * 40
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        out = cv2.remap(out, gx + dx, gy + dy, cv2.INTER_LINEAR, borderValue=0)
    if rng.random() < a["stroke_p"]:
        k = np.ones((2, 2), np.uint8)
        out = cv2.dilate(out, k) if rng.random() < 0.6 else cv2.erode(out, k)
    return out


def read_index(index_csv: Path) -> list[dict]:
    with open(index_csv, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def render_font_letters(letter_map: dict[int, str], font_files: list[str], size: int = 64) -> list[tuple]:
    """Isolated letters rendered with system fonts -> [(255-ink 64x64, label)]. Multi-character
    classes (digraphs) are skipped: without text shaping they would render unjoined."""
    from PIL import Image, ImageDraw, ImageFont
    out = []
    for path in font_files:
        if not Path(path).exists():
            continue
        font = ImageFont.truetype(path, 96)
        notdef = np.asarray(font.getmask("\ue000"))
        for cid, ch in letter_map.items():
            if len(ch) != 1:
                continue
            canvas = Image.new("L", (160, 160), 255)
            ImageDraw.Draw(canvas).text((30, 10), ch, font=font, fill=0)
            arr = np.asarray(canvas)
            if (arr < 128).sum() < 30 or np.array_equal(np.asarray(font.getmask(ch)), notdef):
                continue  # glyph missing in this font
            out.append((fit_to_square(arr, size), cid - 1))
    return out


class LetterDataset(Dataset):
    """Rows of data/preprocessed/letters52/index.csv for one split. Labels are 0..51 (= class_id - 1).

    targets(label) -> (label, body, dot_position, dot_count) when `structure` is given.
    `synthetic`: optional font-rendered letters; with probability `synthetic_p` a sample is
    replaced by a strongly augmented font letter of a random class.
    """

    def __init__(self, root: Path, rows: list[dict], size: int = 64, aug: dict | None = None, seed: int = 0,
                 structure: dict | None = None, synthetic: list | None = None, synthetic_p: float = 0.0,
                 gray: bool = False):
        self.items = []
        for r in rows:
            img = cv2.imread(str(root / r["relpath"]), cv2.IMREAD_GRAYSCALE)
            self.items.append((fit_to_square(img, size, gray=gray), int(r["class_id"]) - 1, r["relpath"]))
        self.aug = aug
        self.rng = np.random.default_rng(seed)
        self.synthetic, self.synthetic_p = synthetic or [], synthetic_p
        self.aux = None
        if structure:
            bodies = sorted({v[0] for v in structure.values()})
            positions = ["none", "above", "below", "center", "tah"]
            self.aux = {cid - 1: (bodies.index(v[0]), positions.index(v[1]), int(v[2]))
                        for cid, v in structure.items()}
            self.aux_sizes = (len(bodies), len(positions), 5)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ink, label, _ = self.items[i]
        if self.synthetic and self.rng.random() < self.synthetic_p:
            ink, label = self.synthetic[self.rng.integers(len(self.synthetic))]
            strong = {**self.aug, "elastic_p": 0.8, "stroke_p": 0.7}
            ink = augment(ink, self.rng, strong)
        elif self.aug:
            ink = augment(ink, self.rng, self.aug)
        x = torch.from_numpy(ink).float().div(255).unsqueeze(0)
        if self.aux is None:
            return x, label
        return x, torch.tensor((label, *self.aux[label]))
