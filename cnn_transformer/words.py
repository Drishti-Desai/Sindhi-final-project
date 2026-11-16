"""Word-recognizer data: vocabulary, image preparation, and the three training sources.

Sources
  printed   text2image renders of corpus words (scripts/step2_make_synthetic.py)
  composed  words assembled right-to-left from REAL handwritten isolated-letter crops
            (letters52, split-aware: train crops for training, test crops for the test set)
  real      verified word crops cut from TESS_TRAIN pages (scripts/step2_extract_real_words.py)

Internally every image is an INK MAP: uint8, 255 = ink, 0 = paper, intermediate values allowed.
Composed words use the grayscale letter crops (like the letter CNN), so faint dots stay as soft
grey instead of being thresholded away. Every image is cropped to its ink, scaled to height 64
(keeping aspect ratio) and flipped horizontally, so the encoder sees the word in reading order
(right-to-left becomes left-to-right).
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset

from common.text import normalize

PAD, BOS, EOS, UNK = 0, 1, 2, 3
EXTRA_CHARS = "0123456789،.,:()_-"


class Vocab:
    def __init__(self, chars: str):
        self.itos = ["<pad>", "<bos>", "<eos>", "<unk>"] + sorted(set(chars))
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [BOS] + [self.stoi.get(c, UNK) for c in text] + [EOS]

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i == EOS:
                break
            if i > UNK:
                out.append(self.itos[i])
        return "".join(out)

    @classmethod
    def build(cls, words: list[str], extra: str = EXTRA_CHARS) -> "Vocab":
        return cls("".join(words) + extra)


def label_norm(text: str) -> str:
    """Labels and references for the recognizer: normalised, no harakat (handwriting rarely has them)."""
    return normalize(text, diacritics=False).replace(" ", "")


# ----------------------------------------------------------------------------- image prep

def to_ink(img: np.ndarray, gray: bool = False) -> np.ndarray:
    """Paper-format uint8 (0 = ink, 255 = paper) -> ink map. gray=True keeps the ink intensity
    after removing the crop's own paper level (shadowed crops would otherwise add grey blocks);
    otherwise the image is binary."""
    if not gray:
        return (img < 128).astype(np.uint8) * 255
    ink = (255 - img).astype(np.float32)
    paper = float(np.median(ink))  # a crop is mostly paper: its median is the paper level
    ink = np.clip((ink - paper) * 255.0 / max(1.0, 255.0 - paper), 0, 255).astype(np.uint8)
    ink[ink < 50] = 0               # leftover paper texture
    return ink


def ink_crop(ink: np.ndarray, pad: int = 2, thr: int = 64) -> np.ndarray:
    """Tight crop of an ink map around pixels brighter than `thr`, with a small border."""
    ys, xs = np.nonzero(ink > thr)
    if len(ys) == 0:
        return np.zeros((8, 8), np.uint8)
    c = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return cv2.copyMakeBorder(c, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)


def augment_word(ink: np.ndarray, rng: np.random.Generator, strength: float = 1.0) -> np.ndarray:
    """Handwriting-like distortion of a 255-ink word crop: slant, slight rotation, aspect change,
    elastic wobble and pen width. `strength` > 1 is used for printed renders (they need more)."""
    h, w = ink.shape
    pad = h // 3
    ink = cv2.copyMakeBorder(ink, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    H, W = ink.shape
    shear = rng.uniform(-0.25, 0.25) * strength
    ang = rng.uniform(-3, 3) * strength
    sx = rng.uniform(0.85, 1.15)
    m = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
    m[0, :] *= sx
    m[0, 1] += shear
    m[0, 2] += (1 - sx) * W / 2 - shear * H / 2
    out = cv2.warpAffine(ink, m, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    if rng.random() < 0.6:
        sigma, alpha = max(3.0, h / 8), h / 14 * strength
        dx = cv2.GaussianBlur(rng.uniform(-1, 1, (H, W)).astype(np.float32), (0, 0), sigma)
        dy = cv2.GaussianBlur(rng.uniform(-1, 1, (H, W)).astype(np.float32), (0, 0), sigma)
        dx *= alpha / max(1e-6, np.abs(dx).max())
        dy *= alpha / max(1e-6, np.abs(dy).max())
        gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        out = cv2.remap(out, gx + dx, gy + dy, cv2.INTER_LINEAR, borderValue=0)
    k = max(1, round(h / 40))
    r = rng.random()
    if r < 0.3:
        out = cv2.dilate(out, np.ones((k + 1, k + 1), np.uint8))
    elif r < 0.5:
        out = cv2.erode(out, np.ones((2, 2), np.uint8))
    return ink_crop(out)


def to_tensor_image(ink: np.ndarray, height: int = 64, max_width: int = 512, vpad: int = 6) -> np.ndarray:
    """255-ink crop -> float32 (height, W) in [0,1], height-normalised, flipped to reading order."""
    h, w = ink.shape
    s = (height - 2 * vpad) / h
    nw = max(8, min(max_width, round(w * s)))
    img = cv2.resize(ink, (nw, height - 2 * vpad), interpolation=cv2.INTER_AREA)
    img = cv2.copyMakeBorder(img, vpad, vpad, 0, 0, cv2.BORDER_CONSTANT, value=0)
    return np.ascontiguousarray(img[:, ::-1]).astype(np.float32) / 255.0


def collate(batch, vocab: Vocab):
    """Pad images to the widest, rounded up to a multiple of 32 (few distinct tensor shapes keep
    GPU memory from fragmenting; also a multiple of the encoder's width stride 8), and labels to
    the longest."""
    imgs, labels = zip(*batch)
    width = max(i.shape[1] for i in imgs)
    width = (width + 31) // 32 * 32
    x = torch.zeros(len(imgs), 1, imgs[0].shape[0], width)
    widths = torch.tensor([i.shape[1] for i in imgs])
    for k, i in enumerate(imgs):
        x[k, 0, :, : i.shape[1]] = torch.from_numpy(i)
    enc = [vocab.encode(t) for t in labels]
    L = max(len(e) for e in enc)
    y = torch.full((len(enc), L), PAD, dtype=torch.long)
    for k, e in enumerate(enc):
        y[k, : len(e)] = torch.tensor(e)
    return x, widths, y, list(labels)


# ----------------------------------------------------------------------------- sources

def read_printed(root: Path) -> list[tuple[Path, str]]:
    with open(root / "labels.csv", encoding="utf-8") as f:
        return [(root / r["file"], label_norm(r["label"])) for r in csv.DictReader(f)]


def read_real(verified_csv: Path, crops_root: Path, split: str) -> list[tuple[Path, str]]:
    with open(verified_csv, encoding="utf-8") as f:
        return [(crops_root / r["file"], label_norm(r["label"])) for r in csv.DictReader(f)
                if r["split"] == split and r["verified"] == "1"]


class LetterBank:
    """Real isolated-letter crops of one split, plus typical relative letter heights.
    gray=True reads the grayscale crops (letters52_gray), matching the letter CNN's input."""

    def __init__(self, letters_root: Path, index_rows: list[dict], letter_map: dict[int, str],
                 raw_manifest: list[dict], split: str, gray: bool = False):
        self.crops = defaultdict(list)
        for r in index_rows:
            if r["split"] == split:
                img = cv2.imread(str(letters_root / r["relpath"]), cv2.IMREAD_GRAYSCALE)
                c = ink_crop(to_ink(img, gray), pad=0)
                if c.shape[0] >= 4:
                    self.crops[int(r["class_id"])].append(c)
        # relative height of each class: raw crop height / median crop height on the same sheet
        by_sheet = defaultdict(list)
        for m in raw_manifest:
            by_sheet[m["sheet"]].append(m)
        ratios = defaultdict(list)
        for ms in by_sheet.values():
            med = np.median([m["h"] for m in ms])
            for m in ms:
                ratios[int(m["class_id"])].append(m["h"] / med)
        self.height_ratio = {c: float(np.clip(np.median(v), 0.5, 2.0)) for c, v in ratios.items()}
        # text -> class id; digraphs first (normalised forms: جھ -> جه, گھ -> گه)
        self.char_to_class = {normalize(v, diacritics=False): k for k, v in letter_map.items()}
        self.multi = sorted([t for t in self.char_to_class if len(t) > 1], key=len, reverse=True)

    def tokenize(self, word: str) -> list[int] | None:
        out, i = [], 0
        while i < len(word):
            for t in self.multi:
                if word.startswith(t, i):
                    out.append(self.char_to_class[t])
                    i += len(t)
                    break
            else:
                c = self.char_to_class.get(word[i])
                if c is None or not self.crops.get(c):
                    return None
                out.append(c)
                i += 1
        return out

    def compose(self, word: str, rng: np.random.Generator, base_h: int = 48) -> np.ndarray | None:
        classes = self.tokenize(word)
        if not classes:
            return None
        glyphs = []
        for c in classes:
            g = self.crops[c][rng.integers(len(self.crops[c]))]
            th = max(6, round(base_h * self.height_ratio.get(c, 1.0) * rng.uniform(0.85, 1.15)))
            tw = max(3, round(g.shape[1] * th / g.shape[0]))
            glyphs.append(cv2.resize(g, (tw, th), interpolation=cv2.INTER_AREA))
        H = max(g.shape[0] for g in glyphs) + base_h // 2
        gaps = [round(base_h * rng.uniform(-0.05, 0.3)) for _ in glyphs]
        W = sum(g.shape[1] for g in glyphs) + sum(max(0, x) for x in gaps) + 4
        canvas = np.zeros((H, W), np.uint8)
        x = W - 2  # right-to-left
        for g, gap in zip(glyphs, gaps):
            y = (H - g.shape[0]) // 2 + round(rng.uniform(-0.1, 0.1) * base_h)
            y = int(np.clip(y, 0, H - g.shape[0]))
            x0 = max(0, x - g.shape[1])
            region = canvas[y:y + g.shape[0], x0:x0 + g.shape[1]]
            np.maximum(region, g[:, : region.shape[1]], out=region)
            x = x0 - gap
        return ink_crop(canvas)


class MixedWords(IterableDataset):
    """Endless stream of (image, label) drawn from the sources according to `mix` weights."""

    def __init__(self, printed, real, composed_words, bank, mix: dict, cfg: dict, seed: int):
        self.printed, self.real, self.words, self.bank = printed, real, composed_words, bank
        self.names = [k for k in ("printed", "composed", "real") if mix.get(k, 0) > 0 and self._has(k)]
        w = np.array([mix[k] for k in self.names], float)
        self.p = w / w.sum()
        self.cfg, self.seed = cfg, seed

    def _has(self, k):
        return bool({"printed": self.printed, "composed": self.words, "real": self.real}[k])

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (info.id if info else 0) + random.randrange(10**6))
        h, mw = self.cfg["height"], self.cfg["max_width"]
        while True:
            src = self.names[rng.choice(len(self.names), p=self.p)]
            if src == "composed":
                word = self.words[rng.integers(len(self.words))]
                ink = self.bank.compose(word, rng)
                if ink is None:
                    continue
                ink, label = augment_word(ink, rng, 0.7), word
            else:
                path, label = (self.printed if src == "printed" else self.real)[
                    rng.integers(len(self.printed if src == "printed" else self.real))]
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                ink = augment_word(ink_crop(to_ink(img)), rng, 1.6 if src == "printed" else 1.0)
            if len(label) == 0 or len(label) > self.cfg["max_label_len"]:
                continue
            yield to_tensor_image(ink, h, mw), label


def fixed_set(kind: str, items, bank: LetterBank | None, cfg: dict, n: int, seed: int):
    """Deterministic evaluation set: list of (image, label)."""
    rng = np.random.default_rng(seed)
    h, mw = cfg["height"], cfg["max_width"]
    out = []
    if kind == "composed":
        words = [w for w in items if bank.tokenize(w)]
        for _ in range(n):
            w = words[rng.integers(len(words))]
            out.append((to_tensor_image(bank.compose(w, rng), h, mw), w))
    else:
        sel = items if n <= 0 or n >= len(items) else [items[i] for i in rng.choice(len(items), n, replace=False)]
        for path, label in sel:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            out.append((to_tensor_image(ink_crop(to_ink(img)), h, mw), label))
    return [(i, l) for i, l in out if 0 < len(l) <= cfg["max_label_len"]]
