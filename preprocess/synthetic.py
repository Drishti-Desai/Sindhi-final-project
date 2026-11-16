"""Synthetic test images with a known ground-truth ink mask.

Real photos have no pixel-level ground truth, so methods are compared (and unit-tested) on
rendered Sindhi text placed on controlled backgrounds. Every condition from the Step 1 brief is
covered: transparent, coloured, gradient, shadow, texture, low contrast, dark, noise, tilt,
ruled lines, faint pencil, and tiny letter crops.

Notes:
- Pillow here has no Arabic shaping (no libraqm), so letters render in isolated forms. That is
  fine for binarisation: what matters is strokes and dots, not correct joining.
- Small dot clusters are added explicitly (1-4 dots per word, above or below), so dot
  preservation can be measured even if a font lacks a glyph.
- Pen pressure is simulated with a smooth random field that fades parts of strokes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LETTERS = "ابٻڀتٿٽٺثپجڄڃچڇحخدڌڏڊڍذرڙزسشصضطظعغفڦقڪکگڳڱلمنڻوهءي"

PAGE_CONDITIONS = ["clean", "transparent", "colored", "gradient", "shadow", "texture",
                   "low_contrast", "dark", "noise", "ruled", "pencil_faint", "desk"]
ALL_CONDITIONS = PAGE_CONDITIONS + ["tilt", "glyph_small"]


@dataclass
class Sample:
    image: np.ndarray        # RGB (or RGBA for "transparent") uint8, the degraded input
    gt_mask: np.ndarray      # bool, True = ink, same geometry as the *clean* page
    dot_mask: np.ndarray     # int32 labels of the ground-truth dots (0 = none)
    condition: str
    tilt_deg: float = 0.0    # for "tilt": the rotation that undoes the tilt (pass to ops.rotate)
    glyph_scale: int = 1     # for "glyph_small": gt_mask is this many times larger than image


def _font(paths: list[str], size: int, rng: np.random.Generator) -> ImageFont.FreeTypeFont:
    usable = [p for p in paths if Path(p).exists()]
    if usable:
        return ImageFont.truetype(usable[int(rng.integers(len(usable)))], size)
    return ImageFont.load_default(size=size)


def _smooth_field(rng: np.random.Generator, h: int, w: int, cells: int, lo: float, hi: float) -> np.ndarray:
    """Smooth random values in [lo, hi] (value noise): used for pressure, texture, lighting."""
    coarse = rng.random((cells, max(2, int(cells * w / max(h, 1)))))
    f = cv2.resize(coarse.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    f = (f - f.min()) / max(1e-6, f.max() - f.min())
    return lo + (hi - lo) * f


def render_text(rng: np.random.Generator, h: int, w: int, fonts: list[str], n_lines: int = 3
                ) -> tuple[np.ndarray, np.ndarray]:
    """Return (ink alpha in [0,1] float32, dot label image int32)."""
    canvas = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(canvas)
    line_h = h / (n_lines + 1)
    size = int(line_h * 0.55)
    font = _font(fonts, size, rng)
    dots = Image.new("L", (w, h), 0)
    ddraw = ImageDraw.Draw(dots)
    stroke = max(2, size // 12)
    for li in range(n_lines):
        y = int(line_h * (li + 0.6))
        x = w - int(w * 0.05)
        while x > w * 0.12:
            word = "".join(rng.choice(list(LETTERS), int(rng.integers(2, 6))))
            box = draw.textbbox((0, 0), word, font=font)
            ww = box[2] - box[0]
            if x - ww < w * 0.05:
                break
            draw.text((x - ww, y), word, font=font, fill=255)
            # explicit dot cluster: 1-4 round dots ~1.2 stroke widths across, above or below
            r = max(1.5, stroke * 0.6)
            cx = x - ww / 2 + rng.uniform(-ww / 4, ww / 4)
            cy = y - size * 0.25 if rng.random() < 0.5 else y + size * 1.25
            for k in range(int(rng.integers(1, 5))):
                dx = (k - 1.5) * r * 2.8
                ddraw.ellipse([cx + dx - r, cy - r, cx + dx + r, cy + r], fill=255)
            x -= ww + int(size * rng.uniform(0.4, 0.9))
    if rng.random() < 0.5:  # pen width variation
        canvas = Image.fromarray(cv2.dilate(np.asarray(canvas), np.ones((2, 2), np.uint8)))
    ink = np.maximum(np.asarray(canvas), np.asarray(dots)).astype(np.float32) / 255.0
    _, dot_labels = cv2.connectedComponents((np.asarray(dots) > 127).astype(np.uint8), connectivity=8)
    return ink, dot_labels.astype(np.int32)


def _compose(ink: np.ndarray, paper_rgb, ink_rgb, pressure: np.ndarray) -> np.ndarray:
    a = (ink * pressure)[..., None]
    paper = np.array(paper_rgb, np.float32)
    inkc = np.array(ink_rgb, np.float32)
    return paper * (1 - a) + inkc * a


def make_sample(condition: str, rng: np.random.Generator, fonts: list[str],
                h: int = 360, w: int = 900) -> Sample:
    if condition == "glyph_small":
        return _glyph_sample(rng, fonts)
    ink, dots = render_text(rng, h, w, fonts, n_lines=int(rng.integers(2, 4)))
    gt = ink > 0.5
    pressure = _smooth_field(rng, h, w, 6, 0.6, 1.0)  # uneven pen pressure on every sample
    paper, inkc = (245, 245, 242), (25, 25, 30)
    tilt = 0.0

    if condition == "colored":
        paper = tuple(int(v) for v in rng.integers(170, 235, 3))
        inkc = (30, 40, 140)
    elif condition == "low_contrast":
        paper, inkc = (205, 205, 205), (150, 150, 150)
    elif condition == "dark":
        paper, inkc = (35, 38, 40), (230, 230, 225)
    elif condition == "pencil_faint":
        paper, inkc = (228, 226, 220), (150, 150, 155)
        pressure = _smooth_field(rng, h, w, 6, 0.45, 1.0)

    img = _compose(ink, paper, inkc, pressure)

    if condition == "gradient":
        x = np.linspace(rng.uniform(0.45, 0.6), 1.0, w, dtype=np.float32)
        img *= (x[None, :] if rng.random() < 0.5 else x[None, ::-1])[..., None]
    elif condition == "shadow":
        mask = np.zeros((h, w), np.float32)
        pts = np.array([[0, 0], [int(w * rng.uniform(0.3, 0.7)), 0],
                        [int(w * rng.uniform(0.2, 0.6)), h], [0, h]], np.int32)
        cv2.fillPoly(mask, [pts], 1.0)
        mask = cv2.GaussianBlur(mask, (0, 0), 25)
        img *= (1 - mask * rng.uniform(0.5, 0.65))[..., None]
    elif condition in ("texture", "pencil_faint"):
        tex = _smooth_field(rng, h, w, 30, -18, 18) + rng.normal(0, 6, (h, w))
        img += tex[..., None]
    elif condition == "noise":
        img += rng.normal(0, 22, img.shape)
        sp = rng.random((h, w))
        img[sp < 0.01] = 0
        img[sp > 0.99] = 255
    elif condition == "ruled":
        spacing = int(h / 4)
        for y in range(spacing // 2, h, spacing // 2):
            cv2.line(img, (0, y), (w, y), (140, 160, 200), 2)
        cv2.line(img, (int(w * 0.93), 0), (int(w * 0.93), h), (200, 120, 120), 2)

    if condition == "pencil_faint":
        img = cv2.GaussianBlur(img, (0, 0), 0.8)
    img = np.clip(img, 0, 255).astype(np.uint8)

    if condition == "desk":
        # phone photo of the sheet lying on a textured, darker table, slightly rotated
        H, W = int(h * 1.45), int(w * 1.3)
        desk = _smooth_field(rng, H, W, 12, 85, 140)[..., None] * np.array([1.0, 0.85, 0.65], np.float32)
        desk = desk + rng.normal(0, 10, (H, W, 1))
        m = cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.uniform(-2, 2)), 1.0)
        m[:, 2] += [(W - w) / 2, (H - h) / 2]
        page = cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        inside = cv2.warpAffine(np.ones((h, w), np.uint8), m, (W, H), flags=cv2.INTER_NEAREST) > 0
        out = np.where(inside[..., None], page.astype(np.float32), desk)
        gt_c = cv2.warpAffine(gt.astype(np.uint8), m, (W, H), flags=cv2.INTER_NEAREST) > 0
        dots_c = cv2.warpAffine(dots.astype(np.float32), m, (W, H), flags=cv2.INTER_NEAREST).astype(np.int32)
        return Sample(np.clip(out, 0, 255).astype(np.uint8), gt_c, dots_c, condition)
    if condition == "transparent":
        rgba = np.zeros((h, w, 4), np.uint8)
        rgba[..., 3] = np.clip(ink * pressure * 255, 0, 255).astype(np.uint8)
        return Sample(rgba, gt, dots, condition)
    if condition == "tilt":
        true = float(rng.uniform(2.0, 7.0) * (1 if rng.random() < 0.5 else -1))
        m = cv2.getRotationMatrix2D((w / 2, h / 2), true, 1.0)
        img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=(245, 245, 242))
        tilt = -true
    return Sample(img, gt, dots, condition, tilt_deg=tilt)


def _glyph_sample(rng: np.random.Generator, fonts: list[str]) -> Sample:
    """Mimic the letters52 crops, using statistics measured on 400 real crops:
    height 22-77 px, ink only 10-45% darker than the paper (median 27%), tinted paper, blur.
    """
    up = 4
    h = int(rng.integers(22, 64))
    w = int(h * rng.uniform(1.1, 1.8))
    H, W = h * up, w * up
    canvas = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(canvas)
    font = _font(fonts, int(H * 0.55), rng)
    letter = str(rng.choice(list(LETTERS)))
    box = d.textbbox((0, 0), letter, font=font)
    d.text(((W - (box[2] - box[0])) / 2 - box[0], (H - (box[3] - box[1])) / 2 - box[1]), letter,
           font=font, fill=255)
    dots = Image.new("L", (W, H), 0)
    r = max(4.0, H * 0.035)
    cx, cy = W / 2 + rng.uniform(-W / 8, W / 8), H * (0.15 if rng.random() < 0.5 else 0.85)
    for k in range(int(rng.integers(1, 4))):
        ImageDraw.Draw(dots).ellipse([cx + (k - 1) * r * 2.8 - r, cy - r, cx + (k - 1) * r * 2.8 + r, cy + r],
                                     fill=255)
    ink = np.maximum(np.asarray(canvas), np.asarray(dots)).astype(np.float32) / 255
    _, dot_labels = cv2.connectedComponents((np.asarray(dots) > 127).astype(np.uint8))
    paper = np.array(rng.integers(170, 235, 3), np.float32)
    depth = rng.uniform(0.10, 0.45)
    img = _compose(ink, paper, paper * (1 - depth), _smooth_field(rng, H, W, 3, 0.7, 1.0))
    img *= _smooth_field(rng, H, W, 2, 0.85, 1.0)[..., None]       # uneven lighting across the crop
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(1.5, 4.5))      # = 0.4-1.1 px blur at crop size
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    img = img + rng.normal(0, 3, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return Sample(img, ink > 0.5, dot_labels.astype(np.int32), "glyph_small", glyph_scale=up)


# ----------------------------------------------------------------------------- metrics

def score(pred_binary: np.ndarray, sample: Sample) -> dict:
    """Pixel F1 of ink and the fraction of ground-truth dots kept (>= 50% of their pixels inked)."""
    gt, dots = sample.gt_mask, sample.dot_mask
    if pred_binary.shape != gt.shape:
        pred_binary = cv2.resize(pred_binary, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    pred = pred_binary == 0
    tp = float((pred & gt).sum())
    precision = tp / max(1.0, float(pred.sum()))
    recall = tp / max(1.0, float(gt.sum()))
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    n_dots = int(dots.max())
    kept = sum(1 for i in range(1, n_dots + 1) if pred[dots == i].mean() >= 0.5)
    return {"f1": f1, "precision": precision, "recall": recall,
            "dot_recall": kept / n_dots if n_dots else 1.0, "n_dots": n_dots}
