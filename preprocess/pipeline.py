"""`preprocess_image()`: measure an image, then apply only the corrections it needs.

Order of operations (each one is a function in preprocess/ops.py):
  load + flatten alpha -> grayscale (best channel) -> profile (glyph / page) + rescale
  -> measure stats -> invert if dark -> denoise if noisy -> flatten illumination (always)
  -> CLAHE if low contrast -> binarise -> deskew if tilted (re-binarise) -> remove lines /
  border blobs (pages only) -> remove specks smaller than a dot.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from common.paths import load_config
from preprocess import ops


@dataclass
class PreprocessResult:
    image: np.ndarray                 # uint8, text = 0, background = 255
    report: dict = field(default_factory=dict)
    gray: np.ndarray | None = None    # the cleaned grayscale just BEFORE thresholding (same geometry);
                                      # keeps faint strokes/dots as soft grey (used by the letter CNN)


def measure_stats(gray: np.ndarray) -> dict:
    """Image statistics that decide which corrections run."""
    dark = ops.is_dark_background(gray)
    g = ops.invert(gray) if dark else gray
    small, _ = ops.resize_max_side(g, 800)
    sw = ops.estimate_stroke_width(ops.local_ink_mask(small))
    bg = ops.quick_background(small, ops.odd(max(sw * 4, max(small.shape) * 0.03), 7))
    p5, p95 = np.percentile(bg, [5, 95])
    return {
        "height": int(gray.shape[0]),
        "width": int(gray.shape[1]),
        "median": float(np.median(gray)),
        "contrast": round(ops.contrast(g), 3),
        "dark_background": bool(dark),
        "noise_sigma": round(ops.noise_sigma(g), 2),
        "uneven_illumination": round(float((p95 - p5) / max(p95, 1.0)), 3),
    }


def _window(cfg_block: dict, sw: float) -> int:
    w = sw * cfg_block["window_sw_factor"]
    return ops.odd(min(max(w, cfg_block["window_min"]), cfg_block["window_max"]))


def _binarize(gray: np.ndarray, method: str, sw: float, cfg: dict, k: float, pcfg: dict) -> np.ndarray:
    if method == "flatten_normalized_fixed":
        # ink (1st percentile) -> 0, paper (median) -> 255, then cut at a fixed level
        norm = ops.normalize_contrast(gray)
        return np.where(norm > cfg["normalized_fixed"]["threshold"], 255, 0).astype(np.uint8)
    if method == "flatten_sauvola":
        return ops.binarize(gray, "sauvola", window=_window(cfg["sauvola"], sw), k=k)
    if method == "flatten_otsu":
        return ops.binarize(gray, "otsu")
    if method == "flatten_adaptive":
        return ops.binarize(gray, "adaptive", window=_window(cfg["adaptive"], sw), c=cfg["adaptive"]["c"])
    raise ValueError(f"unknown method {method}")


def preprocess_image(src, cfg: dict | None = None, profile: str = "auto", method: str | None = None,
                     overrides: dict | None = None) -> PreprocessResult:
    """Clean one image. `src` is a path or an RGB/RGBA/gray uint8 array.

    profile: "glyph" (single-letter crop), "page" (lines of text) or "auto" (by size).
    overrides: per-call profile settings, e.g. {"deskew": False, "max_side": None}.
    """
    t0 = time.perf_counter()
    cfg = cfg or load_config("step1")
    steps: list[str] = []

    rgb = ops.load_rgb(src) if isinstance(src, (str, Path)) else ops.flatten_alpha(np.asarray(src))
    gray, channel = ops.to_gray(rgb, cfg["gray_mode"])
    steps.append(f"gray:{channel}")

    if profile == "auto":
        profile = "glyph" if max(gray.shape) <= cfg["glyph_max_side"] else "page"
    pcfg = {**cfg[profile], **(overrides or {})}
    method = method or pcfg.get("method", cfg["method"])

    scale = 1.0
    if profile == "glyph":
        scale = min(cfg["glyph"]["max_upscale"], max(1.0, cfg["glyph"]["target_height"] / gray.shape[0]))
        if scale > 1:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            steps.append(f"upscale:{scale:.2f}")
    elif pcfg.get("max_side"):
        gray, scale = ops.resize_max_side(gray, pcfg["max_side"])
        if scale < 1:
            steps.append(f"downscale:{scale:.3f}")

    stats = measure_stats(gray)
    if pcfg.get("polarity", "auto") == "dark_ink":
        # Known dark ink on paper: skip detection (on blurry, faint crops it misfires).
        stats["dark_background"] = False
    th = cfg["thresholds"]

    if stats["dark_background"]:
        gray = ops.invert(gray)
        steps.append("invert")
    paper = None
    if pcfg.get("remove_desk") and not stats["dark_background"]:
        paper = ops.find_paper_mask(gray)
        if paper is not None:
            steps.append(f"paper_mask:{paper.mean():.2f}")
    if stats["noise_sigma"] > th["noise_sigma"]:
        gray = ops.denoise(gray, stats["noise_sigma"])
        steps.append("denoise")

    if pcfg.get("pre_blur_sigma"):
        # upscaled JPEG crops have blocky edges and grain; a light blur stabilises the threshold
        gray = cv2.GaussianBlur(gray, (0, 0), pcfg["pre_blur_sigma"])
        steps.append("blur")

    # Stroke width from a shadow-robust rough mask; it sizes every kernel below.
    sw = ops.estimate_stroke_width(ops.local_ink_mask(gray))
    bcfg = cfg["background"]
    min_frac = pcfg.get("bg_kernel_min_frac", bcfg["kernel_min_frac"])
    kernel = ops.odd(max(sw * bcfg["kernel_sw_factor"], max(gray.shape) * min_frac), 7)
    flat = ops.flatten_illumination(gray, ops.estimate_background(gray, kernel))
    steps.append("flatten")
    if pcfg.get("clahe") == "always" or stats["contrast"] < th["low_contrast"]:
        flat = ops.apply_clahe(flat, cfg["clahe"]["clip_limit"], cfg["clahe"]["tile"])
        steps.append("clahe")

    k = pcfg.get("sauvola_k", cfg["sauvola"]["k"])
    binary = _binarize(flat, method, sw, cfg, k, pcfg)
    if method == "flatten_normalized_fixed":
        flat = ops.normalize_contrast(flat)  # the gray output gets the same ink/paper scaling
    if pcfg.get("remove_edge_bars"):
        bars = ops.find_edge_bars(binary)
        if bars.any():
            # A dark paper edge would set the "ink" level for the whole crop: paint it as paper
            # and binarise again so the faint letter is judged on its own.
            flat = flat.copy()
            flat[cv2.dilate(bars.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = 255 if \
                method == "flatten_normalized_fixed" else int(np.median(flat))
            binary = _binarize(flat, method, sw, cfg, k, pcfg)
            steps.append("remove_edge_bar")
    angle = 0.0
    valid = None  # where real image pixels are (after deskew the canvas gains a white margin)
    if pcfg["deskew"]:
        angle = ops.estimate_skew(binary, th["deskew_max_angle"])
        if abs(angle) >= th["deskew_min_angle"]:
            valid = ops.rotate(np.full(flat.shape, 255, np.uint8), angle, fill=0,
                               interp=cv2.INTER_NEAREST) > 0
            if paper is not None:
                paper = ops.rotate(paper.astype(np.uint8) * 255, angle, fill=0, interp=cv2.INTER_NEAREST) > 0
            flat = ops.rotate(flat, angle)
            binary = _binarize(flat, method, sw, cfg, k, pcfg)
            steps.append(f"deskew:{angle:+.2f}")
        else:
            angle = 0.0

    if paper is not None:
        binary[~paper] = 255  # everything outside the sheet is background
        valid = paper if valid is None else (valid & paper)
    sw_bin = ops.estimate_stroke_width(binary == 0)
    if pcfg["remove_lines"]:
        binary, info = ops.remove_lines(binary, sw_bin, cfg["lines"]["h_frac"], cfg["lines"]["v_frac"])
        if info["line_pixels"]:
            steps.append(f"remove_lines:{info['line_pixels']}px")
        binary, n = ops.remove_line_fragments(binary, sw_bin)
        if n:
            steps.append(f"remove_line_fragments:{n}")
    if pcfg["remove_border"]:
        binary, n = ops.remove_border_artifacts(binary, cfg["border"]["area_frac"], cfg["border"]["span_frac"],
                                                valid)
        if n:
            steps.append(f"remove_border:{n}")
    min_area = max(2, int(pcfg["min_dot_area_sw2"] * sw_bin * sw_bin))
    binary, n = ops.remove_small_components(binary, min_area)
    if n:
        steps.append(f"remove_specks:{n}(<{min_area}px)")

    report = {
        "profile": profile,
        "method": method,
        "scale": round(scale, 4),
        "deskew_angle": angle,
        "stroke_width": round(sw_bin, 2),
        "steps": steps,
        "stats": stats,
        "seconds": round(time.perf_counter() - t0, 3),
    }
    gray = flat.copy()
    if paper is not None:
        gray[~paper] = 255
    return PreprocessResult(binary, report, gray)
