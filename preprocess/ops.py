"""Step 1 building blocks: one small, independently testable function per operation.

Conventions used throughout:
- Grayscale images are uint8 HxW arrays: 0 = black, 255 = white.
- Binary images use the same convention: text = 0, background = 255.
- "Masks" are bool arrays where True = ink.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.filters import threshold_sauvola


def odd(n: float, minimum: int = 3) -> int:
    """Round to the nearest odd integer >= minimum (OpenCV kernels must be odd)."""
    n = max(int(round(n)), minimum)
    return n if n % 2 == 1 else n + 1


def resize_max_side(img: np.ndarray, max_side: int, interp=cv2.INTER_AREA) -> tuple[np.ndarray, float]:
    """Downscale so the longer side is <= max_side. Returns (image, scale factor applied)."""
    h, w = img.shape[:2]
    f = max_side / max(h, w)
    if f >= 1:
        return img, 1.0
    return cv2.resize(img, (max(1, round(w * f)), max(1, round(h * f))), interpolation=interp), f


# ----------------------------------------------------------------------------- loading

def flatten_alpha(im: Image.Image | np.ndarray) -> np.ndarray:
    """Composite a possibly transparent image onto pure white and return RGB uint8.

    Transparent PNGs often store black RGB everywhere and put the text only in the alpha
    channel, so simply dropping alpha would give an all-black image.
    """
    if isinstance(im, np.ndarray):
        if im.ndim == 2:
            return np.stack([im] * 3, axis=-1)
        if im.shape[2] == 4:
            rgb = im[..., :3].astype(np.float32)
            a = im[..., 3:4].astype(np.float32) / 255.0
            return np.round(rgb * a + 255.0 * (1.0 - a)).astype(np.uint8)
        return im[..., :3]
    has_alpha = im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)
    if has_alpha:
        return flatten_alpha(np.asarray(im.convert("RGBA")))
    return np.asarray(im.convert("RGB"))


def load_rgb(path) -> np.ndarray:
    """Read any image file as RGB uint8 (EXIF rotation applied, alpha flattened onto white)."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return flatten_alpha(im)


# ----------------------------------------------------------------------------- measurements

def quick_background(gray: np.ndarray, kernel: int) -> np.ndarray:
    """Local paper brightness: a closing (local max) erases dark strokes thinner than `kernel`."""
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, se)
    return cv2.GaussianBlur(bg, (0, 0), kernel / 2)


def ink_separability(gray: np.ndarray) -> float:
    """How cleanly ink separates from paper, in [0, 1] (Otsu's eta on an illumination-flattened copy).

    Flattening first means a shadow edge does not count as "separation".
    """
    small, _ = resize_max_side(gray, 512)
    flat = cv2.divide(small, np.maximum(quick_background(small, odd(max(small.shape) / 20, 7)), 1),
                      scale=255)
    total_var = flat.var()
    if total_var < 1e-6:
        return 0.0
    t, _ = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lo, hi = flat[flat <= t], flat[flat > t]
    if lo.size == 0 or hi.size == 0:
        return 0.0
    w0, w1 = lo.size / flat.size, hi.size / flat.size
    return float(w0 * w1 * (lo.mean() - hi.mean()) ** 2 / total_var)


def to_gray(rgb: np.ndarray, mode: str = "auto", margin: float = 0.05) -> tuple[np.ndarray, str]:
    """Convert to grayscale. In "auto" mode pick the channel where ink separates best.

    Example: blue ink on purple paper is much darker than the paper in the red channel.
    Luminance is kept unless another channel is better by `margin`, to avoid noise-driven switches.
    """
    if rgb.ndim == 2:
        return rgb, "gray"
    lum = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if mode == "luminance":
        return lum, "luminance"
    candidates = {"luminance": lum, "min_channel": rgb.min(axis=2),
                  "red": rgb[..., 0], "green": rgb[..., 1], "blue": rgb[..., 2]}
    scores = {k: ink_separability(np.ascontiguousarray(v)) for k, v in candidates.items()}
    best = max(scores, key=scores.get)
    if scores[best] < scores["luminance"] + margin:
        best = "luminance"
    return np.ascontiguousarray(candidates[best]), best


def is_dark_background(gray: np.ndarray) -> bool:
    """True if the text is lighter than its surroundings (e.g. chalk on a blackboard, dark mode).

    Looks at the most extreme local deviations from a median-filtered copy: those pixels are text.
    Using local deviations (not global brightness) keeps a heavily shadowed page from being
    mistaken for a dark background.
    """
    small, _ = resize_max_side(gray, 512)
    local = cv2.medianBlur(small, odd(max(small.shape) / 15, 5))
    r = small.astype(np.float32) - local.astype(np.float32)
    thr = np.percentile(np.abs(r), 99)
    if thr < 2:  # almost no structure at all: fall back to global brightness
        return float(np.median(small)) < 128
    strong = r[np.abs(r) >= thr]
    frac_bright = float((strong > 0).mean())
    if frac_bright > 0.6:
        return True
    if frac_bright < 0.4:
        return False
    return float(np.median(small)) < 128


_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def noise_sigma(gray: np.ndarray, crop: int = 512) -> float:
    """Estimated Gaussian noise std on a central crop, 0-255 scale (Immerkaer 1996).

    The 3x3 kernel cancels smooth image structure, leaving mostly noise. Text edges still add
    a little, so clean scans read ~1-4 and the threshold in the config sits well above that.
    """
    h, w = gray.shape
    y0, x0 = max(0, (h - crop) // 2), max(0, (w - crop) // 2)
    patch = gray[y0:y0 + crop, x0:x0 + crop].astype(np.float32)
    if min(patch.shape) < 3:
        return 0.0
    resp = cv2.filter2D(patch, -1, _IMMERKAER)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(resp).mean() / 6.0)


def contrast(gray: np.ndarray) -> float:
    """Robust contrast in [0, 1]: spread between the 1st and 99th percentile."""
    p1, p99 = np.percentile(gray, [1, 99])
    return float((p99 - p1) / 255.0)


def estimate_stroke_width(text_mask: np.ndarray) -> float:
    """Typical pen width in pixels: twice the distance-transform value along stroke centre lines."""
    if text_mask.sum() < 20:
        return 3.0
    dt = cv2.distanceTransform(text_mask.astype(np.uint8), cv2.DIST_L2, 3)
    ridge = (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8))) & text_mask
    return float(max(1.0, np.median(dt[ridge]) * 2.0))


def local_ink_mask(gray: np.ndarray) -> np.ndarray:
    """Rough ink mask that ignores shadows: pixels clearly darker than their local median."""
    local = cv2.medianBlur(gray, odd(max(gray.shape) / 15, 5))
    r = local.astype(np.int16) - gray.astype(np.int16)
    thr = max(10.0, float(np.percentile(r, 97)) * 0.5)
    return r > thr


# ----------------------------------------------------------------------------- corrections

def invert(gray: np.ndarray) -> np.ndarray:
    return 255 - gray


def denoise(gray: np.ndarray, sigma: float) -> np.ndarray:
    """Non-local means: removes grain while keeping edges (and dots) better than a median filter."""
    h = float(np.clip(sigma, 3.0, 30.0))
    return cv2.fastNlMeansDenoising(gray, None, h=h, templateWindowSize=7, searchWindowSize=21)


def estimate_background(gray: np.ndarray, kernel: int, max_work_side: int = 1000) -> np.ndarray:
    """Paper brightness map (shadows, gradients, paper tint), computed at reduced resolution.

    `kernel` must be wider than the pen stroke so the closing removes the text completely.
    """
    small, f = resize_max_side(gray, max_work_side)
    k = odd(kernel * f, 5)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.medianBlur(bg, odd(k, 5) if k <= 255 else 255)
    bg = cv2.GaussianBlur(bg, (0, 0), max(1.0, k / 3))
    if f < 1:
        bg = cv2.resize(bg, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    return bg


def flatten_illumination(gray: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Divide by the background map: paper becomes ~255 everywhere, ink stays dark."""
    return cv2.divide(gray, np.maximum(background, 1), scale=255)


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(gray)


def normalize_contrast(gray: np.ndarray, low_pct: float = 1.0) -> np.ndarray:
    """Linear stretch: darkest ink (low_pct percentile) -> 0, median paper -> 255.

    Used on single-letter crops, where the ink can be only 10-30% darker than the paper.
    Not used on pages: there it also stretches paper texture into speckles.
    """
    lo, paper = np.percentile(gray, [low_pct, 50])
    if paper - lo < 4:  # blank crop: nothing to stretch
        return gray
    out = (gray.astype(np.float32) - lo) * (255.0 / (paper - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- binarisation

def binarize(gray: np.ndarray, method: str = "sauvola", window: int = 31, k: float = 0.2,
             c: float = 10.0) -> np.ndarray:
    """Return a binary image (text = 0, background = 255).

    otsu      global threshold; good once illumination is flat
    sauvola   local mean/std threshold; keeps faint strokes (pencil, light pen pressure)
    adaptive  OpenCV Gaussian-weighted local threshold
    """
    if method == "otsu":
        _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return b
    if method == "sauvola":
        t = threshold_sauvola(gray, window_size=odd(window), k=k)
        return np.where(gray > t, 255, 0).astype(np.uint8)
    if method == "adaptive":
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                     odd(window), c)
    raise ValueError(f"unknown binarisation method: {method}")


# ----------------------------------------------------------------------------- clean-up

def remove_small_components(binary: np.ndarray, min_area: int) -> tuple[np.ndarray, int]:
    """Delete ink specks smaller than `min_area` pixels. Returns (image, number removed).

    `min_area` should come from the stroke width so that diacritic dots (about one stroke
    width across) are always larger than it.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats((binary == 0).astype(np.uint8), 8)
    small = np.where(stats[1:, cv2.CC_STAT_AREA] < min_area)[0] + 1
    if small.size == 0:
        return binary, 0
    out = binary.copy()
    out[np.isin(labels, small)] = 255
    return out, int(small.size)


def remove_lines(binary: np.ndarray, stroke_width: float, h_frac: float = 0.5,
                 v_frac: float = 0.5) -> tuple[np.ndarray, dict]:
    """Remove ruled-paper lines and drawn frames, then repair strokes that crossed them.

    A horizontal line must be a straight run longer than h_frac of the image width; handwriting
    connections are far shorter. Vertical lines must span v_frac of the height, much taller than
    any alif. Do NOT use this on single-letter crops, where a letter can span the whole crop.
    """
    ink = (binary == 0).astype(np.uint8)
    h, w = ink.shape
    bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (odd(max(3, stroke_width * 3)), 1))
    h_lines = cv2.morphologyEx(cv2.morphologyEx(ink, cv2.MORPH_CLOSE, bridge), cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, int(w * h_frac)), 1)))
    v_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (1, odd(max(3, stroke_width * 3))))
    v_lines = cv2.morphologyEx(cv2.morphologyEx(ink, cv2.MORPH_CLOSE, v_bridge), cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, int(h * v_frac)))))
    is_h, is_v, is_ink = h_lines > 0, v_lines > 0, ink > 0
    lines = (is_h | is_v) & is_ink
    n_px = int(lines.sum())
    if n_px == 0:
        return binary, {"line_pixels": 0}
    kept = (is_ink & ~lines).astype(np.uint8)
    # A stroke crossing a line loses a short piece. Restore a horizontal-line pixel when ink
    # continues directly above AND below it (vertical closing), and vice versa for vertical lines.
    k = odd(stroke_width * 2 + 3)
    v_close = cv2.morphologyEx(kept, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))) > 0
    h_close = cv2.morphologyEx(kept, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))) > 0
    restored = lines & ((is_h & v_close) | (is_v & h_close))
    out = np.where((kept > 0) | restored, 0, 255).astype(np.uint8)
    return out, {"line_pixels": n_px, "restored_pixels": int(restored.sum())}


def remove_line_fragments(binary: np.ndarray, stroke_width: float, min_len_h: float = 0.1,
                          min_len_v: float = 0.5, max_thick_sw: float = 2.5,
                          min_aspect: float = 12.0) -> tuple[np.ndarray, int]:
    """Delete isolated, thin, straight pieces of ruled lines (faint lines break into dashes).

    A piece qualifies only if it is a separate component, at most `max_thick_sw` stroke widths
    thick, at least `min_aspect` times longer than thick, and long: >= min_len_h of the width
    (horizontal) or >= min_len_v of the height (vertical; an alif is far shorter than half
    a page). Strokes attached to letters are never touched.
    """
    h, w = binary.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats((binary == 0).astype(np.uint8), 8)
    drop = []
    max_thick = max(3.0, max_thick_sw * stroke_width)
    for i in range(1, n):
        _, _, bw, bh, _ = stats[i]
        horizontal = bh <= max_thick and bw >= min_aspect * bh and bw >= min_len_h * w
        vertical = bw <= max_thick and bh >= min_aspect * bw and bh >= min_len_v * h
        if horizontal or vertical:
            drop.append(i)
    if not drop:
        return binary, 0
    out = binary.copy()
    out[np.isin(labels, drop)] = 255
    return out, len(drop)


def remove_border_artifacts(binary: np.ndarray, area_frac: float = 0.01, span_frac: float = 0.6,
                            valid: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Delete dark blobs at the photo's edge (desk, fingers, paper edges, binder shadows).

    "Edge" means the image border or, after deskew, the edge of the rotated original picture
    (`valid` = True where real image pixels are). A blob at the edge is removed if it is large,
    spans most of a side, or is a solid elongated bar. Text at the edge is small and sparse
    (fill ratio ~0.2-0.3), so it is kept.
    """
    h, w = binary.shape
    edge = np.zeros((h, w), np.uint8)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = 1
    if valid is not None:
        edge |= cv2.dilate((~valid).astype(np.uint8), np.ones((7, 7), np.uint8))  # 3-px seam tolerance
    n, labels, stats, _ = cv2.connectedComponentsWithStats((binary == 0).astype(np.uint8), 8)
    touching = np.unique(labels[(edge > 0) & (labels > 0)])
    drop = []
    for i in touching:
        x, y, bw, bh, area = stats[i]
        long_side, short_side = max(bw, bh), max(1, min(bw, bh))
        solid_bar = area >= 0.5 * bw * bh and long_side >= 4 * short_side and \
            (bh >= 0.25 * h if bh >= bw else bw >= 0.25 * w)
        if area > area_frac * h * w or bw > span_frac * w or bh > span_frac * h or solid_bar:
            drop.append(i)
    if not drop:
        return binary, 0
    out = binary.copy()
    out[np.isin(labels, drop)] = 255
    return out, len(drop)


def find_edge_bars(binary: np.ndarray, span: float = 0.9, fill: float = 0.6) -> np.ndarray:
    """Mask of solid bars along a crop's side (paper edge / neighbouring sheet in letter crops).

    Only vertical bars touching the LEFT/RIGHT edge or horizontal bars touching the TOP/BOTTOM
    edge, spanning >= `span` of that side and mostly solid. A centred alif or a curved letter
    stroke does not qualify.
    """
    h, w = binary.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats((binary == 0).astype(np.uint8), 8)
    bars = np.zeros((h, w), bool)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        solid = area >= fill * bw * bh
        vertical = (x == 0 or x + bw >= w) and bh >= span * h and bw < 0.35 * w
        horizontal = (y == 0 or y + bh >= h) and bw >= span * w and bh < 0.35 * h
        if solid and (vertical or horizontal):
            bars |= labels == i
    return bars


SHEET_EDGE_SHARPNESS = 0.40  # 5x5-max boundary gradient / brightness step: synthetic sheets 0.47-0.50, shadows <= 0.34


def find_paper_mask(gray: np.ndarray, min_frac: float = 0.25, max_frac: float = 0.97,
                    margin_frac: float = 0.012) -> np.ndarray | None:
    """Mask of the paper sheet when the photo also shows the surface around it (desk, table).

    Returns None (= do nothing) unless the image border is clearly darker than the largest
    bright, roughly convex region. The mask is the filled convex hull of that region, shrunk
    by `margin_frac` of the shorter side so the shaded paper edge is excluded too.
    """
    small, f = resize_max_side(gray, 600)
    blur = cv2.GaussianBlur(small, (0, 0), 5)
    _, bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, st, _ = cv2.connectedComponentsWithStats((bright > 0).astype(np.uint8), 8)
    if n < 2:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    region = (lab == i).astype(np.uint8)
    frac = region.mean()
    if not (min_frac <= frac <= max_frac):
        return None
    cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    if region.sum() < 0.85 * cv2.contourArea(hull):  # not sheet-like (e.g. a bright blob)
        return None
    band = np.zeros_like(region)
    b = max(2, int(0.03 * min(small.shape)))
    band[:b, :] = band[-b:, :] = band[:, :b] = band[:, -b:] = 1
    outside = band.astype(bool) & (region == 0)
    if outside.mean() < 0.3 * band.mean():  # the border is mostly paper: nothing to remove
        return None
    diff = float(np.median(blur[region > 0]) - np.median(blur[outside]))
    if diff < 25:  # surface not clearly darker
        return None
    # A sheet's edge is a sharp brightness step; a shadow on the page fades gradually.
    # Edge sharpness = brightness gradient across the region boundary / brightness difference.
    # median filter: erases thin text strokes but keeps a step edge sharp (a soft shadow stays soft)
    fine = cv2.medianBlur(small, 11).astype(np.float32)
    grad = np.hypot(cv2.Sobel(fine, cv2.CV_32F, 1, 0), cv2.Sobel(fine, cv2.CV_32F, 0, 1)) / 8.0
    ring = (cv2.dilate(region, np.ones((5, 5), np.uint8)) - cv2.erode(region, np.ones((5, 5), np.uint8))) > 0
    ring &= ~cv2.dilate(band, np.ones((3, 3), np.uint8)).astype(bool)  # ignore the image frame itself
    # strongest gradient within 5x5 of each ring pixel: a 1-2 px wide real edge must not be
    # averaged away by ring pixels that sit just beside it
    grad = cv2.dilate(grad, np.ones((5, 5), np.uint8))
    if ring.sum() < 20 or np.median(grad[ring]) / diff < SHEET_EDGE_SHARPNESS:
        return None
    mask = np.zeros_like(region)
    cv2.fillPoly(mask, [hull], 1)
    m = max(1, int(margin_frac * min(small.shape)))
    mask = cv2.erode(mask, np.ones((2 * m + 1, 2 * m + 1), np.uint8))
    return cv2.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST) > 0


# ----------------------------------------------------------------------------- deskew

def _projection_score(mask: np.ndarray, angle: float) -> float:
    h, w = mask.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rot = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    rows = rot.sum(axis=1, dtype=np.float64)
    return float(np.sum(np.diff(rows) ** 2))  # sharp row profile = text lines are horizontal


def estimate_skew(binary: np.ndarray, max_angle: float = 10.0) -> float:
    """Angle (degrees) that makes text lines horizontal when passed to `rotate`.

    Projection-profile search: try angles, keep the one with the sharpest row histogram.
    Coarse 0.5 degree steps, then 0.1 degree refinement.
    """
    mask = (binary == 0).astype(np.uint8)
    if mask.sum() < 50:
        return 0.0
    small, _ = resize_max_side(mask * 255, 800)
    small = (small > 64).astype(np.uint8)
    coarse = np.arange(-max_angle, max_angle + 1e-6, 0.5)
    best = max(coarse, key=lambda a: _projection_score(small, a))
    fine = np.arange(best - 0.5, best + 0.5 + 1e-6, 0.1)
    return float(round(max(fine, key=lambda a: _projection_score(small, a)), 2))


def rotate(gray: np.ndarray, angle: float, fill: int = 255, interp=cv2.INTER_CUBIC) -> np.ndarray:
    """Rotate around the centre, enlarging the canvas so no text is cut off."""
    h, w = gray.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos + 0.5), int(h * cos + w * sin + 0.5)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(gray, m, (nw, nh), flags=interp, borderValue=fill)
