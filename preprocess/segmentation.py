"""Option (b) of the comparison: neural background removal with rembg (U^2-Net), then Otsu.

rembg is built to cut a salient *object* (a person, a product) out of a photo. It is included
as the brief asked, and to show with evidence whether it suits text on paper.
"""
from __future__ import annotations

import os
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

from common.paths import load_paths
from preprocess import ops


@lru_cache(maxsize=2)
def _session(model: str):
    # Keep model weights inside the repo (checkpoints/rembg) instead of ~/.u2net.
    os.environ.setdefault("U2NET_HOME", str(load_paths()["checkpoints"]["rembg"]))
    from rembg import new_session  # imported lazily: slow import, optional dependency
    return new_session(model)


def rembg_otsu(rgb: np.ndarray, model: str = "u2net") -> np.ndarray:
    """Remove the background with rembg, flatten onto white, then Otsu. Text = 0."""
    from rembg import remove
    rgba = np.asarray(remove(Image.fromarray(rgb), session=_session(model)))
    gray = cv2.cvtColor(ops.flatten_alpha(rgba), cv2.COLOR_RGB2GRAY)
    if ops.is_dark_background(gray):
        gray = ops.invert(gray)
    return ops.binarize(gray, "otsu")


def otsu_raw(rgb: np.ndarray) -> np.ndarray:
    """Baseline used by the old project: grayscale + global Otsu, nothing else."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if ops.is_dark_background(gray):
        gray = ops.invert(gray)
    return ops.binarize(gray, "otsu")
