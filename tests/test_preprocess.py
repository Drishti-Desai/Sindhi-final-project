"""Unit tests for Step 1 preprocessing.

Run: .venv/bin/python -m pytest tests/test_preprocess.py -q

Each degradation from the brief (transparent, dark, shadowed, low-contrast, tilted, noisy) is
tested on synthetic images with known ground truth, plus tests for single operations.
"""
import numpy as np
import pytest

from common.paths import load_config
from preprocess import ops, preprocess_image
from preprocess.synthetic import make_sample, score

CFG = load_config("step1")
FONTS = CFG["benchmark"]["fonts"]


def sample(condition, seed=0):
    return make_sample(condition, np.random.default_rng(seed), FONTS)


def run(s, **kw):
    overrides = kw.pop("overrides", {"deskew": False})
    return preprocess_image(s.image, CFG, overrides=overrides, **kw)


def assert_clean_binary(img):
    assert img.dtype == np.uint8 and img.ndim == 2
    assert set(np.unique(img)) <= {0, 255}
    assert (img == 255).mean() > 0.6, "background should be (mostly) pure white"


# ----------------------------------------------------------------------------- conditions

@pytest.mark.parametrize("seed", [0, 1])
def test_transparent_png_is_flattened_onto_white(seed):
    s = sample("transparent", seed)
    assert s.image.shape[2] == 4 and s.image[..., :3].max() == 0  # text only in the alpha channel
    r = run(s)
    assert_clean_binary(r.image)
    assert score(r.image, s)["f1"] > 0.9


@pytest.mark.parametrize("seed", [0, 1])
def test_dark_background_is_inverted(seed):
    s = sample("dark", seed)
    r = run(s)
    assert r.report["stats"]["dark_background"] and "invert" in r.report["steps"]
    assert_clean_binary(r.image)
    assert score(r.image, s)["f1"] > 0.9


@pytest.mark.parametrize("seed", [0, 1])
def test_shadow_is_removed(seed):
    s = sample("shadow", seed)
    r = run(s)
    assert not r.report["stats"]["dark_background"], "a shadow must not be mistaken for a dark page"
    m = score(r.image, s)
    assert m["f1"] > 0.9 and m["dot_recall"] >= 0.9


@pytest.mark.parametrize("seed", [0, 1])
def test_low_contrast_triggers_clahe(seed):
    s = sample("low_contrast", seed)
    r = run(s)
    assert "clahe" in r.report["steps"]
    assert score(r.image, s)["f1"] > 0.85


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tilt_is_detected_and_corrected(seed):
    s = sample("tilt", seed)
    r = run(s, overrides={})
    assert abs(r.report["deskew_angle"] - s.tilt_deg) < 0.5
    assert any(step.startswith("deskew") for step in r.report["steps"])


def test_straight_page_is_not_rotated():
    r = run(sample("clean"), overrides={})
    assert r.report["deskew_angle"] == 0.0


@pytest.mark.parametrize("seed", [0, 1])
def test_noise_is_denoised(seed):
    s = sample("noise", seed)
    r = run(s)
    assert "denoise" in r.report["steps"]
    m = score(r.image, s)
    assert m["f1"] > 0.9 and m["precision"] > 0.85  # salt-and-pepper must not survive as ink


def test_clean_white_image_stays_clean():
    s = sample("clean")
    r = run(s)
    assert "invert" not in r.report["steps"] and "denoise" not in r.report["steps"]
    assert score(r.image, s)["f1"] > 0.93


@pytest.mark.parametrize("seed", [0, 1])
def test_desk_around_paper_is_removed(seed):
    s = sample("desk", seed)
    r = run(s)
    assert any(step.startswith("paper_mask") for step in r.report["steps"])
    m = score(r.image, s)
    assert m["f1"] > 0.9 and m["precision"] > 0.85  # desk left in gives ~0.4; 1-px stroke rims cost a little


def test_full_frame_page_has_no_paper_mask():
    r = run(sample("clean"))
    assert not any(step.startswith("paper_mask") for step in r.report["steps"])


def test_ruled_lines_removed_text_kept():
    s = sample("ruled")
    r = run(s)
    assert any(step.startswith("remove_lines") for step in r.report["steps"])
    m = score(r.image, s)
    assert m["f1"] > 0.9 and m["dot_recall"] >= 0.9


def test_small_letter_crops_keep_dots():
    # Aggregate over 12 crops: single crops can be ~10% ink contrast (barely visible), so a
    # per-crop assertion would test noise. The Step 1 gate uses the same idea on 60 crops.
    recalls = []
    for seed in range(12):
        s = sample("glyph_small", seed)
        r = preprocess_image(s.image, CFG)  # auto profile must pick "glyph" for a tiny crop
        assert r.report["profile"] == "glyph"
        assert_clean_binary(r.image)
        recalls.append(score(r.image, s)["dot_recall"])
    assert np.mean(recalls) >= 0.85


# ----------------------------------------------------------------------------- single operations

def test_flatten_alpha_rgba_array():
    rgba = np.zeros((10, 10, 4), np.uint8)
    rgba[2:4, 2:4, 3] = 255  # opaque black square on a transparent background
    rgb = ops.flatten_alpha(rgba)
    assert rgb[0, 0].tolist() == [255, 255, 255] and rgb[3, 3].tolist() == [0, 0, 0]


def test_flatten_illumination_removes_gradient():
    gray = np.tile(np.linspace(90, 250, 400, dtype=np.float32), (200, 1)).astype(np.uint8)
    gray[90:110, 50:350] = (gray[90:110, 50:350] * 0.3).astype(np.uint8)  # a dark stroke
    flat = ops.flatten_illumination(gray, ops.estimate_background(gray, 61))
    paper = np.concatenate([flat[:60].ravel(), flat[140:].ravel()])
    assert paper.std() < 6 and paper.mean() > 240
    assert flat[95:105, 100:300].mean() < 120


def test_binarize_methods_return_binary():
    # A pen-width stroke (6 px). Local methods are sized to ~12x stroke width; a solid block
    # wider than the window would be hollowed out by any local threshold, by design.
    gray = np.full((80, 120), 230, np.uint8)
    gray[37:43, 20:100] = 40
    for method in ("otsu", "sauvola", "adaptive"):
        b = ops.binarize(gray, method, window=71)
        assert set(np.unique(b)) <= {0, 255}
        assert (b[37:43, 25:95] == 0).mean() > 0.95
        assert (b[:30] == 255).all()


def test_remove_small_components_keeps_dot_sized_blobs():
    b = np.full((60, 60), 255, np.uint8)
    b[10, 10] = 0                       # 1-pixel speck
    b[30:34, 30:34] = 0                 # 16-pixel "dot" (stroke width ~4)
    sw = 4.0
    out, n = ops.remove_small_components(b, max(2, int(CFG["page"]["min_dot_area_sw2"] * sw * sw)))
    assert n == 1 and out[10, 10] == 255 and (out[30:34, 30:34] == 0).all()


def test_remove_lines_keeps_crossing_stroke():
    b = np.full((200, 600), 255, np.uint8)
    b[100:103, :] = 0                   # ruled line across the page
    b[70:130, 300:304] = 0              # a vertical letter stroke crossing it
    out, info = ops.remove_lines(b, stroke_width=3.0)
    assert info["line_pixels"] > 0
    assert (out[100:103, 20:280] == 255).all(), "line should be gone"
    assert (out[70:130, 300:304] == 0).mean() > 0.95, "stroke should be intact"


def test_line_fragments_removed_but_alif_kept():
    b = np.full((300, 800), 255, np.uint8)
    for x in range(0, 800, 120):        # faint ruled line broken into 90-px dashes
        b[200:202, x:x + 90] = 0
    b[100:160, 400:405] = 0             # alif: 60 px tall, 5 px wide, isolated
    b[150:155, 300:370] = 0             # a short flat stroke (kashida-like), 70 px
    out, n = ops.remove_line_fragments(b, stroke_width=4.0)
    assert (out[200:202] == 255).all(), "dashes should be removed"
    assert (out[100:160, 400:405] == 0).all(), "alif must stay"
    assert (out[150:155, 300:370] == 0).all(), "short strokes (< 10% of width) must stay"


def test_border_artifacts_after_deskew_margin():
    b = np.full((300, 600), 255, np.uint8)
    valid = np.ones((300, 600), bool)
    valid[:, :20] = False               # white margin added by deskew rotation
    b[50:200, 22:40] = 0                # finger / binder bar at the edge of the real photo
    b[140:150, 300:420] = 0             # a word-like stroke in the middle
    b[120:160, 560:599] = 0             # a letter touching the right border...
    b[125:155, 565:594] = 255           # ...hollow, i.e. sparse like real handwriting
    out, n = ops.remove_border_artifacts(b, valid=valid)
    assert (out[50:200, 22:40] == 255).all(), "edge bar should be removed"
    assert (out[140:150, 300:420] == 0).all(), "text in the middle must stay"
    assert (out[120:160, 560:599] == 0).sum() > 0, "sparse letter at the border must stay"


def test_find_edge_bars_ignores_centered_alif():
    b = np.full((60, 40), 255, np.uint8)
    b[:, 18:22] = 0                     # alif spanning the whole height, centred
    assert not ops.find_edge_bars(b).any()
    b[:, 36:40] = 0                     # paper edge along the right side
    assert ops.find_edge_bars(b)[:, 37].all()


def test_estimate_skew_sign_matches_rotate():
    s = sample("tilt", 5)
    gray = ops.to_gray(s.image)[0]
    angle = ops.estimate_skew(ops.binarize(gray, "otsu"))
    assert abs(angle - s.tilt_deg) < 0.5
