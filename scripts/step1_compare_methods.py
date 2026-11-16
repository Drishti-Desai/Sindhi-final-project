"""Step 1 evidence: compare background-removal / binarisation methods.

1. Synthetic benchmark (known ground truth): every condition from the brief, N samples each,
   scored by text-pixel F1 and diacritic-dot recall. Also checks that the automatic condition
   handling fires (invert on dark, denoise on noise, CLAHE on low contrast, deskew on tilt).
2. Real samples: a side-by-side grid of every method on real images (no FINAL_TEST images),
   for visual confirmation.

Outputs: results/step1/method_comparison.{json,md}, results/step1/comparison_real_*.png

Usage:
  python scripts/step1_compare_methods.py            # full run
  python scripts/step1_compare_methods.py --small    # 2 samples per condition, quick check
  python scripts/step1_compare_methods.py --no-rembg # skip option (b) (no model download)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import load_config, load_paths  # noqa: E402
from preprocess import ops, preprocess_image  # noqa: E402
from preprocess.segmentation import otsu_raw, rembg_otsu  # noqa: E402
from preprocess.synthetic import PAGE_CONDITIONS, make_sample, score  # noqa: E402

PIPELINE_METHODS = ["flatten_otsu", "flatten_adaptive", "flatten_sauvola", "flatten_normalized_fixed"]
EXPECTED_STEP = {"dark": "invert", "noise": "denoise", "low_contrast": "clahe", "pencil_faint": "clahe"}


def run_method(method: str, sample_img: np.ndarray, profile: str, cfg: dict) -> np.ndarray:
    rgb = ops.flatten_alpha(sample_img)
    if method == "otsu_raw":
        out = otsu_raw(rgb)
    elif method == "rembg_otsu":
        out = rembg_otsu(rgb)
    else:
        return preprocess_image(sample_img, cfg, profile=profile, method=method,
                                overrides={"deskew": False}).image
    if profile == "glyph":  # score at the same scale as the pipeline output
        s = cfg["glyph"]["target_height"] / out.shape[0]
        out = cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    return out


def oracle_f1(sample, profile: str, cfg: dict) -> float:
    """Best F1 any single global threshold could reach on the illumination-flattened image
    (threshold chosen WITH the ground truth). A ceiling, not a method."""
    gray, _ = ops.to_gray(ops.flatten_alpha(sample.image))
    if ops.is_dark_background(gray):
        gray = ops.invert(gray)
    frac = cfg["background"]["kernel_min_frac"]
    if profile == "glyph":
        s = cfg["glyph"]["target_height"] / gray.shape[0]
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        frac = cfg["glyph"]["bg_kernel_min_frac"]
    flat = ops.flatten_illumination(gray, ops.estimate_background(gray, ops.odd(max(gray.shape) * frac, 7)))
    return max(score(np.where(flat > t, 255, 0).astype(np.uint8), sample)["f1"] for t in range(60, 255, 4))


def synthetic_benchmark(cfg: dict, n: int, n_glyph: int, methods: list[str]) -> dict:
    fonts = cfg["benchmark"]["fonts"]
    rng = np.random.default_rng(cfg["benchmark"]["seed"])
    conditions = PAGE_CONDITIONS + ["glyph_small"]
    res = {c: {m: [] for m in methods} for c in conditions}
    oracle = {c: [] for c in conditions}
    seconds = {m: [] for m in methods}
    detection = {c: [] for c in EXPECTED_STEP}
    for c in conditions:
        profile = "glyph" if c == "glyph_small" else "page"
        for _ in range(n_glyph if c == "glyph_small" else n):
            s = make_sample(c, rng, fonts)
            oracle[c].append(oracle_f1(s, profile, cfg))
            for m in methods:
                t0 = time.perf_counter()
                out = run_method(m, s.image, profile, cfg)
                seconds[m].append(time.perf_counter() - t0)
                res[c][m].append(score(out, s))
            if c in EXPECTED_STEP:
                steps = preprocess_image(s.image, cfg, overrides={"deskew": False}).report["steps"]
                detection[c].append(EXPECTED_STEP[c] in steps)
        print(f"  {c:13s} " + " ".join(f"{m}={np.mean([r['f1'] for r in res[c][m]]):.3f}" for m in methods)
              + f" oracle={np.mean(oracle[c]):.3f}")

    # Deskew: tilted pages with a known angle, full pipeline.
    tilt_err = []
    for _ in range(n * 2):
        s = make_sample("tilt", rng, fonts)
        est = preprocess_image(s.image, cfg).report["deskew_angle"]
        tilt_err.append(abs(est - s.tilt_deg))

    agg = lambda rs, k: float(np.mean([r[k] for r in rs]))  # noqa: E731
    table = {c: {m: {"f1": agg(res[c][m], "f1"), "dot_recall": agg(res[c][m], "dot_recall"),
                     "precision": agg(res[c][m], "precision"), "recall": agg(res[c][m], "recall")}
                 for m in methods} for c in conditions}
    summary = {}
    for m in methods:
        page_f1 = float(np.mean([table[c][m]["f1"] for c in PAGE_CONDITIONS]))
        page_dots = float(np.mean([table[c][m]["dot_recall"] for c in PAGE_CONDITIONS]))
        summary[m] = {"page_mean_f1": page_f1, "page_dot_recall": page_dots,
                      "page_worst_f1": float(min(table[c][m]["f1"] for c in PAGE_CONDITIONS)),
                      "glyph_f1": table["glyph_small"][m]["f1"],
                      "glyph_p10_f1": float(np.percentile([r["f1"] for r in res["glyph_small"][m]], 10)),
                      "glyph_dot_recall": table["glyph_small"][m]["dot_recall"],
                      "seconds_per_image": float(np.mean(seconds[m]))}
    return {
        "samples_per_condition": n,
        "glyph_samples": n_glyph,
        "seed": cfg["benchmark"]["seed"],
        "per_condition": table,
        "oracle_f1": {c: float(np.mean(v)) for c, v in oracle.items()},
        "summary": summary,
        "deskew_mean_abs_error_deg": float(np.mean(tilt_err)),
        "deskew_max_abs_error_deg": float(np.max(tilt_err)),
        "condition_detection_rate": {c: float(np.mean(v)) for c, v in detection.items()},
    }


def choose(summary: dict, key_f1: str, key_dots: str, min_dots: float) -> str:
    """Best `key_f1` among methods that keep >= min_dots of the diacritic dots.

    Pages use the WORST condition's F1 (maximin): a method must not fail badly on any
    background type, since a new image can be any of them.
    """
    ok = {m: s for m, s in summary.items() if s[key_dots] >= min_dots} or summary
    return max(ok, key=lambda m: ok[m][key_f1])


def real_grid(cfg: dict, paths: dict, methods: list[str], out_dir: Path) -> list[Path]:
    """Side-by-side grids on real data. FINAL_TEST images are deliberately excluded."""
    pairs = list(csv.DictReader(open(paths["splits"]["handwritten"] / "pairs.csv", encoding="utf-8")))
    pages = [paths["raw"]["handwritten_images"] / p["image"] for p in pairs
             if p["split"] != "FINAL_TEST" and p["image"] in ("A2.png", "D4.png", "H4.png", "J2.png")]
    rf = sorted((paths["raw"]["roboflow_sindhi_ocrf"] / "train").glob("*.jpg"))[:1]
    letters = list(csv.DictReader(open(paths["splits"]["letters52"] / "split.csv", encoding="utf-8")))
    random.Random(cfg["debug"]["seed"]).shuffle(letters)
    glyphs = [paths["raw"]["letters52"] / r["relpath"] for r in letters if r["split"] == "train"][:8]
    outs = []
    for name, files, profile, width in (("pages", pages + rf, "page", 520), ("letters", glyphs, "glyph", 150)):
        rows = []
        for f in files:
            rgb = ops.load_rgb(f)
            tiles = [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)]
            for m in methods:
                out = run_method(m, rgb, profile, cfg) if m in ("otsu_raw", "rembg_otsu") else \
                    preprocess_image(rgb, cfg, profile=profile, method=m).image
                tiles.append(cv2.cvtColor(out, cv2.COLOR_GRAY2BGR))
            h = int(rgb.shape[0] * width / rgb.shape[1])
            tiles = [cv2.resize(t, (width, h), interpolation=cv2.INTER_AREA) for t in tiles]
            sep = np.full((h, 4, 3), (0, 0, 255), np.uint8)
            rows.append(np.hstack([x for t in tiles for x in (t, sep)]))
        header = np.full((26, rows[0].shape[1], 3), 255, np.uint8)
        for i, label in enumerate(["original"] + methods):
            cv2.putText(header, label, (i * (width + 4) + 4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        path = out_dir / f"comparison_real_{name}.png"
        cv2.imwrite(str(path), np.vstack([header] + rows))
        outs.append(path)
    return outs


def write_markdown(result: dict, path: Path) -> None:
    methods = list(result["summary"].keys())
    lines = ["# Step 1 method comparison (synthetic benchmark, known ground truth)", "",
             f"{result['samples_per_condition']} samples per page condition, {result['glyph_samples']} glyph samples, "
             f"seed {result['seed']}. Pages: method chosen by worst-condition F1; glyphs: by mean F1. "
             "Cells: text-pixel F1 / diacritic-dot recall. `oracle` = best possible single global "
             "threshold chosen with the ground truth (a ceiling, not a method).", "",
             "| condition | " + " | ".join(methods) + " | oracle |",
             "|---|" + "---|" * (len(methods) + 1)]
    for c, row in result["per_condition"].items():
        lines.append(f"| {c} | " + " | ".join(f"{row[m]['f1']:.3f} / {row[m]['dot_recall']:.2f}" for m in methods)
                     + f" | {result['oracle_f1'][c]:.3f} |")
    lines += ["", "| summary | " + " | ".join(methods) + " |", "|---|" + "---|" * len(methods)]
    for k in ("page_mean_f1", "page_worst_f1", "page_dot_recall", "glyph_f1", "glyph_p10_f1",
              "glyph_dot_recall", "seconds_per_image"):
        lines.append(f"| {k} | " + " | ".join(f"{result['summary'][m][k]:.3f}" for m in methods) + " |")
    lines += ["", f"Chosen for pages: **{result['chosen']['page']}**; for letter crops: **{result['chosen']['glyph']}**.",
              f"Deskew: mean |error| {result['deskew_mean_abs_error_deg']:.2f} deg, "
              f"max {result['deskew_max_abs_error_deg']:.2f} deg.",
              "Condition detection rate (expected step fired): " +
              ", ".join(f"{c}: {v:.0%}" for c, v in result["condition_detection_rate"].items()), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--small", action="store_true", help="2 samples per condition (quick check)")
    ap.add_argument("--no-rembg", action="store_true", help="skip the rembg method")
    args = ap.parse_args()
    cfg, paths = load_config("step1"), load_paths()
    out_dir = paths["results"]["step1"]
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = ["otsu_raw"] + ([] if args.no_rembg else ["rembg_otsu"]) + PIPELINE_METHODS
    n = 2 if args.small else cfg["benchmark"]["samples_per_condition"]
    n_glyph = 4 if args.small else cfg["benchmark"]["glyph_samples"]

    print(f"Synthetic benchmark ({n}/page condition, {n_glyph} glyphs): F1 per method")
    result = synthetic_benchmark(cfg, n, n_glyph, methods)
    min_dots = cfg["gate"]["min_dot_recall"]
    result["chosen"] = {"page": choose(result["summary"], "page_worst_f1", "page_dot_recall", min_dots),
                        "glyph": choose(result["summary"], "glyph_f1", "glyph_dot_recall", min_dots)}
    result["configured"] = {"page": cfg["method"], "glyph": cfg["glyph"]["method"]}
    for prof in ("page", "glyph"):
        if result["chosen"][prof] != result["configured"][prof]:
            print(f"NOTE: best {prof} method is {result['chosen'][prof]} but config uses "
                  f"{result['configured'][prof]}. Update configs/step1_preprocess.yaml if you agree.")
    (out_dir / "method_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, out_dir / "method_comparison.md")
    grids = real_grid(cfg, paths, methods, out_dir)
    print(f"Chosen: {result['chosen']}. Deskew error {result['deskew_mean_abs_error_deg']:.2f} deg.")
    print(f"Wrote {out_dir / 'method_comparison.md'} and {[g.name for g in grids]}")


if __name__ == "__main__":
    main()
