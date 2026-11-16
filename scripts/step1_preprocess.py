"""Step 1 batch runner: preprocess every dataset and write the Step 1 gate.

Input -> output (paths from configs/paths.yaml):
  letters52             unique images from data/splits/letters52/manifest.csv
                        -> data/preprocessed/letters52/<class_id>/<original stem>.png  (+ index.csv)
  handwritten           data/raw/handwritten/images/*.png -> data/preprocessed/handwritten/<stem>.png
  roboflow_sindhi_ocrf  <split>/*.jpg -> <split>/<stem>.png, geometry unchanged, COCO json copied
                        with file names switched to .png (boxes stay valid)

Also writes results/step1/report_<dataset>.csv (stats + steps per image),
results/step1/debug/<dataset>/*_before_after.png, results/step1/preprocess_summary.json and,
on a full run, results/gates/step1_preprocessing.json.

Usage:
  python scripts/step1_preprocess.py                      # everything + gate
  python scripts/step1_preprocess.py --dataset handwritten
  python scripts/step1_preprocess.py --small              # 20 images per dataset, no gate
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.gates import write_gate  # noqa: E402
from common.paths import REPO_ROOT, load_config, load_paths  # noqa: E402
from preprocess import ops, preprocess_image  # noqa: E402

DATASETS = ["letters52", "handwritten", "roboflow_sindhi_ocrf"]
_CFG: dict = {}


def _init_worker(cfg: dict) -> None:
    global _CFG
    _CFG = cfg
    cv2.setNumThreads(1)  # one OpenCV thread per process; parallelism comes from the pool


def _process(job: tuple[str, str, str, dict]) -> dict:
    src, dst, profile, overrides = job
    try:
        r = preprocess_image(src, _CFG, profile=profile, overrides=overrides)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(dst, r.image)
        rep = r.report
        return {"src": src, "dst": dst, "ok": True, "profile": rep["profile"], "method": rep["method"],
                "steps": " ".join(rep["steps"]), "deskew_angle": rep["deskew_angle"], "scale": rep["scale"],
                "stroke_width": rep["stroke_width"], "seconds": rep["seconds"], **rep["stats"]}
    except Exception as e:  # keep going; failures are counted and block the gate
        return {"src": src, "dst": dst, "ok": False, "error": repr(e)}


def jobs_for(dataset: str, cfg: dict, paths: dict) -> tuple[list[tuple], dict]:
    dcfg = dict(cfg["datasets"][dataset])
    profile = dcfg.pop("profile")
    overrides = dcfg  # e.g. roboflow: deskew false, max_side null
    out_root = paths["preprocessed"][dataset]
    jobs, meta = [], {}
    if dataset == "letters52":
        raw = paths["raw"]["letters52"]
        rows = list(csv.DictReader(open(paths["splits"]["letters52"] / "split.csv", encoding="utf-8")))
        for r in rows:
            out = out_root / r["class_id"] / (Path(r["relpath"]).stem + ".png")
            jobs.append((str(raw / r["relpath"]), str(out), profile, overrides))
        meta["index"] = [{"raw_relpath": r["relpath"], "relpath": f"{r['class_id']}/{Path(r['relpath']).stem}.png",
                          "class_id": r["class_id"], "sheet": r["sheet"], "split": r["split"]} for r in rows]
    elif dataset == "handwritten":
        for img in sorted(paths["raw"]["handwritten_images"].glob("*.png")):
            jobs.append((str(img), str(out_root / img.name), profile, overrides))
    elif dataset == "roboflow_sindhi_ocrf":
        raw = paths["raw"]["roboflow_sindhi_ocrf"]
        for split in ("train", "valid", "test"):
            for img in sorted((raw / split).glob("*.jpg")):
                jobs.append((str(img), str(out_root / split / (img.stem + ".png")), profile, overrides))
    return jobs, meta


def write_dataset_extras(dataset: str, meta: dict, paths: dict) -> None:
    out_root = paths["preprocessed"][dataset]
    if dataset == "letters52":
        with open(out_root / "index.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(meta["index"][0].keys()))
            w.writeheader()
            w.writerows(meta["index"])
    elif dataset == "roboflow_sindhi_ocrf":
        raw = paths["raw"]["roboflow_sindhi_ocrf"]
        for split in ("train", "valid", "test"):
            coco = json.loads((raw / split / "_annotations.coco.json").read_text(encoding="utf-8"))
            for im in coco["images"]:
                im["file_name"] = Path(im["file_name"]).stem + ".png"
            (out_root / split).mkdir(parents=True, exist_ok=True)
            (out_root / split / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")


def save_debug(results: list[dict], dataset: str, cfg: dict, paths: dict) -> None:
    ok = [r for r in results if r["ok"]]
    random.Random(cfg["debug"]["seed"]).shuffle(ok)
    out_dir = paths["results"]["step1_debug"] / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in ok[: cfg["debug"]["samples_per_dataset"]]:
        before = cv2.cvtColor(ops.load_rgb(r["src"]), cv2.COLOR_RGB2BGR)
        after = cv2.cvtColor(cv2.imread(r["dst"], cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
        h = 160 if dataset == "letters52" else 500
        before = cv2.resize(before, (int(before.shape[1] * h / before.shape[0]), h))
        after = cv2.resize(after, (int(after.shape[1] * h / after.shape[0]), h), interpolation=cv2.INTER_AREA)
        sep = np.full((h, 8, 3), (0, 0, 255), np.uint8)
        cv2.imwrite(str(out_dir / f"{Path(r['dst']).stem}_before_after.png"), np.hstack([before, sep, after]))


def write_report(results: list[dict], dataset: str, paths: dict) -> None:
    keys = sorted({k for r in results for k in r})
    with open(paths["results"]["step1"] / f"report_{dataset}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)


def run_unit_tests() -> bool:
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_preprocess.py"],
                         cwd=REPO_ROOT, capture_output=True, text=True)
    print(out.stdout.strip().splitlines()[-1] if out.stdout else out.stderr)
    return out.returncode == 0


def gate(summary: dict, cfg: dict, paths: dict) -> dict:
    comp_path = paths["results"]["step1"] / "method_comparison.json"
    if not comp_path.exists():
        sys.exit("Run scripts/step1_compare_methods.py first (the gate uses its benchmark).")
    comp = json.loads(comp_path.read_text(encoding="utf-8"))
    g = cfg["gate"]
    page_m, glyph_m = cfg["method"], cfg["glyph"]["method"]
    page, glyph = comp["summary"][page_m], comp["summary"][glyph_m]
    glyph_vs_oracle = glyph["glyph_f1"] / comp["oracle_f1"]["glyph_small"]
    failed = sum(d["failed"] for d in summary["datasets"].values())
    missing = sum(d["expected"] - d["processed"] for d in summary["datasets"].values())
    checks = {
        "page_mean_f1": {"value": round(page["page_mean_f1"], 4), "threshold": g["min_page_mean_f1"]},
        "page_dot_recall": {"value": round(page["page_dot_recall"], 4), "threshold": g["min_dot_recall"]},
        "glyph_f1_vs_oracle": {"value": round(glyph_vs_oracle, 4), "threshold": g["min_glyph_f1_vs_oracle"]},
        "glyph_dot_recall": {"value": round(glyph["glyph_dot_recall"], 4), "threshold": g["min_dot_recall"]},
        "deskew_mean_abs_error_deg": {"value": round(comp["deskew_mean_abs_error_deg"], 3),
                                      "threshold": g["max_deskew_error_deg"], "op": "<="},
        "configured_methods_match_benchmark_choice": {
            "value": int(comp["chosen"] == {"page": page_m, "glyph": glyph_m}), "threshold": 1, "op": "=="},
        "images_failed": {"value": failed, "threshold": 0, "op": "=="},
        "images_missing": {"value": missing, "threshold": 0, "op": "=="},
        "unit_tests_pass": {"value": int(run_unit_tests()), "threshold": 1, "op": "=="},
    }
    return write_gate("step1_preprocessing", checks, extra={
        "page_method": page_m, "glyph_method": glyph_m,
        "glyph_f1_absolute": round(glyph["glyph_f1"], 4),
        "glyph_oracle_f1": round(comp["oracle_f1"]["glyph_small"], 4),
        "page_worst_condition_f1": round(page["page_worst_f1"], 4),
        "datasets": summary["datasets"],
        "note": "Synthetic benchmark metrics (known ground truth); real images checked visually "
                "in results/step1/comparison_real_*.png and results/step1/debug/.",
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=DATASETS + ["all"], default="all")
    ap.add_argument("--small", action="store_true", help="first 20 images per dataset, no gate")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    cfg, paths = load_config("step1"), load_paths()
    paths["results"]["step1"].mkdir(parents=True, exist_ok=True)
    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    summary = {"datasets": {}, "small_mode": args.small}
    for ds in datasets:
        jobs, meta = jobs_for(ds, cfg, paths)
        expected = len(jobs)
        if args.small:
            jobs = jobs[:20]
        t0 = time.time()
        with Pool(args.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
            results = pool.map(_process, jobs, chunksize=8)
        n_ok = sum(r["ok"] for r in results)
        if not args.small:
            write_dataset_extras(ds, meta, paths)
        write_report(results, ds, paths)
        save_debug(results, ds, cfg, paths)
        steps = {}
        for r in results:
            for s in r.get("steps", "").split():
                name = s.split(":")[0]
                steps[name] = steps.get(name, 0) + 1
        summary["datasets"][ds] = {"expected": expected if not args.small else len(jobs),
                                   "processed": n_ok, "failed": len(results) - n_ok,
                                   "seconds": round(time.time() - t0, 1), "step_counts": steps}
        print(f"{ds}: {n_ok}/{len(results)} ok in {time.time() - t0:.1f}s; steps {steps}")
        for r in results:
            if not r["ok"]:
                print(f"  FAILED {r['src']}: {r['error']}")

    (paths["results"]["step1"] / "preprocess_summary.json").write_text(json.dumps(summary, indent=2),
                                                                       encoding="utf-8")
    if args.small or args.dataset != "all":
        print("Gate not written (needs a full run: no --small, --dataset all).")
        return
    g = gate(summary, cfg, paths)
    print(f"Gate step1_preprocessing: {g['status']}")
    for k, c in g["checks"].items():
        print(f"  {'OK  ' if c['passed'] else 'FAIL'} {k}: {c['value']} ({c['op']} {c['threshold']})")


if __name__ == "__main__":
    main()
