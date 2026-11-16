"""Step 2 data: word lists (fixed splits) and printed-font word renders.

1. Word lists from Sindhi Wikipedia (normalised, letters only, 2-12 chars, seen >= 3 times):
     data/splits/step2_words/{train,val,test}.txt   (tracked in git)
   The 300 most frequent words always go to train. val/test are held-out word TYPES drawn from
   the rest, so the test measures reading words the model has never seen.
   Leakage check: none of the FINAL_TEST transcriptions are used here; the overlap of FINAL_TEST
   words with this vocabulary is reported (common words naturally overlap).
2. Printed renders via Tesseract's text2image (proper Arabic-script shaping):
     data/synthetic/step2/printed/{train,val,test}/<id>.png + labels.csv
   train/val use the training fonts; test uses held-out words AND a held-out font (Tahoma).

Usage: python scripts/step2_make_synthetic.py [--small] [--workers N]
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cnn_transformer.train_utils import load_step2_config  # noqa: E402
from common.paths import REPO_ROOT, load_paths  # noqa: E402
from common.text import normalize  # noqa: E402

# Sindhi letters (Arabic block + Sindhi additions) and the hamza forms used in Sindhi words.
LETTERS = set("ابٻڀتٿٽٺثپجڄڃچڇحخدڌڏڊڍذرڙزژسشصضطظعغفڦقڪکگڳڱلمنڻوهھءيآئ۾۽أؤإ")
DIGIT_RUN = re.compile(r"\d+")


def word_lists(cfg: dict, paths: dict, out_dir: Path, seed: int) -> dict[str, list[str]]:
    pf = next(paths["raw"]["corpus_wikipedia_sd"].glob("*.parquet"))
    counts = Counter()
    for text in pq.read_table(pf, columns=["text"]).column("text").to_pylist():
        for tok in normalize(text, diacritics=False).split():
            tok = tok.strip(".,،؛:!?()[]\"'«»-_")
            if cfg["min_word_len"] <= len(tok) <= cfg["max_word_len"] and set(tok) <= LETTERS:
                counts[tok] += 1
    vocab = [w for w, c in counts.most_common() if c >= 3]
    head, rest = vocab[:300], vocab[300:]
    rng = random.Random(seed)
    rng.shuffle(rest)
    n_test, n_val, n_train = cfg["words_test"], cfg["words_val"], cfg["words_train"]
    splits = {"test": rest[:n_test], "val": rest[n_test:n_test + n_val],
              "train": head + rest[n_test + n_val:n_test + n_val + max(0, n_train - len(head))]}
    out_dir.mkdir(parents=True, exist_ok=True)
    for s, ws in splits.items():
        (out_dir / f"{s}.txt").write_text("\n".join(ws) + "\n", encoding="utf-8")
    # leakage report: how many FINAL_TEST words are in the training vocabulary?
    pairs = list(csv.DictReader(open(paths["splits"]["handwritten"] / "pairs.csv", encoding="utf-8")))
    final_words = []
    for p in pairs:
        if p["split"] == "FINAL_TEST":
            t = (paths["raw"]["handwritten_text_gt"] / p["text_gt"]).read_text(encoding="utf-8")
            final_words += [w.strip(".,،:()") for w in normalize(t, diacritics=False).split()]
    train_set = set(splits["train"])
    overlap = sum(w in train_set for w in final_words) / max(1, len(final_words))
    print(f"vocab {len(vocab)} words (>=3 occurrences); train {len(splits['train'])}, val {len(splits['val'])}, "
          f"test {len(splits['test'])}; FINAL_TEST word tokens covered by train vocab: {overlap:.0%}")
    return splits


def decorate(word: str, rng: random.Random) -> str:
    """Occasionally attach punctuation or use a number, like the real texts."""
    r = rng.random()
    if r < 0.05:
        return str(rng.choice([rng.randint(1, 9), rng.randint(10, 99), rng.randint(100, 999)]))
    if r < 0.10:
        return word + rng.choice(["،", "."])
    if r < 0.13:
        return rng.choice(["،", "."]) + word
    return word


def visual_to_logical(box_text: str) -> str:
    """text2image writes RTL words in visual order; digit runs stay left-to-right."""
    rev = box_text[::-1]
    return DIGIT_RUN.sub(lambda m: m.group(0)[::-1], rev)


def render_chunk(job: tuple) -> list[tuple[str, np.ndarray]]:
    words, font, ptsize, fc_dir = job
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "in.txt").write_text(" ".join(words) + "\n", encoding="utf-8")
        cmd = ["text2image", "--text", str(tmp / "in.txt"), "--outputbase", str(tmp / "out"),
               "--font", font, "--fonts_dir", "/System/Library/Fonts", "--fontconfig_tmpdir", str(fc_dir),
               "--ptsize", str(ptsize), "--resolution", "150", "--xsize", "1400", "--ysize", "2400",
               "--margin", "40", "--output_word_boxes", "--degrade_image=false",
               "--leading", "20", "--char_spacing", "0.0"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not (tmp / "out.tif").exists():
            return []
        tif = Image.open(tmp / "out.tif")
        expected = Counter(words)
        out = []
        for line in (tmp / "out.box").read_text(encoding="utf-8").splitlines():
            parts = line.rsplit(" ", 5)
            if len(parts) != 6 or not parts[0].strip():
                continue
            label = visual_to_logical(parts[0])
            if expected[label] <= 0:  # unmatched box (mis-split word) -> skip
                continue
            expected[label] -= 1
            l, b, rgt, t, page = map(int, parts[1:])
            tif.seek(page)
            page_img = np.asarray(tif.convert("L"))
            h = page_img.shape[0]
            pad = max(4, (t - b) // 4)
            crop = page_img[max(0, h - t - pad):min(h, h - b + pad), max(0, l - pad):min(page_img.shape[1], rgt + pad)]
            if crop.size == 0:
                continue
            _, crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            out.append((label, crop))
        return out


def render_split(name: str, words: list[str], fonts: list[str], cfg: dict, out_root: Path, workers: int,
                 seed: int) -> int:
    rng = random.Random(seed)
    out_dir = out_root / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    fc_dir = out_root / ".fontconfig"
    fc_dir.mkdir(exist_ok=True)
    reps = cfg["renders_per_word"]
    items = [decorate(w, rng) for w in words for _ in range(reps)]
    rng.shuffle(items)
    jobs = [(items[i:i + 150], rng.choice(fonts), rng.randint(*cfg["ptsize"]), fc_dir)
            for i in range(0, len(items), 150)]
    rows, n = [], 0
    with Pool(workers) as pool:
        for chunk, (_, font, pts, _) in zip(pool.imap(render_chunk, jobs), jobs):
            for label, crop in chunk:
                fn = f"{n:06d}.png"
                cv2.imwrite(str(out_dir / fn), crop)
                rows.append({"file": fn, "label": label, "font": font, "ptsize": pts})
                n += 1
    with open(out_dir / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "font", "ptsize"])
        w.writeheader()
        w.writerows(rows)
    print(f"  printed/{name}: {n} word images from {len(items)} requested ({n / max(1, len(items)):.0%} matched)")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cfg = load_step2_config(args.small)
    sc = cfg["synthetic"]
    paths = load_paths()
    words_dir = REPO_ROOT / "data" / "splits" / ("step2_words_small" if args.small else "step2_words")
    splits = word_lists(sc, paths, words_dir, cfg["seed"])
    out_root = paths["synthetic"] / ("step2_small" if args.small else "step2") / "printed"
    render_split("train", splits["train"], sc["fonts_train"], sc, out_root, args.workers, cfg["seed"])
    render_split("val", splits["val"], sc["fonts_train"], sc, out_root, args.workers, cfg["seed"] + 1)
    render_split("test", splits["test"], sc["fonts_test"], sc, out_root, args.workers, cfg["seed"] + 2)


if __name__ == "__main__":
    main()
