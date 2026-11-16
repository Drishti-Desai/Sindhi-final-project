"""Inspect the two POS-tagging datasets copied on 2026-11-14 (see data/PROVENANCE.md). Nothing is trained.

Images (data/raw/pos_handwritten/): duplicates (exact and near), sizes; every unique image is cleaned with Step 1,
  split into lines and words by our own finder (Step 3 settings) and read by the Step 2 recognizer. The draft is
  matched against every text we have (the POS sentences, Sindhi_Legal_Dataset.csv, our handwritten texts) to see
  whether any image has a known transcription. The drafts are kept: they are the pre-fill for transcription.
Text (data/raw/pos_text/SindhiPosDataset.csv): one token per row with POS tags; sentences restart at ID 1. Tags are
  stripped, the text is checked for non-Sindhi characters, and any sentence that overlaps a test or FINAL_TEST
  text (same sentence, or 4 consecutive words in common) is removed.

Outputs: data/splits/pos_handwritten/images.csv, data/preprocessed/pos_text/{sentences.csv,sentences_clean.txt},
         results/step3/pos_inspection.json
Usage: python scripts/inspect_pos_data.py [--limit N]
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cnn_transformer.segment import components  # noqa: E402
from cnn_transformer.train_utils import get_device  # noqa: E402
from common.metrics import edit_distance  # noqa: E402
from common.paths import REPO_ROOT, load_config, load_paths  # noqa: E402
from preprocess import preprocess_image  # noqa: E402
from tesseract_ft.boxes import own_layout  # noqa: E402
from tesseract_ft.pages import EvalPage, filter_comps, keep_only, label_norm_words, load_upright  # noqa: E402

_spec = importlib.util.spec_from_file_location("step3_baseline", REPO_ROOT / "scripts" / "step3_baseline.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

LEGAL_CSV = Path.home() / "Downloads" / "Sindhi_Legal_Dataset.csv"   # read only
OWN = dict(finder="tracked", gap_h=0.3, gap_med=0.75)                  # Step 3 own-finder settings


def dhash(path: Path) -> int:
    im = ImageOps.exif_transpose(Image.open(path)).convert("L").resize((17, 16))
    a = np.asarray(im, dtype=np.int16)
    return int("".join("1" if v else "0" for v in (a[:, 1:] > a[:, :-1]).flatten()), 2)


def pos_sentences(path: Path) -> list[list[dict]]:
    sents, cur = [], []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["ID"].strip() == "1" and cur:
            sents.append(cur)
            cur = []
        cur.append(r)
    return sents + [cur]


def known_texts(paths) -> dict[str, list[str]]:
    """Every transcription we have, normalised as recognizer labels, grouped by source."""
    out = collections.defaultdict(list)
    for r in csv.DictReader(open(LEGAL_CSV, encoding="utf-8-sig")):
        for v in r.values():
            if v and v.strip():
                out["legal_csv"].append(" ".join(label_norm_words(v)))
    for p in sorted(paths["raw"]["handwritten_text_gt"].glob("*.txt")):
        out["handwritten"].append(" ".join(label_norm_words(p.read_text(encoding="utf-8"))))
    for p in sorted(paths["raw"]["handwritten_batch2"].rglob("*.txt")):
        out["batch2"].append(" ".join(label_norm_words(p.read_text(encoding="utf-8"))))
    return out


def test_texts(paths) -> list[str]:
    """Texts of every test page: unseen writer, TESS_TEST, FINAL_TEST (and batch 2 EXCLUDED copies of D pages)."""
    texts = []
    pairs = csv.DictReader(open(paths["splits"]["handwritten"] / "pairs.csv", encoding="utf-8"))
    for p in pairs:
        if p["split"] in ("TESS_TEST", "FINAL_TEST"):
            texts.append((paths["raw"]["handwritten_text_gt"] / p["text_gt"]).read_text(encoding="utf-8"))
    root = paths["raw"]["handwritten_batch2"]
    for m in csv.DictReader(open(paths["splits"]["handwritten_batch2"] / "manifest.csv", encoding="utf-8")):
        if m["split"] in ("TEST_UNSEEN_WRITER", "EXCLUDED") and m["text"]:
            corr = paths["raw"]["handwritten_batch2_text_corrected"] / f"{m['name']}.txt"
            src = corr if corr.exists() else root / m["text"]
            if src.exists():
                texts.append(src.read_text(encoding="utf-8"))
    return [" ".join(label_norm_words(t)) for t in texts]


def ngrams(words: list[str], n: int) -> set:
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def trigram_vec(s: str) -> collections.Counter:
    s = f"  {s}  "
    return collections.Counter(s[i:i + 3] for i in range(len(s) - 2))


def best_matches(draft: str, cands: list[str], vecs: list[collections.Counter], k: int = 15):
    """Character-trigram shortlist, then exact CER of the draft against each shortlisted text."""
    v = trigram_vec(draft)
    tv = sum(v.values())
    sims = [2 * sum(min(n, c[g]) for g, n in v.items()) / (tv + sum(c.values())) for c in vecs]  # Dice overlap
    top = np.argsort(sims)[::-1][:k]
    scored = sorted((edit_distance(cands[i], draft) / max(1, len(cands[i])), int(i)) for i in top)
    return scored[:2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, help="only the first N unique images (quick check)")
    args = ap.parse_args()
    paths, cfg1 = load_paths(), load_config("step1")
    report = {}

    # ---------------- text
    sents = pos_sentences(paths["raw"]["pos_text"] / "SindhiPosDataset.csv")
    tests = test_texts(paths)
    test_ng = set().union(*(ngrams(t.split(), 4) for t in tests))
    test_joined = "\n".join(tests)
    allowed = set("،؟۔!.,:;\"'()-") | {" "}
    rows_out, odd_chars = [], collections.Counter()
    for k, s in enumerate(sents, 1):
        text = unicodedata.normalize("NFC", " ".join(r["FORM"].strip() for r in s if r["FORM"].strip()))
        text = text.replace(" .", ".").replace(" !", "!").replace(" ؟", "؟").replace(" ،", "،")
        for ch in text:
            if not ("؀" <= ch <= "ۿ" or ch in allowed):
                odd_chars[ch] += 1
        words = label_norm_words(text)
        reason = ""
        if not any("\u0620" <= ch <= "\u06ff" for ch in text):
            reason = "no Sindhi words"
        elif " ".join(words) in test_joined:
            reason = "sentence appears in a test text"
        elif ngrams(words, 4) & test_ng:
            reason = "shares 4 consecutive words with a test text"
        rows_out.append({"id": k, "text": text, "tokens": len(s), "removed": reason})
    seen, dup = set(), 0
    for r in rows_out:
        key = " ".join(label_norm_words(r["text"]))
        if not r["removed"] and key in seen:
            r["removed"] = "duplicate sentence"
            dup += 1
        seen.add(key)
    out_text = REPO_ROOT / "data" / "preprocessed" / "pos_text"
    out_text.mkdir(parents=True, exist_ok=True)
    with open(out_text / "sentences.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    kept = [r["text"] for r in rows_out if not r["removed"]]
    (out_text / "sentences_clean.txt").write_text("\n".join(kept) + "\n", encoding="utf-8")
    kept_words = [w for t in kept for w in label_norm_words(t)]
    report["text"] = {"sentences": len(sents), "tokens": sum(len(s) for s in sents),
                      "removed": dict(collections.Counter(r["removed"] for r in rows_out if r["removed"])),
                      "kept_sentences": len(kept), "kept_words": len(kept_words), "kept_unique_words": len(set(kept_words)),
                      "odd_characters": dict(odd_chars), "upos": dict(collections.Counter(r["UPOS"] for s in sents for r in s))}
    print(json.dumps(report["text"], ensure_ascii=False))

    # ---------------- images
    root = paths["raw"]["pos_handwritten"]
    files = sorted(p for p in root.rglob("*") if p.is_file())
    by_md5, rows = {}, []
    for p in files:
        h = hashlib.md5(p.read_bytes()).hexdigest()
        im = Image.open(p)
        rows.append({"file": str(p.relative_to(root)), "md5": h, "format": im.format,
                     "width": ImageOps.exif_transpose(im).size[0], "height": ImageOps.exif_transpose(im).size[1],
                     "exact_dup_of": by_md5.get(h, "")})
        by_md5.setdefault(h, str(p.relative_to(root)))
    uniq = [r for r in rows if not r["exact_dup_of"]]
    hashes = {r["file"]: dhash(root / r["file"]) for r in uniq}
    names = list(hashes)
    for i, a in enumerate(names):  # near duplicates: re-saved / re-scanned copies of the same photo
        for b in names[:i]:
            if bin(hashes[a] ^ hashes[b]).count("1") <= 6:
                next(r for r in rows if r["file"] == a)["near_dup_of"] = b
                break
    ours = {}
    for d in ("letters52", "handwritten", "handwritten_batch2", "handwritten_batch3"):
        for p in paths["raw"][d if d != "handwritten" else "handwritten_images"].rglob("*"):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                ours[hashlib.md5(p.read_bytes()).hexdigest()] = str(p)
    exact_ours = sum(r["md5"] in ours for r in rows)

    cands = {**known_texts(paths), "pos_sentences": [" ".join(label_norm_words(r["text"])) for r in rows_out]}
    flat = [(src, t) for src, ts in cands.items() for t in ts if t]
    texts = [t for _, t in flat]
    vecs = [trigram_vec(t) for t in texts]
    device = get_device("auto")
    rec = base.load_recognizer(device)
    todo = [r for r in uniq if not r.get("near_dup_of")][: args.limit]
    for n, r in enumerate(todo, 1):
        rgb = load_upright(root / r["file"], "")
        res = preprocess_image(rgb, cfg1, profile="page")
        comps, lab = components(res.image)
        b_img, g_img = keep_only(res.image, res.gray, lab, filter_comps(comps, res.image.shape))
        lines = base.reading_order(own_layout(b_img, **OWN))
        boxes = [x for ln in lines for x in ln]
        page = EvalPage(r["file"], "", "", b_img, g_img, [])
        r["lines"] = len(lines)
        r["words_found"] = len(boxes)
        r["draft"] = " ".join(t for t in base.read_boxes(rec, page, boxes, device) if t) if boxes else ""
        if r["draft"]:
            (c1, i1), (c2, _) = best_matches(r["draft"], texts, vecs)
            r.update(best_source=flat[i1][0], best_text=texts[i1], best_cer=round(c1, 3), second_cer=round(c2, 3))
        if n % 50 == 0:
            print(f"  drafted {n}/{len(todo)}")
    out_split = REPO_ROOT / "data" / "splits" / "pos_handwritten"
    out_split.mkdir(parents=True, exist_ok=True)
    fields = ["file", "md5", "format", "width", "height", "exact_dup_of", "near_dup_of", "lines", "words_found",
              "draft", "best_source", "best_cer", "second_cer", "best_text"]
    with open(out_split / "images.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    cer = np.array([r["best_cer"] for r in todo if "best_cer" in r])
    report["images"] = {"files": len(rows), "formats": dict(collections.Counter(r["format"] for r in rows)),
                        "exact_duplicates_within": len(rows) - len(uniq),
                        "near_duplicates_within": sum(bool(r.get("near_dup_of")) for r in rows),
                        "exact_duplicates_of_our_images": exact_ours, "drafted": len(todo),
                        "lines_per_image": dict(collections.Counter(r["lines"] for r in todo)),
                        "best_match_cer_percentiles": {p: round(float(np.percentile(cer, p)), 3) for p in (0, 1, 5, 25, 50)},
                        "best_match_cer_below": {t: int((cer < t).sum()) for t in (0.2, 0.3, 0.4, 0.5)},
                        "best_source_when_below_0.4": dict(collections.Counter(r["best_source"] for r in todo
                                                                               if r.get("best_cer", 1) < 0.4))}
    (paths["results"]["step3"] / "pos_inspection.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                                                   encoding="utf-8")
    print(json.dumps(report["images"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
