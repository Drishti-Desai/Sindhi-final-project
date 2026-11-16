"""Cut handwritten batch 2 into labelled WORD crops, check them, and build a review page.

For every usable file in data/splits/handwritten_batch2/manifest.csv (status ok, or text_mismatch
with a confirmed corrected text):
  1. load, turn upright (rotate_ccw), upscale tiny line images, clean with Step 1 (page profile)
  2. find the visual lines automatically; on top-level pages drop hand-drawn circles round numbers
  3. align the file's word sequence to the ink across all lines (cnn_transformer.segment.align_page)
  4. quality check: the current recognizer reads each crop. A line whose crops it reads very badly on
     average (mean CER > LINE_FLAG_CER) has most likely got shifted labels, so it is FLAGGED for a
     visual check. Nothing is rejected automatically (that would bias training toward easy writing).

Outputs:
  data/preprocessed/handwritten_batch2_words/<split>/<name>_w<k>.png
  data/splits/handwritten_batch2/words.csv        one row per crop; `verified` starts as the QC guess
  results/step2/debug/batch2_review/index.html    review page: page images with numbered boxes and every
                                                  crop with its label; untick wrong ones, then download CSV
FINAL_TEST pages are never read. Nothing here trains a model.

Usage: python scripts/step2_extract_batch2_words.py [--only NAME ...] [--no-qc]
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cnn_transformer.align import forced_align  # noqa: E402
from cnn_transformer.models import WordRecognizer  # noqa: E402
from cnn_transformer.segment import (align_page, cluster_lines, cluster_lines_simple, components,  # noqa: E402
                                    erase_ruled_lines, remove_circled_numbers)
from cnn_transformer.train_utils import get_device, load_step2_config  # noqa: E402
from cnn_transformer.words import EOS, BOS, Vocab, collate, ink_crop, label_norm, to_ink, to_tensor_image  # noqa: E402
from common.metrics import edit_distance  # noqa: E402
from common.paths import REPO_ROOT, load_config, load_paths  # noqa: E402
from preprocess import preprocess_image  # noqa: E402

# Notebook pages whose curved ruled lines touch the letters: line finding fails on them. Skipped this round
# (about 227 words); see the review page. A curve-aware ruling remover would be needed first.
SKIP_RULED = {"221-223", "224-226", "227-229", "poem1"}
LINE_FLAG_CER = 0.65   # mean recognizer CER over a line above this -> labels probably shifted
MIN_HEIGHT = 150       # tiny line images (A6.. are ~40-50 px tall) are upscaled to this before cleaning
COLORS = [(0, 140, 255), (0, 170, 0), (200, 0, 200), (255, 120, 0), (0, 160, 160)]


def load_upright(path: Path, rotate_ccw: str) -> np.ndarray:
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGBA")
    im = Image.alpha_composite(Image.new("RGBA", im.size, "white"), im).convert("RGB")
    if rotate_ccw:
        im = im.rotate(int(rotate_ccw), expand=True, fillcolor="white")
    rgb = np.asarray(im)
    if rgb.shape[0] < MIN_HEIGHT:
        s = MIN_HEIGHT / rgb.shape[0]
        rgb = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    return rgb


def load_recognizer(device):
    st = torch.load(REPO_ROOT / "checkpoints" / "step2_recognizer" / "best.pt", map_location="cpu", weights_only=False)
    cfg = load_step2_config()
    rc = cfg["recognizer"]
    vocab = Vocab("")
    vocab.itos = st["extra"]["vocab"]
    vocab.stoi = {c: i for i, c in enumerate(vocab.itos)}
    m = WordRecognizer(len(vocab), cfg["letters"]["widths"], rc["d_model"], rc["enc_layers"], rc["dec_layers"],
                       rc["heads"], rc["ff"], rc["dropout"], rc["max_label_len"]).to(device)
    m.load_state_dict(st["model"])
    return m.eval(), vocab, rc


def read_crops(model, vocab, rc, device, paths: list[Path]) -> list[str]:
    items = [(to_tensor_image(ink_crop(to_ink(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))), rc["height"],
                              rc["max_width"]), "") for p in paths]
    out = []
    for i in range(0, len(items), 32):
        x, w, _, _ = collate(items[i:i + 32], vocab)
        seq, _ = model.greedy(x.to(device), w.to(device), BOS, EOS, 26)
        out += [vocab.decode(s) for s in seq.cpu()]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="process only these file names")
    ap.add_argument("--no-qc", action="store_true", help="skip the recognizer check")
    ap.add_argument("--width-only", action="store_true", help="align by gap/width only (no model guidance)")
    ap.add_argument("--html-only", action="store_true", help="only rebuild the review page from words.csv")
    args = ap.parse_args()
    paths, cfg1 = load_paths(), load_config("step1")
    if args.html_only:
        rebuild_html_only(paths)
        return
    root = paths["raw"]["handwritten_batch2"]
    corrected = paths["raw"]["handwritten_batch2_text_corrected"]
    manifest = list(csv.DictReader(open(paths["splits"]["handwritten_batch2"] / "manifest.csv", encoding="utf-8")))
    todo = [m for m in manifest if m["split"] in ("TRAIN", "VAL", "TEST_UNSEEN_WRITER")
            and (not args.only or m["name"] in args.only)]
    out_root = REPO_ROOT / "data" / "preprocessed" / "handwritten_batch2_words"
    review = REPO_ROOT / "results" / "step2" / "debug" / "batch2_review"
    (review / "pages").mkdir(parents=True, exist_ok=True)

    device = get_device("auto")
    model = vocab = rc = None
    if not (args.width_only and args.no_qc):
        model, vocab, rc = load_recognizer(device)
    rows, pages = [], []
    for m in todo:
        if m["name"] in SKIP_RULED:
            print(f"  {m['name']}: skipped (curved ruled lines touch the letters)")
            pages.append({"name": m["name"], "m": m, "n_lines": 0, "skipped": True, "rings": 0,
                          "reason": "curved notebook ruling touches the letters; needs a better ruling remover"})
            continue
        text_path = (corrected / f"{m['name']}.txt" if m["status"] in ("text_mismatch", "ok_text_corrected")
                     else root / m["text"])
        words = label_norm_words(text_path.read_text(encoding="utf-8"))
        rgb = load_upright(root / m["image"], m["rotate_ccw"])
        binary = preprocess_image(rgb, cfg1, profile="page").image
        n_rulings = 0
        if m["group"] == "top":  # notebook pages: curved ruled lines survive Step 1
            binary, n_rulings = erase_ruled_lines(binary)
        comps, labels = components(binary)
        H, W = binary.shape
        comps = [c for c in comps if c["a"] >= 6 and c["x"] > 1 and c["y"] > 1
                 and c["x"] + c["w"] < W - 1 and c["y"] + c["h"] < H - 1]
        n_rings = 0
        if m["group"] == "top":
            comps, n_rings = remove_circled_numbers(comps)
        # two line finders (by height only / slope-tracking); with the model, keep whichever lets the
        # known words fit better (higher mean per-character log-probability)
        skipped_spans, scores, finder = 0, None, "simple"
        lines = cluster_lines_simple(comps)
        if not lines:
            res = None
        elif args.width_only:
            res = align_page(lines, words)
        else:
            best = None
            for name, ln in (("simple", lines), ("tracked", cluster_lines(comps))):
                fa = forced_align(labels, ln, words, model, vocab, rc, device)
                if fa is not None and (best is None or np.mean(fa[2]) > np.mean(best[1][2])):
                    best = (name, fa, ln)
            res = None
            if best is not None:
                finder, fa, lines = best
                res, scores, skipped_spans = fa[:2], fa[2], fa[3]
        page = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        if res is None:
            print(f"  {m['name']}: fewer ink groups than words ({len(words)}), skipped")
            pages.append({"name": m["name"], "m": m, "n_lines": len(lines), "skipped": True, "rings": n_rings,
                          "reason": "fewer ink groups than words"})
            continue
        groups, line_of = res
        (out_root / m["split"]).mkdir(parents=True, exist_ok=True)
        for k, (g, word, li) in enumerate(zip(groups, words, line_of), 1):
            x0, y0 = min(c["x"] for c in g), min(c["y"] for c in g)
            x1, y1 = max(c["x"] + c["w"] for c in g), max(c["y"] + c["h"] for c in g)
            mask = np.full(binary.shape, 255, np.uint8)
            mask[np.isin(labels, [c["i"] for c in g])] = 0
            crop = mask[max(0, y0 - 4):y1 + 4, max(0, x0 - 4):x1 + 4]
            fn = f"{m['split']}/{m['name']}_w{k:03d}.png"
            cv2.imwrite(str(out_root / fn), crop)
            crosses = len({line_of[k - 1]} | {li2 for li2, ln in enumerate(lines) if any(c in ln for c in g)}) > 1
            rows.append({"file": fn, "label": word, "name": m["name"], "writer": m["writer"], "split": m["split"],
                         "line": li + 1, "word": k, "box": f"{x0},{y0},{x1},{y1}", "crosses_lines": int(crosses),
                         "align": "width" if args.width_only else f"model/{finder}",
                         "align_score": round(scores[k - 1], 3) if scores else ""})
        page_rows = [r for r in rows if r["name"] == m["name"]]
        pages.append({"name": m["name"], "m": m, "n_lines": len(lines), "skipped": False, "rings": n_rings,
                      "page": page, "rows": page_rows})
        print(f"  {m['name']:8s} {m['split']:18s} lines {len(lines):2d}  words {len(words):3d}  "
              f"rings removed {n_rings}  rulings erased {n_rulings}  spans left out {skipped_spans}  lines by {finder}")

    # ---- quality check with the current recognizer (reads each crop; flags lines that read very badly)
    if not args.no_qc and rows:
        preds = read_crops(model, vocab, rc, device, [out_root / r["file"] for r in rows])
        for r, p in zip(rows, preds):
            r["model_read"] = p
            r["qc_cer"] = round(edit_distance(r["label"], p) / max(1, len(r["label"])), 3)
        by_line = {}
        for r in rows:
            by_line.setdefault((r["name"], r["line"]), []).append(r["qc_cer"])
        for r in rows:
            mean = float(np.mean(by_line[(r["name"], r["line"])]))
            r["line_mean_cer"] = round(mean, 3)
            r["flag"] = "line_reads_badly" if mean > LINE_FLAG_CER else ("crosses_lines" if r["crosses_lines"] else "")
            r["verified"] = 0 if r["flag"] else 1
    else:
        for r in rows:
            r.update(model_read="", qc_cer="", line_mean_cer="", flag="unchecked", verified=0)

    # ---- manual review decisions override QC (kept in a separate file so re-running never loses them)
    review_csv = paths["splits"]["handwritten_batch2"] / "words_review.csv"
    if review_csv.exists():
        decisions = {(d["name"], int(d["word"])): d for d in csv.DictReader(open(review_csv, encoding="utf-8"))}
        stale = 0
        for r in rows:
            d = decisions.get((r["name"], int(r["word"])))
            if d is None:
                continue
            if d["label"] != r["label"]:  # the crop under this decision changed: do not trust it
                stale += 1
                continue
            r["verified"] = int(d["verified"])
            r["flag"] = (r["flag"] + ";" if r["flag"] else "") + f"reviewed:{d['reviewer']}"
        print(f"applied {len(decisions) - stale} manual review decisions ({stale} stale, ignored)")

    # ---- outputs
    fields = ["file", "label", "name", "writer", "split", "line", "word", "box", "crosses_lines", "align", "align_score",
              "model_read",
              "qc_cer", "line_mean_cer", "flag", "verified"]
    words_csv = paths["splits"]["handwritten_batch2"] / "words.csv"
    with open(words_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    for pg in pages:
        if pg["skipped"]:
            continue
        img = pg["page"]
        for r in pg["rows"]:
            x0, y0, x1, y1 = map(int, r["box"].split(","))
            col = (0, 0, 255) if r["flag"] else COLORS[r["word"] % len(COLORS)]
            cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
            cv2.putText(img, str(r["word"]), (x0, max(14, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        s = min(1.0, 1400 / img.shape[1])
        cv2.imwrite(str(review / "pages" / f"{pg['name']}.png"), cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
    meta = {pg["name"]: {"n_lines": pg["n_lines"], "rings": pg.get("rings", 0), "skipped": pg["skipped"],
                         "reason": pg.get("reason", "")} for pg in pages}
    (review / "pages_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    write_html(review / "index.html", pages, rows, out_root)
    n_ok = sum(int(r["verified"]) for r in rows)
    print(f"\n{len(rows)} word crops from {sum(not p['skipped'] for p in pages)} files "
          f"({sum(p['skipped'] for p in pages)} skipped); {n_ok} pass QC, {len(rows) - n_ok} flagged for a visual check")
    print(f"words: {words_csv}\nreview page: {review / 'index.html'}")


def label_norm_words(text: str) -> list[str]:
    return [label_norm(w) for w in text.split() if label_norm(w)]


def write_html(path: Path, pages: list[dict], rows_unused: list[dict], crops_root: Path) -> None:
    """Self-contained review page. Crop thumbnails are embedded as data: URIs (Safari will not load
    file:// images from outside the page's own folder); page overviews are sibling files in pages/.
    A filter at the top shows all rows, only ticked crops (the ones to check), or only unticked ones."""
    def data_uri(rel: str) -> str:
        p = crops_root / rel
        return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode() if p.exists() else ""

    parts = ["""<!doctype html><html><head><meta charset="utf-8"><title>Batch 2 word review</title>
<style>body{font-family:-apple-system,sans-serif;margin:16px;background:#fafafa}
.page{background:#fff;border:1px solid #ddd;margin:0 0 28px;padding:12px}
.page img.full{max-width:100%;border:1px solid #eee}
table{border-collapse:collapse;margin-top:8px} td{border-bottom:1px solid #eee;padding:3px 8px;vertical-align:middle}
td.lbl,td.read{font-size:22px;direction:rtl;font-family:'Geeza Pro','Damascus',serif} td.read{color:#888;font-size:16px}
tr.flag{background:#ffecec} .crop{max-height:60px;max-width:420px;border:1px solid #ccc;background:#fff}
#bar{position:sticky;top:0;background:#222;color:#fff;padding:8px 12px;z-index:9;line-height:2}
#bar select,#bar button{font-size:15px} .hidden{display:none}</style>
</head><body><div id="bar">
Show: <select id="mode" onchange="filt()">
  <option value="all">all rows</option>
  <option value="ticked" selected>only ticked crops (to check)</option>
  <option value="unticked">only unticked / auto-excluded (pink)</option>
</select>
<label><input type="checkbox" id="pageimg" checked onchange="filt()"> page images</label>
&nbsp; <span id="count"></span> &nbsp;
<button onclick="dl()">Download words_verified.csv</button><br>
<small>Untick every crop whose image does not show exactly its label. Rows stay in place while you tick/untick;
the filter re-applies when you change it.</small></div>"""]
    for pg in pages:
        m = pg["m"]
        head = (f"<h2>{html.escape(pg['name'])} — writer {html.escape(m['writer'])}, {m['split']}, "
                f"{pg['n_lines']} lines found, {pg.get('rings', '?')} circled numbers removed</h2>")
        if pg["skipped"]:
            parts.append(f"<div class='page skipped'>{head}<p><b>Skipped:</b> {html.escape(pg.get('reason', ''))}</p></div>")
            continue
        parts.append(f"<div class='page'>{head}<img class='full' src='pages/{html.escape(pg['name'])}.png'><table>")
        for r in pg["rows"]:
            cls = " class='flag'" if str(r["verified"]) == "0" else ""  # pink = unticked (excluded)
            chk = " checked" if int(r["verified"]) else ""
            parts.append(f"<tr{cls}><td><input type='checkbox' data-file='{html.escape(r['file'])}'{chk} onchange='cnt()'></td>"
                         f"<td>{r['word']}</td><td><img class='crop' src='{data_uri(r['file'])}'></td>"
                         f"<td class='lbl'>{html.escape(r['label'])}</td><td class='read'>{html.escape(str(r['model_read']))}</td>"
                         f"<td>{html.escape(str(r['flag']))}</td></tr>")
        parts.append("</table></div>")
    parts.append("""<script>
const boxes=()=>[...document.querySelectorAll('input[data-file]')];
function cnt(){const b=boxes();
document.getElementById('count').textContent=b.filter(x=>x.checked).length+' / '+b.length+' ticked';}
function filt(){const mode=document.getElementById('mode').value, img=document.getElementById('pageimg').checked;
for(const pg of document.querySelectorAll('.page')){
  let shown=0;
  for(const tr of pg.querySelectorAll('tr')){const c=tr.querySelector('input').checked;
    const show=mode==='all'||(mode==='ticked'&&c)||(mode==='unticked'&&!c);
    tr.classList.toggle('hidden',!show); if(show) shown++;}
  const full=pg.querySelector('img.full'); if(full) full.classList.toggle('hidden',!img);
  pg.classList.toggle('hidden', mode!=='all' && (pg.classList.contains('skipped') || shown===0));}
cnt();}
function dl(){const csv='file,verified\\n'+boxes().map(x=>x.dataset.file+','+(x.checked?1:0)).join('\\n');
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));
a.download='words_verified.csv';a.click();}
filt();</script></body></html>""")
    path.write_text("\n".join(parts), encoding="utf-8")


def rebuild_html_only(paths: dict) -> None:
    """Rebuild the review page from words.csv + pages_meta.json without re-running extraction."""
    review = REPO_ROOT / "results" / "step2" / "debug" / "batch2_review"
    out_root = REPO_ROOT / "data" / "preprocessed" / "handwritten_batch2_words"
    rows = list(csv.DictReader(open(paths["splits"]["handwritten_batch2"] / "words.csv", encoding="utf-8")))
    manifest = list(csv.DictReader(open(paths["splits"]["handwritten_batch2"] / "manifest.csv", encoding="utf-8")))
    meta_path = review / "pages_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    pages = []
    for m in manifest:
        if m["split"] not in ("TRAIN", "VAL", "TEST_UNSEEN_WRITER"):
            continue
        prow = sorted([r for r in rows if r["name"] == m["name"]], key=lambda r: int(r["word"]))
        info = meta.get(m["name"], {})
        skipped = not prow
        reason = info.get("reason") or (("curved notebook ruling touches the letters; needs a better ruling remover"
                                         if m["name"] in SKIP_RULED else "no crops (see logs/step2/extract_batch2.log)")
                                        if skipped else "")
        pages.append({"name": m["name"], "m": m, "rows": prow, "skipped": skipped, "reason": reason,
                      "n_lines": info.get("n_lines", max((int(r["line"]) for r in prow), default=0)),
                      "rings": info.get("rings", "?")})
    meta = {pg["name"]: {"n_lines": pg["n_lines"], "rings": pg.get("rings", 0), "skipped": pg["skipped"],
                         "reason": pg.get("reason", "")} for pg in pages}
    (review / "pages_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    write_html(review / "index.html", pages, rows, out_root)
    size = (review / "index.html").stat().st_size / 1e6
    print(f"rebuilt {review / 'index.html'} ({size:.1f} MB, {len(rows)} crops embedded)")


if __name__ == "__main__":
    main()
