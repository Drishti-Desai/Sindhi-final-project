"""Step 4 data: text splits, word list, recognizer error model, frozen corrupted sets, real validation readings.

1. Splits (xlmr/data.py): legal CSV / POS / Wikipedia, test and FINAL_TEST sentences removed.
   -> data/splits/step4/<source>_<split>.txt
2. Word list from the TRAIN split only -> data/preprocessed/step4/lexicon.tsv (Wikipedia words need 3+ occurrences).
3. Error model (xlmr/errors.py) from the fine-tuned recognizer's readings of validation words it never trained on
   (Step 2 composed_val, rebuilt with the same seed, and the 71 batch-2 VAL crops); merge/split rates of the Step 3
   gap rule on the tuning pages. -> results/step4/error_model.json
4. Frozen corrupted sets, exactly 20% of words wrong: domain = legal + POS, and Wikipedia, for val and test.
   -> data/splits/step4/corrupted_<set>_<split>.jsonl  (the test files are written now and only read at the end)
5. Real validation: the recognizer's readings of the batch-2 VAL pages (kept crops in page order, and the gap-rule
   boxes of Step 5). -> data/splits/step4/real_val.jsonl
Nothing is trained here.
Usage: python scripts/step4_prepare.py
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cnn_transformer.train_utils import get_device  # noqa: E402
from cnn_transformer.words import LetterBank, fixed_set, label_norm  # noqa: E402
from common.gates import require_gate  # noqa: E402
from common.paths import REPO_ROOT, load_config, load_paths, load_yaml  # noqa: E402
from tesseract_ft.boxes import own_layout  # noqa: E402
from tesseract_ft.pages import label_norm_words, load_batch2_pages  # noqa: E402
from xlmr.data import build_splits, lexicon_from  # noqa: E402
from xlmr.errors import ErrorModel  # noqa: E402


def load_module(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


step2 = load_module("step2_train_recognizer", "scripts/step2_train_recognizer.py")
step3 = load_module("step3_baseline", "scripts/step3_baseline.py")
RATE = 0.20


def recognizer_val_pairs(paths, rec, device) -> tuple[list, dict]:
    """(true word, reading) for Step 2's composed_val (seed 2, 600 words) and the 71 batch-2 VAL crops."""
    model, vocab, rc = rec
    words = step2.read_words(REPO_ROOT / "data" / "splits" / "step2_words" / "val.txt")
    idx = list(csv.DictReader(open(paths["preprocessed"]["letters52"] / "index.csv", encoding="utf-8")))
    man = list(csv.DictReader(open(paths["splits"]["letters52"] / "manifest.csv", encoding="utf-8")))
    for m in man:
        m["h"] = cv2.imread(str(paths["raw"]["letters52"] / m["relpath"]), cv2.IMREAD_GRAYSCALE).shape[0]
    lmap = {int(k): v for k, v in load_yaml(paths["configs"]["letter_map"])["classes"].items()}
    bank = LetterBank(paths["preprocessed"]["letters52_gray"], idx, lmap, man, "val", gray=True)
    composed = fixed_set("composed", [w for w in words if bank.tokenize(w)], bank, rc, rc["val_size"], 2)
    b2_val, _ = step2.read_batch2("VAL")
    real = fixed_set("real", b2_val, None, rc, 0, 6)
    out, n = [], {}
    for name, items in (("composed_val", composed), ("real_val", real)):
        preds, _ = step2.predict(model, items, vocab, device)
        out += [(label_norm(l), p) for (_, l), p in zip(items, preds)]
        n[name] = len(items)
    return out, n


def page_item(rec, p, box_cfg: dict, device) -> dict:
    """A page read through the Step 5 gap-rule boxes. Some page texts cover only part of the page (as in Step 3): the
    reference is the verified words in reading order, and only boxes whose ink lies mostly on verified word boxes
    are read."""
    ink = p.binary < 128
    gt_ink = np.zeros_like(ink)
    for x0, y0, x1, y1 in p.gt_words:
        gt_ink[y0:y1, x0:x1] = True
    lines = step3.reading_order(own_layout(p.binary, box_cfg["line_finder"], box_cfg["gap_h"], box_cfg["gap_med"]))
    flat = [x for ln in lines for x in ln if step3.tighten(x, ink) is not None
            and gt_ink[x[1]:x[3], x[0]:x[2]][ink[x[1]:x[3], x[0]:x[2]]].mean() > 0.5]
    hyp = [t for t in step3.read_boxes(rec, p, flat, device) if t] if flat else []
    ref = [w for l in p.gt_labels for w in label_norm(l).split()]
    return {"kind": "page", "page": p.name, "ref": ref, "input": hyp}


def main() -> None:
    require_gate("step3_tesseract", allow_user_accepted=True)
    paths = load_paths()
    split_dir = REPO_ROOT / "data" / "splits" / "step4"
    pre_dir = REPO_ROOT / "data" / "preprocessed" / "step4"
    res_dir = REPO_ROOT / "results" / "step4"
    for d in (split_dir, pre_dir, res_dir):
        d.mkdir(parents=True, exist_ok=True)
    report = {}

    # 1. splits
    splits, stats = build_splits(paths)
    for src, d in splits.items():
        for sp, sents in d.items():
            (split_dir / f"{src}_{sp}.txt").write_text("\n".join(sents) + "\n", encoding="utf-8")
    report["splits"] = {src: {sp: {"sentences": len(v), "words": sum(len(s.split()) for s in v)} for sp, v in d.items()}
                        for src, d in splits.items()}
    report["split_stats"] = stats
    print(json.dumps(report["splits"]))

    # 2. word list (train only)
    lex = lexicon_from(splits["legal"]["train"] + splits["pos"]["train"])
    for w, c in lexicon_from(splits["wiki"]["train"], min_count=3).items():
        lex[w] = lex.get(w, 0) + c
    with open(pre_dir / "lexicon.tsv", "w", encoding="utf-8") as fh:
        for w, c in sorted(lex.items(), key=lambda x: -x[1]):
            fh.write(f"{w}\t{c}\n")
    report["lexicon_words"] = len(lex)

    # 3. error model
    device = get_device("auto")
    rec = step3.load_recognizer(device)
    pairs, n_pairs = recognizer_val_pairs(paths, rec, device)
    lm = json.loads((paths["results"]["step3"] / "levers_metrics.json").read_text(encoding="utf-8"))
    own = lm["in_sample"]["baseline_own"]["all_tuning"]
    em = ErrorModel(pairs, merge_rate=own["merged"] / own["words"], split_rate=own["split"] / own["words"])
    em.save(res_dir / "error_model.json")
    report["error_model"] = {"pairs": n_pairs, **em.summary()}
    print(json.dumps(report["error_model"], ensure_ascii=False))

    # 4. frozen corrupted sets
    for setname, sources in (("domain", ("legal", "pos")), ("wiki", ("wiki",))):
        for k, sp in enumerate(("val", "test")):
            sents = [label_norm_words(s) for src in sources for s in splits[src][sp]]
            sents = [s for s in sents if s]
            toks, origin = em.corrupt(sents, RATE, seed=100 + 10 * k + len(setname))
            with open(split_dir / f"corrupted_{setname}_{sp}.jsonl", "w", encoding="utf-8") as fh:
                for ref, t, o in zip(sents, toks, origin):
                    fh.write(json.dumps({"ref": ref, "input": t, "origin": o}, ensure_ascii=False) + "\n")
            wrong = sum(len({i for t, o in zip(s_t, s_o) if len(o) > 1 or t != s_r[o[0]] or s_o.count(o) > 1 for i in o})
                        for s_t, s_o, s_r in zip(toks, origin, sents))   # distinct true words made wrong
            report[f"corrupted_{setname}_{sp}"] = {"sentences": len(sents), "words": sum(map(len, sents)),
                                                   "words_made_wrong": wrong}
    # 5. real validation readings (batch-2 VAL pages; never trained on by the recognizer)
    rows = [r for r in csv.DictReader(open(paths["splits"]["handwritten_batch2"] / "words.csv", encoding="utf-8"))
            if r["split"] == "VAL" and r["verified"] == "1"]
    crops_root = REPO_ROOT / "data" / "preprocessed" / "handwritten_batch2_words"
    model, vocab, rc = rec
    items = fixed_set("real", [(crops_root / r["file"], label_norm(r["label"])) for r in rows], None, rc, 0, 6)
    preds, _ = step2.predict(model, items, vocab, device)
    box_cfg = load_config("step3")["box_finder"]
    with open(split_dir / "real_val.jsonl", "w", encoding="utf-8") as fh:
        for name in dict.fromkeys(r["name"] for r in rows):   # kept crops in page order: word-aligned
            ks = [i for i, r in enumerate(rows) if r["name"] == name]
            fh.write(json.dumps({"kind": "crops", "page": name, "ref": [label_norm(rows[i]["label"]) for i in ks],
                                 "input": [preds[i] for i in ks], "origin": [[j] for j in range(len(ks))]},
                                ensure_ascii=False) + "\n")
        for p in load_batch2_pages(("VAL",)):                  # gap-rule boxes: verified part of the page, not word-aligned
            fh.write(json.dumps(page_item(rec, p, box_cfg, device), ensure_ascii=False) + "\n")
    report["real_val"] = {"crops": len(rows), "pages": len({r['name'] for r in rows})}
    (res_dir / "prepare_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k.startswith("corrupted") or k in ("lexicon_words", "real_val")}))


if __name__ == "__main__":
    main()
