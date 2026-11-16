"""Step 4 final measurement on the TEST sets, run once with the variant and weights chosen on validation
(results/step4/val_comparison.json). Nothing is chosen here.

Sets:
  domain_test          synthetic, legal + POS test sentences, 20% of words wrong (the gate)
  wiki_test            synthetic, Wikipedia test sentences (non-legal text)
  unseen_crops         REAL: the 144 verified crops of the unseen writer (purple pen), read by the Step 2
                       recognizer, in page order (the main result; Step 2 reference: 71.5% words correct)
  unseen_pages         REAL: the same pages read through the Step 5 gap-rule boxes (verified part of each page)
  hj_crops             REAL: the 56 verified crops of TESS_TEST pages H1, J2, J3 (writers seen in training)
  hj_pages             REAL: those pages through the gap-rule boxes
Reference only (not a re-selection): XLM-R as downloaded (xlmr_base), with its own validation weights.

Outputs: results/step4/test_metrics.json, results/step4/test_real_changes.csv, results/gates/step4_xlmroberta.json
Usage: python scripts/step4_test.py
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import REPO_ROOT, load_config, load_paths  # noqa: E402


def load_module(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


compare = load_module("step4_compare", "scripts/step4_compare.py")   # sets the MPS memory limits first
prep = load_module("step4_prepare", "scripts/step4_prepare.py")

from cnn_transformer.train_utils import get_device  # noqa: E402
from cnn_transformer.words import fixed_set, read_real  # noqa: E402
from common.gates import require_gate, write_gate  # noqa: E402
from tesseract_ft.pages import load_batch2_pages, load_tess_test_pages  # noqa: E402
from xlmr.correct import Lexicon, set_confusions  # noqa: E402

REFERENCE = "xlmr_base"


def crop_items(rec, device) -> dict[str, list[dict]]:
    """Recognizer readings of the verified test crops, grouped per page in page order (the Step 2 test sets,
    same seeds)."""
    model, vocab, rc = rec
    rows = [r for r in csv.DictReader(open(REPO_ROOT / "data" / "splits" / "handwritten_batch2" / "words.csv",
                                           encoding="utf-8")) if r["split"] == "TEST_UNSEEN_WRITER" and r["verified"] == "1"]
    b2 = [(REPO_ROOT / "data" / "preprocessed" / "handwritten_batch2_words" / r["file"], prep.label_norm(r["label"]))
          for r in rows]
    verified = REPO_ROOT / "data" / "splits" / "step2_real_words" / "verified.csv"
    hj = read_real(verified, load_paths()["preprocessed"]["handwritten"].parent / "handwritten_words", "TESS_TEST")
    hj_pages = [r["page"][:-4] for r in csv.DictReader(open(verified, encoding="utf-8"))
                if r["split"] == "TESS_TEST" and r["verified"] == "1"]
    out = {}
    for name, pairs, pages, seed in (("unseen_crops", b2, [r["name"] for r in rows], 7), ("hj_crops", hj, hj_pages, 5)):
        preds, _ = prep.step2.predict(model, fixed_set("real", pairs, None, rc, 0, seed), vocab, device)
        items = []
        for page in dict.fromkeys(pages):
            ks = [i for i, p in enumerate(pages) if p == page]
            items.append({"kind": "crops", "page": page, "ref": [pairs[i][1] for i in ks], "input": [preds[i] for i in ks],
                          "origin": [[j] for j in range(len(ks))]})
        out[name] = items
    return out


def main() -> None:
    require_gate("step3_tesseract", allow_user_accepted=True)
    cfg = load_config("step4")
    res = REPO_ROOT / "results" / "step4"
    val = json.loads((res / "val_comparison.json").read_text(encoding="utf-8"))
    final = val["final_variant"]
    weights = {v: val["weights"][v] for v in (final, REFERENCE, "lexicon")}
    counts = {l.split("\t")[0]: int(l.split("\t")[1])
              for l in open(REPO_ROOT / "data" / "preprocessed" / "step4" / "lexicon.tsv", encoding="utf-8")}
    lex = Lexicon(counts)
    set_confusions(json.loads((res / "error_model.json").read_text(encoding="utf-8")))

    d = REPO_ROOT / "data" / "splits" / "step4"
    sets = {f"{n}_test": [json.loads(l) for l in open(d / f"corrupted_{n}_test.jsonl", encoding="utf-8")]
            for n in ("domain", "wiki")}
    device = get_device("auto")
    rec = prep.step3.load_recognizer(device)
    sets.update(crop_items(rec, device))
    box_cfg = load_config("step3")["box_finder"]
    for name, pages in (("unseen_pages", load_batch2_pages(("TEST_UNSEEN_WRITER",))), ("hj_pages", load_tess_test_pages())):
        sets[name] = [it for it in (prep.page_item(rec, p, box_cfg, device) for p in pages) if it["ref"]]
    del rec

    pos = compare.prepare(sets, lex, {}, device, res / "cache" / "test_positions.pkl")
    variants = ["none", "lexicon", REFERENCE, final]
    table = {v: {n: compare.run(v, weights.get(v, {}), pos[n], sets[n]) for n in sets} for v in variants}
    # H and J separately (crops)
    for v in variants:
        for wr in ("H", "J"):
            idx = [i for i, it in enumerate(sets["hj_crops"]) if it["page"].startswith(wr)]
            table[v][f"hj_crops_{wr}"] = compare.run(v, weights.get(v, {}), [pos["hj_crops"][i] for i in idx],
                                                     [sets["hj_crops"][i] for i in idx])

    # every change the final variant makes on the real crops
    rows = []
    for n in ("unseen_crops", "hj_crops"):
        compare.run(final, weights[final], pos[n], sets[n])
        for it, P in zip(sets[n], pos[n]):
            o, a = compare.decide(P, weights[final], flag_known=True)
            j = 0
            for k, (tok, act) in enumerate(zip(it["input"], a)):
                if act == "keep":
                    j += 1
                    continue
                if act == "merged":
                    continue
                new = " ".join(o[j:j + 1]) if act != "split" else " ".join(o[j:j + 2])
                rows.append({"set": n, "page": it["page"], "action": act, "input": tok, "output": new,
                             "reference": it["ref"][k] if k < len(it["ref"]) else "",
                             "was_correct": tok == it["ref"][k], "now_correct": new == it["ref"][k]})
                j += 2 if act == "split" else 1
    with open(res / "test_real_changes.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["set", "page", "action", "input", "output", "reference", "was_correct",
                                            "now_correct"])
        wr.writeheader()
        wr.writerows(rows)

    sizes = {n: {"items": len(s), "words": sum(len(it["ref"]) for it in s)} for n, s in sets.items()}
    out = {"rules": "variant and weights chosen on validation (val_comparison.json); test measured once",
           "final_variant": final, "reference_only": REFERENCE, "weights": weights, "set_sizes": sizes, "test": table,
           "step2_reference_unseen_crops": {"word_accuracy": 0.7153, "one_minus_cer": 0.9119}}
    (res / "test_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    g = cfg["gate"]
    dom = table[final]["domain_test"]
    gate = write_gate("step4_xlmroberta", {
        "domain_test_word_accuracy": {"value": dom["word_accuracy"], "threshold": g["min_word_accuracy"], "op": ">="},
        "domain_test_false_correction_rate": {"value": dom["false_correction_rate"],
                                              "threshold": g["max_false_correction_rate"], "op": "<="}},
        extra={"final_variant": final, "chosen_on": "synthetic domain val (val_comparison.json)",
               "main_result_real_handwriting": {n: table[final][n] for n in
                                                ("unseen_crops", "unseen_pages", "hj_crops", "hj_crops_H",
                                                 "hj_crops_J", "hj_pages")},
               "reference_only_xlmr_base": table[REFERENCE], "wiki_test": table[final]["wiki_test"],
               "known_limitation": "weights tuned on legal + POS text change too many correct words on Wikipedia "
                                   "(validation 4.1% > 2%); a stricter 1% setting is a future option"})
    for v, row in table.items():
        print(f"{v:18s} " + "  ".join(f"{n}: {r['word_accuracy_before']:.3f}->{r['word_accuracy']:.3f} "
                                      f"cer {r['cer_before']:.3f}->{r['cer']:.3f} chg {r['clean_changed']}/{r['clean_tokens']}"
                                      for n, r in row.items()))
    print(f"[gate] step4_xlmroberta: {gate['status']}")


if __name__ == "__main__":
    main()
