"""Step 4 comparisons on VALIDATION only (the test sets are not read here).

Variants (xlmr/correct.py; each one's weights chosen on the synthetic domain val set: best word accuracy with a
false-correction rate <= 2%, by random search + coordinate refinement):
  none                no correction
  lexicon             word list + weighted edit distance + word frequency, out-of-vocabulary tokens only
  xlmr_base           + XLM-R (as downloaded) masked-LM score; known words may change too (threshold tau_known)
  xlmr_sindhi         the same with XLM-R after further training on Sindhi text (scripts/step4_train_mlm.py)
  xlmr_sindhi_det     + the detector's flags (P(wrong) > theta)
  <best>+rules        + the Sindhi hard rules as flags (xlmr/rules.py); kept only if validation improves
Reported on: domain val (synthetic, legal + POS, 20% of words wrong), Wikipedia val (synthetic), and REAL
recognizer output on the batch-2 VAL pages (kept crops in page order; whole pages through the gap-rule boxes).

Outputs: results/step4/val_comparison.json, results/step4/val_search.csv
Usage: python scripts/step4_compare.py
"""
from __future__ import annotations

import csv
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import REPO_ROOT, load_config  # noqa: E402

_cfg = load_config("step4")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", str(_cfg["mps_memory_ratio"]))
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", str(_cfg["mps_memory_ratio"] * 0.8))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402

from cnn_transformer.train_utils import get_device  # noqa: E402
from common.gates import require_gate  # noqa: E402
from tesseract_ft.pages import label_norm_words  # noqa: E402
from xlmr.correct import Lexicon, build_positions, decide, evaluate, is_word, set_confusions, split_tail  # noqa: E402
from xlmr.lm import Detector, MaskedScorer  # noqa: E402
from xlmr.rules import check, rules, single_letter_words  # noqa: E402

SEED = 42
N_RANDOM = 300
SPACE = {"alpha": (0.0, 3.0), "beta": (0.0, 1.0), "eta": (0.0, 5.0), "tau_oov": (-3.0, 3.0),
         "gamma": (0.0, 2.0), "tau_known": (0.0, 15.0), "theta": (0.2, 0.95)}
MODELS = {"xlmr_base": _cfg["model"], "xlmr_sindhi": _cfg["mlm"]["out"] + "/best"}
MAX_FCR = _cfg["gate"]["max_false_correction_rate"]


def load_sets(split: str) -> dict[str, list[dict]]:
    d = REPO_ROOT / "data" / "splits" / "step4"
    sets = {name: [json.loads(l) for l in open(d / f"corrupted_{name}_{split}.jsonl", encoding="utf-8")]
            for name in ("domain", "wiki")}
    real = [json.loads(l) for l in open(d / f"real_{split}.jsonl", encoding="utf-8")]
    sets["real_crops"] = [r for r in real if r["kind"] == "crops"]
    sets["real_pages"] = [{k: v for k, v in r.items() if k != "origin"} for r in real if r["kind"] == "page" and r["ref"]]
    return sets


def prepare(sets: dict, lex: Lexicon, rule_fns: dict, device, cache: Path) -> dict:
    """Positions with candidates, LM scores per model, detector probabilities and rule flags (cached)."""
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    pos = {name: [build_positions(it["input"], lex) for it in items] for name, items in sets.items()}
    for mname, mdir in MODELS.items():
        scorer = MaskedScorer(REPO_ROOT / mdir, device)
        for name, items in sets.items():
            for k, (it, P) in enumerate(zip(items, pos[name])):
                scorer.score(it["input"], P)
                for p in P:
                    for c in p.cands:
                        c.__dict__.setdefault("lm_by", {})[mname] = c.lm
            print(f"  scored {name} with {mname}")
        del scorer
    det = Detector(REPO_ROOT / _cfg["detector"]["out"] / "best", device)
    for name, items in sets.items():
        probs = det.probs([it["input"] for it in items])
        for P, pr in zip(pos[name], probs):
            for p, q in zip(P, pr):
                p.det = q
                p.rule = is_word(p.core) and any(f(p.core) for f in rule_fns.values())
    cache.write_bytes(pickle.dumps(pos))
    return pos


def run(variant: str, w: dict, P_list: list, items: list) -> dict:
    model = "xlmr_sindhi" if variant.startswith("xlmr_sindhi") else ("xlmr_base" if variant.startswith("xlmr_base") else None)
    outs, acts = [], []
    for P in P_list:
        for p in P:
            p.flags = {}
            if "_det" in variant:
                p.flags["det"] = p.det > w["theta"]
            if variant.endswith("+rules"):
                p.flags["rule"] = p.rule
            for c in p.cands:
                c.lm = c.lm_by[model] if model else 0.0
        if variant == "none":
            o = [c for p in P for c in [p.cands[0].words[0] + p.tail]]
            a = ["keep"] * len(P)
        else:
            o, a = decide(P, w, flag_known=variant != "lexicon")
        outs.append(o)
        acts.append(a)
    return evaluate(items, outs, acts)


def params_for(variant: str) -> list[str]:
    keys = ["alpha", "beta", "eta", "tau_oov"]
    if variant != "lexicon":
        keys += ["gamma", "tau_known"]
    if "_det" in variant:
        keys.append("theta")
    return keys


def search(variant: str, P_list, items, rng, log: list) -> tuple[dict, dict]:
    keys = params_for(variant)

    def full(w):
        return {**{"gamma": 0.0, "tau_known": 99.0, "theta": 1.0}, **w}

    def better(e, b):
        ok_e, ok_b = e["false_correction_rate"] <= MAX_FCR, b is None or b[1]["false_correction_rate"] <= MAX_FCR
        if b is None:
            return True
        if ok_e != ok_b:
            return ok_e
        return e["word_accuracy"] > b[1]["word_accuracy"] if ok_e else e["false_correction_rate"] < b[1]["false_correction_rate"]

    best = None
    for _ in range(N_RANDOM):
        w = full({k: float(rng.uniform(*SPACE[k])) for k in keys})
        e = run(variant, w, P_list, items)
        log.append({"variant": variant, **w, **e})
        if better(e, best):
            best = (w, e)
    for _ in range(2):
        for k in keys:
            for v in np.linspace(*SPACE[k], 11):
                w = {**best[0], k: float(v)}
                e = run(variant, w, P_list, items)
                log.append({"variant": variant, **w, **e})
                if better(e, best):
                    best = (w, e)
    return best


def main() -> None:
    require_gate("step3_tesseract", allow_user_accepted=True)
    res = REPO_ROOT / "results" / "step4"
    (res / "cache").mkdir(parents=True, exist_ok=True)
    counts = {l.split("\t")[0]: int(l.split("\t")[1])
              for l in open(REPO_ROOT / "data" / "preprocessed" / "step4" / "lexicon.tsv", encoding="utf-8")}
    lex = Lexicon(counts)
    set_confusions(json.loads((res / "error_model.json").read_text(encoding="utf-8")))
    train_text = [w for w in (split_tail(x)[0] for src in ("legal_train", "pos_train", "wiki_train")
                              for l in open(REPO_ROOT / "data" / "splits" / "step4" / f"{src}.txt", encoding="utf-8")
                              for x in label_norm_words(l)) if w]
    singles = single_letter_words(train_text)
    rule_report = check(train_text, singles)
    kept_rules = {k: f for k, f in rules(singles).items() if rule_report[k]["kept"]}
    print("rules kept:", list(kept_rules))
    sets = load_sets("val")
    device = get_device("auto")
    pos = prepare(sets, lex, kept_rules, device, res / "cache" / "val_positions.pkl")

    rng = np.random.default_rng(SEED)
    log, chosen = [], {"none": ({}, run("none", {}, pos["domain"], sets["domain"]))}
    for v in ("lexicon", "xlmr_base", "xlmr_sindhi", "xlmr_sindhi_det"):
        chosen[v] = search(v, pos["domain"], sets["domain"], rng, log)
        print(f"{v:18s} domain val: {chosen[v][1]}")
    base_v = max(("xlmr_sindhi", "xlmr_sindhi_det"), key=lambda v: chosen[v][1]["word_accuracy"])
    rv = base_v + "+rules"
    chosen[rv] = search(rv, pos["domain"], sets["domain"], rng, log)
    rules_help = (chosen[rv][1]["word_accuracy"] > chosen[base_v][1]["word_accuracy"]
                  and chosen[rv][1]["false_correction_rate"] <= MAX_FCR)
    print(f"{rv:18s} domain val: {chosen[rv][1]} -> rules {'kept' if rules_help else 'not kept'}")
    final = max((v for v in chosen if v != "none" and (v != rv or rules_help)),
                key=lambda v: (chosen[v][1]["false_correction_rate"] <= MAX_FCR, chosen[v][1]["word_accuracy"]))

    table = {}
    for v, (w, _) in chosen.items():
        table[v] = {name: run(v, w, pos[name], sets[name]) for name in sets}
    out = {"selection": "weights per variant chosen on the synthetic domain val set: max word accuracy with false "
                        f"corrections <= {MAX_FCR}", "final_variant": final, "rules_kept_after_validation": rules_help,
           "rules_check_on_training_text": rule_report, "weights": {v: w for v, (w, _) in chosen.items()},
           "validation": table,
           "set_sizes": {n: {"items": len(s), "words": sum(len(it["ref"]) for it in s)} for n, s in sets.items()}}
    (res / "val_comparison.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    keys = list(dict.fromkeys(k for r in log for k in r))
    with open(res / "val_search.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys, restval="")
        wr.writeheader()
        wr.writerows(log)
    for v, row in table.items():
        print(f"{v:24s} " + "  ".join(f"{n}: acc {r['word_accuracy_before']:.3f}->{r['word_accuracy']:.3f} "
                                      f"fcr {r['false_correction_rate']:.3f}" for n, r in row.items()))
    print("final variant (validation):", final)


if __name__ == "__main__":
    main()
