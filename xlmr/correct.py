"""Spelling correction of recognizer output: candidate generation + scoring (Step 4).

For every input token (a recognized word, punctuation kept at its end):
  candidates  keep it; words from the word list within a weighted edit distance (look-alike letters, and letter
              pairs the recognizer often confuses on validation words, are cheap: see COSTS, set_confusions); split it into two known words; merge it with the next token into a known word.
  features    edit cost, log frequency of the word(s), unknown-word flag, and (LM variants) the masked-LM score:
              the span is replaced by as many <mask> tokens as the candidate has pieces, and the log-probabilities
              of the candidate's pieces are summed.
  decision    only FLAGGED tokens may change (flags: not in the word list; for LM variants also a better-scoring
              look-alike; optional detector / rule flags). score = -alpha*cost + beta*logfreq - eta*unknown + gamma*lm;
              an out-of-vocabulary token takes the best candidate if it beats keeping by tau_oov, a known word only
              if by tau_known. The weights are chosen on validation.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from xlmr.errors import same_family

TAIL = re.compile(r"^(.*?)([،؟?!.:;\"'()\-_]*)$")
COSTS = {"family_sub": 0.35, "seen_sub": 0.5, "other_sub": 1.0, "alef_indel": 0.6, "indel": 1.0, "split": 0.8,
         "merge": 0.6}
SEEN: set[tuple[str, str]] = set()   # substitutions the recognizer makes often (set_confusions)


def set_confusions(error_model: dict, min_count: float = 5) -> None:
    """Letter pairs the error model saw at least `min_count` times (either direction) become cheap substitutions."""
    SEEN.clear()
    for a, row in error_model["sub"].items():
        for b, n in row.items():
            if n >= min_count:
                SEEN.update({(a, b), (b, a)})
    wdist.cache_clear()
MAX_COST = 2.0
N_REPLACE = 10


def split_tail(tok: str) -> tuple[str, str]:
    m = TAIL.match(tok)
    return m.group(1), m.group(2)


@lru_cache(maxsize=500_000)
def wdist(a: str, b: str) -> float:
    """Weighted edit distance: substitutions inside a look-alike family are cheap, so is a missing/extra alef."""
    n, m = len(a), len(b)
    prev = [j * COSTS["indel"] for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [prev[0] + (COSTS["alef_indel"] if a[i - 1] == "ا" else COSTS["indel"])]
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                sub = 0.0
            else:
                sub = (COSTS["family_sub"] if same_family(a[i - 1], b[j - 1]) else
                       COSTS["seen_sub"] if (a[i - 1], b[j - 1]) in SEEN else COSTS["other_sub"])
            dl = COSTS["alef_indel"] if a[i - 1] == "ا" else COSTS["indel"]
            ins = COSTS["alef_indel"] if b[j - 1] == "ا" else COSTS["indel"]
            cur.append(min(prev[j] + dl, cur[j - 1] + ins, prev[j - 1] + sub))
        prev = cur
    return prev[m]


def deletes(w: str, d: int) -> set[str]:
    out, frontier = {w}, {w}
    for _ in range(d):
        frontier = {x[:i] + x[i + 1:] for x in frontier for i in range(len(x))}
        out |= frontier
    return out


class Lexicon:
    def __init__(self, counts: dict[str, int]):
        self.counts = counts
        self.index: dict[str, list[str]] = {}
        for w in counts:
            for v in deletes(w, 2 if len(w) >= 4 else 1):
                self.index.setdefault(v, []).append(w)

    def __contains__(self, w: str) -> bool:
        return w in self.counts

    def logf(self, w: str) -> float:
        return math.log1p(self.counts.get(w, 0))

    def near(self, w: str, max_cost: float = MAX_COST, n: int = N_REPLACE) -> list[tuple[str, float]]:
        cands = set()
        for v in deletes(w, 2):
            cands.update(self.index.get(v, ()))
        cands.discard(w)
        scored = [(c, wdist(w, c)) for c in cands]
        scored = [x for x in scored if x[1] <= max_cost]
        scored.sort(key=lambda x: (x[1], -self.counts[x[0]]))
        return scored[:n]


@dataclass
class Cand:
    words: list[str]          # output word(s) (cores; the tail punctuation is added back at the end)
    consumes: int             # 1, or 2 for a merge with the next token
    cost: float
    kind: str                 # keep / replace / split / merge
    logf: float = 0.0
    unknown: float = 0.0
    lm: float = 0.0


@dataclass
class Position:
    core: str
    tail: str
    oov: bool
    cands: list[Cand] = field(default_factory=list)
    flags: dict = field(default_factory=dict)


def is_word(core: str) -> bool:
    return bool(core) and any("ؠ" <= ch <= "ۿ" for ch in core) and not any(ch.isdigit() for ch in core)


def build_positions(tokens: list[str], lex: Lexicon) -> list[Position]:
    out = []
    parts = [split_tail(t) for t in tokens]
    for i, (core, tail) in enumerate(parts):
        p = Position(core, tail, oov=is_word(core) and core not in lex)
        p.cands.append(Cand([core], 1, 0.0, "keep"))
        if is_word(core):
            for c, d in lex.near(core):
                p.cands.append(Cand([c], 1, d, "replace"))
            splits = [(core[:k], core[k:]) for k in range(1, len(core)) if core[:k] in lex and core[k:] in lex]
            splits.sort(key=lambda ab: -(lex.logf(ab[0]) + lex.logf(ab[1])))
            for a, b in splits[:3]:
                p.cands.append(Cand([a, b], 1, COSTS["split"], "split"))
            if i + 1 < len(parts) and is_word(parts[i + 1][0]) and not tail:
                merged = core + parts[i + 1][0]
                opts = ([(merged, 0.0)] if merged in lex else []) + lex.near(merged, 1.0, 3)
                for c, d in opts:
                    p.cands.append(Cand([c], 2, COSTS["merge"] + d, "merge"))
        for c in p.cands:
            c.logf = sum(lex.logf(w) for w in c.words) / len(c.words)
            c.unknown = float(any(w not in lex for w in c.words if is_word(w)))
        out.append(p)
    return out


def cand_score(c: Cand, w: dict) -> float:
    return -w["alpha"] * c.cost + w["beta"] * c.logf - w["eta"] * c.unknown + w.get("gamma", 0.0) * c.lm


def ranked(positions: list[Position], i: int, w: dict) -> list[Cand]:
    """The alternatives to keeping token i, best first (ties keep candidate order; a merge needs a next token)."""
    others = [c for c in positions[i].cands[1:] if c.consumes == 1 or i + 1 < len(positions)]
    return sorted(others, key=lambda c: -cand_score(c, w))


def cand_words(positions: list[Position], i: int, c: Cand) -> list[str]:
    """A candidate's output words, with the punctuation tail of the last token it consumes put back."""
    words = list(c.words)
    words[-1] = words[-1] + positions[i + c.consumes - 1].tail
    return words


def decide(positions: list[Position], w: dict, flag_known: bool) -> tuple[list[str], list[str]]:
    """Returns output words and, per input token, the action taken ('keep', 'replace', 'split', 'merge', 'merged')."""
    out, actions, i = [], [], 0
    n = len(positions)
    while i < n:
        p = positions[i]
        keep = p.cands[0]
        flagged = p.oov or any(p.flags.values()) or flag_known
        best = keep
        if flagged and len(p.cands) > 1:
            others = ranked(positions, i, w)
            if others:
                need = w["tau_oov"] if (p.oov or any(p.flags.values())) else w["tau_known"]
                if cand_score(others[0], w) - cand_score(keep, w) > need:
                    best = others[0]
        out += cand_words(positions, i, best)
        actions.append(best.kind)
        if best.consumes == 2:
            actions.append("merged")
        i += best.consumes
    return out, actions


def suggest(tokens: list[str], positions: list[Position], w: dict, flag_known: bool, k: int = 3) -> list[dict]:
    """Suggest mode: the tokens `decide` would change, each with its top-k alternatives (best first). Nothing is
    changed. Each candidate: {'text', 'words' (output words, punctuation restored), 'kind', 'consumes'}."""
    _, actions = decide(positions, w, flag_known)
    out = []
    for i, act in enumerate(actions):
        if act in ("keep", "merged"):
            continue
        cands, seen = [], set()
        for c in ranked(positions, i, w):
            words = cand_words(positions, i, c)
            text = " ".join(words)
            if text in seen or text == tokens[i]:
                continue
            seen.add(text)
            cands.append({"text": text, "words": words, "kind": c.kind, "consumes": c.consumes})
            if len(cands) == k:
                break
        if cands:
            out.append({"index": i, "reading": tokens[i], "candidates": cands})
    return out


def apply_choices(tokens: list[str], suggestions: list[dict], choices: dict[int, int]) -> tuple[list[str], list[str]]:
    """The user's picks (token index -> 0 = keep the reading, n = n-th candidate) -> (output words, action per input
    token), in the same form as `decide`. A merge takes the next token too; any choice for that token is ignored."""
    by_index = {s["index"]: s for s in suggestions}
    out, actions, i = [], [], 0
    while i < len(tokens):
        s, pick = by_index.get(i), choices.get(i, 0)
        if s is None or not pick or pick > len(s["candidates"]) or (s["candidates"][pick - 1]["consumes"] == 2
                                                                     and i + 1 >= len(tokens)):
            out.append(tokens[i])
            actions.append("keep")
            i += 1
            continue
        c = s["candidates"][pick - 1]
        out += c["words"]
        actions.append(c["kind"])
        if c["consumes"] == 2:
            actions.append("merged")
        i += c["consumes"]
    return out, actions


def word_accuracy(refs: list[list[str]], outs: list[list[str]]) -> float:
    from common.metrics import edit_distance
    e = sum(edit_distance(r, o) for r, o in zip(refs, outs))
    return 1.0 - e / max(1, sum(len(r) for r in refs))


def clean_tokens(ref: list[str], tokens: list[str], origin: list[list[int]] | None) -> list[bool]:
    """Which input tokens are correct words. With an origin map (synthetic sets, crop sets) exactly; without one
    (whole pages) by aligning tokens to the reference words."""
    if origin is not None:
        return [len(o) == 1 and origin.count(o) == 1 and t == ref[o[0]] for t, o in zip(tokens, origin)]
    n, m = len(ref), len(tokens)
    d = np.zeros((n + 1, m + 1), dtype=int)
    d[:, 0], d[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != tokens[j - 1]))
    ok, i, j = [False] * m, n, m
    while i > 0 and j > 0:
        if d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != tokens[j - 1]):
            ok[j - 1] = ref[i - 1] == tokens[j - 1]
            i, j = i - 1, j - 1
        elif d[i, j] == d[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return ok


def evaluate(items: list[dict], outs: list[list[str]], actions: list[list[str]]) -> dict:
    """items: {'ref', 'input', 'origin'?}. Word accuracy (1 - WER), CER, and the false-correction rate: the share of
    input tokens that were already correct and got changed."""
    from common.metrics import cer
    refs = [it["ref"] for it in items]
    ins = [it["input"] for it in items]
    clean = changed = 0
    for it, act in zip(items, actions):
        ok = clean_tokens(it["ref"], it["input"], it.get("origin"))
        clean += sum(ok)
        changed += sum(o and a != "keep" for o, a in zip(ok, act))
    return {"word_accuracy_before": round(word_accuracy(refs, ins), 4), "word_accuracy": round(word_accuracy(refs, outs), 4),
            "cer_before": round(cer([" ".join(r) for r in refs], [" ".join(x) for x in ins]), 4),
            "cer": round(cer([" ".join(r) for r in refs], [" ".join(x) for x in outs]), 4),
            "clean_tokens": clean, "clean_changed": changed, "false_correction_rate": round(changed / max(1, clean), 4)}
