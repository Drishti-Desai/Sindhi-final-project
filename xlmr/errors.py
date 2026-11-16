"""A model of the recognizer's errors, and the frozen corruption used for Step 4's synthetic sets.

Character confusions come from the fine-tuned recognizer's readings of validation words it was never trained on:
the Step 2 composed_val set (600 words built from validation letter crops, same seed as Step 2) and the 71 batch-2
VAL crops. They are smoothed with a hand-made table of look-alike letters (VISUAL_FAMILIES). Word merges and splits
come from the Step 3 gap rule on the tuning pages.

Corruption: exactly `rate` of the words of a set are made wrong: a merge with the next word (both count), a split, or
1-3 character edits sampled from the error model (the word must actually change).
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np

# letters that differ only by dots or a small mark: the recognizer's typical confusions
VISUAL_FAMILIES = ["بٻڀپتٿٽٺثن", "جڄڃچڇحخ", "دڊڌڍڏذ", "رڙزژ", "سش", "صض", "طظ", "عغ", "فڦق", "ڪکگڳڱ", "ڻن",
                   "هة", "يئ", "اآ", "وؤ"]
FAMILY = {ch: k for k, fam in enumerate(VISUAL_FAMILIES) for ch in fam}


def same_family(a: str, b: str) -> bool:
    return a in FAMILY and FAMILY.get(a) == FAMILY.get(b)


def align_ops(ref: str, hyp: str) -> list[tuple[str, str, str]]:
    """Levenshtein alignment -> list of ('sub', a, b) / ('del', a, '') / ('ins', '', b) edits."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=int)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                ops.append(("sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            ops.append(("del", ref[i - 1], ""))
            i -= 1
        else:
            ops.append(("ins", "", hyp[j - 1]))
            j -= 1
    return ops[::-1]


class ErrorModel:
    def __init__(self, pairs: list[tuple[str, str]], merge_rate: float, split_rate: float, smoothing: float = 0.5):
        """pairs: (true word, recognizer reading). Rates are per true word."""
        self.sub = collections.defaultdict(collections.Counter)
        self.dele = collections.Counter()
        self.ins = collections.Counter()
        self.char_count = collections.Counter()
        self.edits_per_wrong_word = collections.Counter()
        for ref, hyp in pairs:
            self.char_count.update(ref)
            if ref == hyp:
                continue
            ops = align_ops(ref, hyp)
            self.edits_per_wrong_word[min(len(ops), 3)] += 1
            for op, a, b in ops:
                if op == "sub":
                    self.sub[a][b] += 1
                elif op == "del":
                    self.dele[a] += 1
                else:
                    self.ins[b] += 1
        for fam in VISUAL_FAMILIES:   # smoothing: every look-alike pair gets a small pseudo-count
            for a in fam:
                for b in fam:
                    if a != b:
                        self.sub[a][b] += smoothing
        self.word_error_rate = sum(self.edits_per_wrong_word.values()) / max(1, len(pairs))
        self.merge_rate, self.split_rate = merge_rate, split_rate
        n_sub = sum(sum(c.values()) for c in self.sub.values())
        tot = n_sub + sum(self.dele.values()) + sum(self.ins.values())
        self.op_p = {"sub": n_sub / tot, "del": sum(self.dele.values()) / tot, "ins": sum(self.ins.values()) / tot}

    def summary(self) -> dict:
        top = sorted(((n, a, b) for a, c in self.sub.items() for b, n in c.items()), reverse=True)[:15]
        return {"word_error_rate": round(self.word_error_rate, 4), "merge_rate": round(self.merge_rate, 4),
                "split_rate": round(self.split_rate, 4), "op_share": {k: round(v, 3) for k, v in self.op_p.items()},
                "edits_per_wrong_word": dict(self.edits_per_wrong_word),
                "top_substitutions": [f"{a}->{b} {n:g}" for n, a, b in top],
                "top_deletions": [f"{a} {n}" for a, n in self.dele.most_common(8)],
                "top_insertions": [f"{b} {n}" for b, n in self.ins.most_common(8)]}

    def char_edit(self, word: str, rng) -> str:
        k = rng.choice(list(self.edits_per_wrong_word), p=np.array(list(self.edits_per_wrong_word.values()), float)
                       / sum(self.edits_per_wrong_word.values()))
        w = list(word)
        for _ in range(int(k)):
            op = rng.choice(list(self.op_p), p=list(self.op_p.values()))
            if op == "sub" and w:
                weights = np.array([sum(self.sub[c].values()) + 1e-3 for c in w])
                i = rng.choice(len(w), p=weights / weights.sum())
                c = self.sub[w[i]]
                if c:
                    w[i] = rng.choice(list(c), p=np.array(list(c.values()), float) / sum(c.values()))
            elif op == "del" and len(w) > 1:
                weights = np.array([self.dele[c] + 0.1 for c in w])
                del w[rng.choice(len(w), p=weights / weights.sum())]
            elif self.ins:
                ch = rng.choice(list(self.ins), p=np.array(list(self.ins.values()), float) / sum(self.ins.values()))
                w.insert(int(rng.integers(len(w) + 1)), ch)
        return "".join(w)

    def corrupt(self, sentences: list[list[str]], rate: float, seed: int) -> tuple[list[list[str]], list[list[int]]]:
        """Corrupt exactly `rate` of all words. Returns the corrupted token lists and, per sentence, a map
        corrupted-token -> list of original word indices it came from (for scoring)."""
        rng = np.random.default_rng(seed)
        slots = [(si, wi) for si, s in enumerate(sentences) for wi in range(len(s))]
        target = int(round(rate * len(slots)))
        p_merge = self.merge_rate / (self.merge_rate + self.split_rate + self.word_error_rate)
        p_split = self.split_rate / (self.merge_rate + self.split_rate + self.word_error_rate)
        kind: dict[tuple[int, int], str] = {}
        done = 0
        for si, wi in (slots[i] for i in rng.permutation(len(slots))):
            if done >= target or (si, wi) in kind:
                continue
            r = rng.random()
            if r < p_merge and wi + 1 < len(sentences[si]) and (si, wi + 1) not in kind and done + 2 <= target:
                kind[(si, wi)], kind[(si, wi + 1)] = "merge_head", "merge_tail"
                done += 2
            elif r < p_merge + p_split and len(sentences[si][wi]) >= 4:
                kind[(si, wi)] = "split"
                done += 1
            else:
                kind[(si, wi)] = "char"
                done += 1
        out, origin = [], []
        for si, s in enumerate(sentences):
            toks, org = [], []
            for wi, w in enumerate(s):
                k = kind.get((si, wi))
                if k is None:
                    toks.append(w)
                    org.append([wi])
                elif k == "merge_head":
                    toks.append(w + s[wi + 1])
                    org.append([wi, wi + 1])
                elif k == "merge_tail":
                    continue
                elif k == "split":
                    cut = int(rng.integers(1, len(w)))
                    toks += [w[:cut], w[cut:]]
                    org += [[wi], [wi]]
                else:
                    new = w
                    for _ in range(20):
                        new = self.char_edit(w, rng)
                        if new != w and new:
                            break
                    toks.append(new)
                    org.append([wi])
            out.append(toks)
            origin.append(org)
        return out, origin

    @classmethod
    def load(cls, path: Path) -> "ErrorModel":
        d = json.loads(path.read_text(encoding="utf-8"))
        em = cls.__new__(cls)
        em.sub = collections.defaultdict(collections.Counter, {a: collections.Counter(r) for a, r in d["sub"].items()})
        em.dele, em.ins = collections.Counter(d["del"]), collections.Counter(d["ins"])
        sm = d["summary"]
        em.edits_per_wrong_word = collections.Counter({int(k): v for k, v in sm["edits_per_wrong_word"].items()})
        em.word_error_rate, em.merge_rate, em.split_rate = sm["word_error_rate"], sm["merge_rate"], sm["split_rate"]
        em.op_p = sm["op_share"]
        em.char_count = collections.Counter()
        return em

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"sub": {a: dict(c) for a, c in self.sub.items()}, "del": dict(self.dele),
                                    "ins": dict(self.ins), "summary": self.summary()}, indent=1, ensure_ascii=False),
                        encoding="utf-8")
