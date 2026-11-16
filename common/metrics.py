"""CER / WER, identical for every step.

CER = (substitutions + deletions + insertions) / number of reference characters
WER = the same over whitespace-separated words
Text accuracy is reported as 1 - CER (can be negative if the output is much longer than the reference).
"""
from __future__ import annotations


def edit_distance(ref, hyp) -> int:
    """Levenshtein distance between two sequences (characters or word lists)."""
    if len(ref) < len(hyp):
        ref, hyp = hyp, ref
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def cer(refs: list[str], hyps: list[str]) -> float:
    """Corpus-level CER: total edits / total reference characters (spaces count as characters)."""
    edits = sum(edit_distance(r, h) for r, h in zip(refs, hyps))
    return edits / max(1, sum(len(r) for r in refs))


def wer(refs: list[str], hyps: list[str]) -> float:
    edits = sum(edit_distance(r.split(), h.split()) for r, h in zip(refs, hyps))
    return edits / max(1, sum(len(r.split()) for r in refs))


def exact_match(refs: list[str], hyps: list[str]) -> float:
    return sum(r == h for r, h in zip(refs, hyps)) / max(1, len(refs))
