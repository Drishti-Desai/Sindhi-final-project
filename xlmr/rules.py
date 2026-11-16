"""Sindhi hard rules, used ONLY to flag a token for correction (never to change a word by themselves).

Each rule was checked against the Step 4 training text (4.0M words): a rule that correct words break more than
MAX_VIOLATION of the time is not used (scripts/step4_compare.py reports the rate per rule).
- ھ (heh doachashmee) rules are not possible here: the shared normalisation (common/text.py) maps ھ to ه.
- Dropped after the check: "no digits mixed with letters" (years are written 1947ع, 0.8% of training words).
- Refined after the check: R6 (a lone ء is a common word), R7 (letters only: "2000" is fine), R9 (only punctuation
  BETWEEN letters; a leading bracket is fine), R10 (allowed one-letter words = those seen 20+ times in training text).
"""
from __future__ import annotations

import collections
import re

MAX_VIOLATION = 0.0005   # 1 in 2,000 correct training words
LETTER = "ؠ-ۿ"
INNER_PUNCT = re.compile(f"[{LETTER}][،؟?!.:;\"'()\\-_][{LETTER}]")
TRIPLE = re.compile(f"([{LETTER}])\\1\\1")


def single_letter_words(words: list[str], min_count: int = 20) -> set[str]:
    c = collections.Counter(w for w in words if len(w) == 1)
    return {w for w, n in c.items() if n >= min_count and "ؠ" <= w <= "ۿ"}


def rules(singles: set[str]) -> dict:
    """name -> test(core) returning True when the word breaks the rule."""
    return {
        "R1 ۾ only as a whole word": lambda w: "۾" in w and w != "۾",
        "R2 ۽ only as a whole word": lambda w: "۽" in w and w != "۽",
        "R3 no word starts with ڻ": lambda w: w.startswith("ڻ"),
        "R4 no word starts with ڙ": lambda w: w.startswith("ڙ"),
        "R5 no word starts with ڱ": lambda w: w.startswith("ڱ"),
        "R6 no longer word starts with ء": lambda w: w.startswith("ء") and len(w) > 1,
        "R7 no letter three times in a row": lambda w: TRIPLE.search(w) is not None,
        "R9 no punctuation between letters": lambda w: INNER_PUNCT.search(w) is not None,
        "R10 one-letter words only from the training list": lambda w: len(w) == 1 and "ؠ" <= w <= "ۿ"
                                                            and w not in singles,
    }


def check(words: list[str], singles: set[str]) -> dict:
    """Violation rate of each rule on correct (training) words; which rules are kept."""
    out = {}
    for name, f in rules(singles).items():
        bad = [w for w in words if f(w)]
        rate = len(bad) / max(1, len(words))
        out[name] = {"violations": len(bad), "rate": round(rate, 6), "kept": rate <= MAX_VIOLATION,
                     "examples": [w for w, _ in collections.Counter(bad).most_common(4)]}
    return out
