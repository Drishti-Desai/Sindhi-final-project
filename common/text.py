"""Sindhi text normalisation used by every step (training labels AND evaluation).

Why: the same letter can be typed with different Unicode code points. Wikipedia writes the
aspirate as گھ (U+06BE), our handwritten source texts as گه (U+0647); Persian/Arabic keyboards
produce ی / ك instead of Sindhi ي / ڪ. Without one canonical form, CER would count these as errors.
"""
from __future__ import annotations

import re
import unicodedata

_MAP = str.maketrans({
    "ھ": "ه",  # ھ heh doachashmee -> ه heh (both are written the same by hand)
    "ی": "ي",  # ی farsi yeh -> ي
    "ى": "ي",  # ى alef maksura -> ي
    "ك": "ڪ",  # ك arabic kaf -> ڪ sindhi swash kaf
    "ہ": "ه",  # ہ heh goal -> ه
    "ـ": None,      # tatweel (pure typography)
    "‌": None, "‍": None, "‏": None, "‎": None,  # zero-width / direction marks
    **{chr(0x0660 + i): str(i) for i in range(10)},  # Arabic-Indic digits -> 0-9
    **{chr(0x06F0 + i): str(i) for i in range(10)},  # Extended Arabic-Indic digits -> 0-9
})
DIACRITICS = re.compile("[ً-ْٰ]")  # harakat + superscript alef
PUNCT = re.compile(r"[،؛؟۔.,;:!?()\[\]\"'«»\-_–—/\\]")


def normalize(text: str, diacritics: bool = True) -> str:
    """Canonical form: NFC, unified letter variants, single spaces per line, stripped lines."""
    text = unicodedata.normalize("NFC", text).translate(_MAP)
    if not diacritics:
        text = DIACRITICS.sub("", text)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(l for l in lines if l)


def strip_punct(text: str) -> str:
    return " ".join(PUNCT.sub(" ", text).split())
