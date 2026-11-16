"""Opt-in record of the words a user reviewed in the app's Suggest mode.

Each "Apply choices" with "Save my choices for future training" ticked writes one folder
  user_feedback/<YYYYmmdd-HHMMSS>_<image name>/
    word_001.png ...   the word's crop from the cleaned page (what the recognizer read)
    choices.jsonl      one record per reviewed word
at the project root, kept apart from data/ so it can never be mixed into a dataset by accident (git-ignored).
SINDHI_OCR_FEEDBACK_DIR overrides the folder (used by the tests).

Nothing in this project reads these files. They are NOT training data until a person has checked them
(README "Saved choices"; data/PROVENANCE.md).
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from common.paths import REPO_ROOT
from pipeline.page import word_crop


def feedback_root() -> Path:
    return Path(os.environ.get("SINDHI_OCR_FEEDBACK_DIR", REPO_ROOT / "user_feedback"))


def save_feedback(binary: np.ndarray, image_name: str, page: dict, choices: dict[int, int],
                  now: datetime | None = None, root: Path | None = None) -> Path:
    """Write the reviewed words of one page (its suggestions, with the user's picks). Returns the folder."""
    now = now or datetime.now()
    stem = re.sub(r"[^\w.-]+", "_", Path(image_name).stem)[:60] or "page"
    folder = (root or feedback_root()) / f"{now:%Y%m%d-%H%M%S}_{stem}"
    k, base = 1, folder
    while folder.exists():
        k += 1
        folder = Path(f"{base}-{k}")
    folder.mkdir(parents=True)
    swallowed = set()   # a token joined into the previous word by a chosen merge: its own pick was not applied
    with open(folder / "choices.jsonl", "w", encoding="utf-8") as fh:
        for n, sg in enumerate(page["suggestions"], 1):
            pick = int(choices.get(sg["index"], 0))
            if sg["index"] in swallowed or not 0 <= pick <= len(sg["candidates"]):
                pick = 0
            if pick and sg["candidates"][pick - 1]["consumes"] == 2:
                swallowed.add(sg["index"] + 1)
            crop = f"word_{n:03d}.png"
            cv2.imwrite(str(folder / crop), word_crop(binary, sg["box"]))
            fh.write(json.dumps({
                "time": now.isoformat(timespec="seconds"), "image": image_name, "word": n, "crop": crop,
                "box": sg["box"], "recognizer_reading": sg["reading"], "next_reading": sg.get("next_reading"),
                "candidates": [c["text"] for c in sg["candidates"]],
                "chosen": sg["reading"] if pick == 0 else sg["candidates"][pick - 1]["text"],
                "kept_original": pick == 0, "words_mode": page.get("words_mode"),
            }, ensure_ascii=False) + "\n")
    return folder
