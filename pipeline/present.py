"""What the web app shows, as plain functions (no Gradio here, so they can be tested): the step indicator, image
checks with friendly messages, the boxes picture, the recognizer-vs-final comparison, the summary line, the warning
banner and the review choices. The app (app.py) only arranges these; the pipeline results are not changed.
"""
from __future__ import annotations

import html

import cv2
import numpy as np

STEPS = ("Upload", "Read", "Review", "Download")
PROGRESS = {"clean": (0.05, "Cleaning page…"), "read": (0.25, "Finding words and reading…"),
            "correct": (0.70, "Correcting…"), "done": (1.0, "Done")}
MIN_SIDE = 120                       # px; smaller images cannot hold readable handwriting
IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "TIFF", "BMP", "GIF", "MPO"}
COLOURS = {"line": (40, 90, 220), "word": (240, 130, 0), "changed": (30, 160, 60)}   # RGB, as in the legend


def stepper_html(active: int, done: set[int], review_used: bool = True) -> str:
    """The 1 Upload → 2 Read → 3 Review → 4 Download indicator. `active`/`done` are step numbers (1-4); step 3 is
    shown as not needed unless the Suggest mode is in use."""
    items = []
    for n, name in enumerate(STEPS, 1):
        state = "done" if n in done else ("active" if n == active else "todo")
        if n == 3 and not review_used and state != "active":
            state, name = "skip", "Review (Suggest only)"
        mark = "✓" if state == "done" else str(n)
        items.append(f'<li class="st {state}"><span class="dot">{mark}</span><span class="nm">{html.escape(name)}</span></li>')
    return '<ol class="stepper" aria-label="Progress">' + "".join(items) + "</ol>"


def check_image(path) -> None:
    """Raise ValueError with a message for the user when the file cannot be a readable page photo."""
    from PIL import Image, UnidentifiedImageError
    if not path:
        raise ValueError("Please choose a page photo first (JPG or PNG).")
    try:
        with Image.open(path) as im:
            fmt, (w, h) = im.format, im.size
    except (UnidentifiedImageError, OSError):
        raise ValueError("This file is not an image we can open. Please upload a JPG or PNG photo of the page.")
    if fmt not in IMAGE_FORMATS:
        raise ValueError(f"{fmt} images are not supported. Please upload a JPG or PNG photo of the page.")
    if min(w, h) < MIN_SIDE:
        raise ValueError(f"This image is too small to read ({w}×{h} pixels). Please use a larger, sharper photo; "
                         f"the shorter side should be at least {MIN_SIDE} pixels.")


def draw_boxes(binary: np.ndarray, page: dict) -> np.ndarray:
    """The cleaned page with line boxes (blue) and word boxes (orange; changed by the correction: green)."""
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    t = max(2, round(max(binary.shape) / 800))
    changed = [a != "keep" for a in page["actions"]]
    k = 0
    for ln in page["lines"]:
        x0, y0, x1, y1 = ln["box"]
        cv2.rectangle(vis, (x0, y0), (x1, y1), COLOURS["line"], t)
        for w in ln["words"]:
            x0, y0, x1, y1 = w["box"]
            cv2.rectangle(vis, (x0, y0), (x1, y1), COLOURS["changed"] if changed[k] else COLOURS["word"], t)
            k += 1
    return vis


def legend_html() -> str:
    sw = "".join(f'<span class="lg"><i style="border-color: rgb{c}"></i>{name}</span>'
                 for name, c in (("text lines", COLOURS["line"]), ("words", COLOURS["word"]),
                                 ("changed words", COLOURS["changed"])))
    return f'<div class="legend">{sw}</div>'


def final_text(page: dict) -> str:
    return page["text"]


def compare_html(page: dict) -> str:
    """Recognizer output and final text side by side, right to left. Changed words are highlighted; in Suggest mode
    the words with suggestions are underlined on the recognizer side."""
    mode = page["correct_mode"]
    changed = [a != "keep" for a in page["actions"]]
    pending = {s["index"] for s in page["suggestions"]} if mode == "suggest" else set()
    before, after, k = [], [], 0
    for ln in page["lines"]:
        b = []
        for w in ln["words"]:
            cls = "chg" if changed[k] else ("sug" if k in pending else "")
            b.append(f'<span class="{cls}">{html.escape(w["text"])}</span>' if cls else html.escape(w["text"]))
            k += 1
        before.append(" ".join(b))
    k = 0
    for ln in page["lines"]:   # output words of this line, marked where the action was not "keep"
        acts = page["actions"][k:k + len(ln["words"])]
        k += len(ln["words"])
        words, j, out = ln["corrected"], 0, []
        for a in acts:
            if a == "merged":
                continue
            n = 2 if a == "split" else 1
            out += [f'<span class="chg">{html.escape(wd)}</span>' if a != "keep" else html.escape(wd)
                    for wd in words[j:j + n]]
            j += n
        out += [html.escape(wd) for wd in words[j:]]
        after.append(" ".join(out))
    right = {"off": "Final text (correction off)", "auto": "Corrected text",
             "suggest": "Final text (your choices)"}[mode]
    key = {"off": "", "auto": '<span class="chg">changed</span> by the correction',
           "suggest": '<span class="sug">has suggestions</span> · <span class="chg">changed</span> by your choices'}[mode]
    return (f'<div class="cmp"><section><h4>Recognizer output</h4><div class="sd">' + "<br>".join(before)
            + f'</div></section><section><h4>{right}</h4><div class="sd">' + "<br>".join(after)
            + "</div></section></div>" + (f'<p class="key">{key}</p>' if key else ""))


def summary_line(page: dict, seconds: float | None = None) -> str:
    parts = [f"{len(page['lines'])} line{'s' * (len(page['lines']) != 1)}",
             f"{len(page['tokens'])} word{'s' * (len(page['tokens']) != 1)}"]
    changed = sum(a != "keep" for a in page["actions"])
    if page["correct_mode"] == "auto":
        parts.append(f"{changed} changed by the correction")
    elif page["correct_mode"] == "suggest":
        parts.append(f"{len(page['suggestions'])} suggestion{'s' * (len(page['suggestions']) != 1)}")
        parts.append(f"{changed} changed by your choices")
    else:
        parts.append("correction off")
    if seconds is not None:
        parts.append(f"{seconds:.1f} s")
    return " · ".join(parts)


MODE_NAMES = {"off": "Correction off", "auto": "Auto-correct", "suggest": "Suggest"}


def top_summary(page: dict, seconds: float | None = None) -> str:
    """Results top bar: what was used, then the stats, e.g. "Auto-correct · word finding: joins · 3 lines · …"."""
    used = f"{MODE_NAMES[page['correct_mode']]} · word finding: {page.get('words_mode', 'joins')}"
    stats = summary_line(page, seconds).replace(" · correction off", "")
    return f"{used} · {stats}"


def text_rows(page: dict) -> int:
    """Height of the final-text box in rows: one per text line (at least 1), so it fits its content."""
    return max(1, page["text"].count("\n") + 1) if page["text"] else 1


def warning_html(page: dict) -> str:
    """A polite banner when the page was probably not read as text, or when nothing was found; '' otherwise."""
    if not page["tokens"]:
        return ('<div class="banner info"><b>No handwriting was found on this page.</b> Please check that the photo '
                "shows the written side, is upright, sharp and evenly lit, and try again.</div>")
    if not page["warnings"]:
        return ""
    details = "".join(f"<li>{html.escape(w)}</li>" for w in page["warnings"][:-1])
    return ('<div class="banner warn" role="alert"><b>This page may not have been read reliably.</b>'
            "<p>We found far more text lines or tiny marks than a page of handwriting usually has. This often happens "
            "with ruled or dotted paper, pencil, or a dark or blurry photo, and the text below may then be mostly "
            "guesswork.</p><p>For a better result, try:</p><ul><li>a dark pen on plain, unlined paper</li>"
            "<li>a sharp photo taken straight from above in even light</li>"
            "<li>the page upright and filling most of the photo</li></ul>"
            f"<details><summary>Details</summary><ul>{details}</ul></details></div>")


def row_choices(sg: dict) -> list[tuple[str, str]]:
    """Dropdown options of one review card: keep the reading (value "0"), then the candidates ("1".."3")."""
    opts = [(f"Keep original: {sg['reading']}", "0")]
    for n, c in enumerate(sg["candidates"], 1):
        label = f"{n}. {c['text']}"
        if c["consumes"] == 2:
            label += f"   (joined with the next word {sg.get('next_reading', '')})"
        opts.append((label, str(n)))
    return opts


def bulk_choices(page: dict | None, n_rows: int, accept: bool) -> list[str]:
    """Values for all review cards: the corrector's first candidate ("1") or keep ("0"); hidden cards keep "0"."""
    n = len(page["suggestions"]) if page and page.get("correct_mode") == "suggest" else 0
    return [("1" if accept else "0") if r < n else "0" for r in range(n_rows)]


def review_title(page: dict, max_rows: int) -> str:
    n = len(page["suggestions"])
    if not n:
        return "### Nothing to review\nThe corrector has no suggestions for this page."
    text = (f"### Review {n} suggested change{'s' * (n != 1)}\nFor each word, compare the image with the reading and "
            "pick the right text. *Keep original* leaves the recognizer's reading.")
    if n > max_rows:
        text += f"\n\nOnly the first {max_rows} are shown; the others stay as read."
    return text


EMPTY_HTML = ('<div class="empty"><div class="big">Your text will appear here</div>'
              "<p>Upload a photo of a handwritten Sindhi page, choose how to correct it, and press "
              "<b>Read page</b>. Reading takes about 10–60 seconds.</p></div>")
