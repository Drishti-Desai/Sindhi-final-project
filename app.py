"""Local web frontend for the Sindhi OCR pipeline (Gradio). Same code as run_pipeline.py.

Flow: 1 Upload → 2 Read → 3 Review (Suggest mode only) → 4 Download, on two screens:
- Screen 1 "Set up": upload a page photo (or "Try an example": a training page), choose the correction mode and press Read page.
  Correction: Off (recognizer only) · Auto-correct (default) · Suggest (you choose every change). Word finding
  (joins | gaps) is under Advanced settings.
- Screen 2 "Results" (screen 1 is hidden): "← Read another page" and a summary of the settings and stats; the final
  text sized to its content (copy, .txt download); in Suggest mode the review cards; the cleaned page and the photo
  large (zoom, boxes toggle and colour legend); the recognizer-vs-final comparison; a warning banner when the page
  was probably not read as text. A new upload or "Read another page" clears every earlier result, and Apply / the
  boxes toggle refuse to act on a page read from a different photo.
- Suggest mode: one card per word the corrector would change (its image, the reading, keep / top-3 candidates),
  "Accept all suggestions" / "Keep all originals", and "Apply choices". The opt-in "save my choices" writes the
  reviewed words to user_feedback/ (pipeline/feedback.py); nothing reads that folder and it is never used for training
  automatically.

What is shown is built by pipeline/present.py. Runs on this computer only (127.0.0.1, no share link, Gradio analytics
off, no fonts or files fetched from the internet). One page at a time; the models are loaded one after another per
page within the same memory cap as run_pipeline.py. Applying choices runs no model.

Usage: .venv/bin/python app.py [--port 7860]      then open http://127.0.0.1:7860   (stop: Ctrl+C)
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_pipeline  # noqa: E402  (sets the MPS memory limits first)

import cv2  # noqa: E402
import gradio as gr  # noqa: E402

from cnn_transformer.train_utils import get_device  # noqa: E402
from common.gates import require_gate  # noqa: E402
from common.paths import REPO_ROOT  # noqa: E402
from pipeline import present  # noqa: E402
from pipeline.feedback import feedback_root, save_feedback  # noqa: E402
from pipeline.page import apply_page_choices, clean_page, load_rgb, word_crop  # noqa: E402

DEVICE = None
MAX_ROWS = 50          # review cards prepared in the page; more suggestions than this are left unchanged
EXAMPLE = REPO_ROOT / "data" / "raw" / "handwritten" / "images" / "A1.png"   # TESS_TRAIN: a training page
MODES = [("Off: the recognizer's text only", "off"),
         ("Auto-correct: fix likely mistakes automatically (recommended)", "auto"),
         ("Suggest: show the fixes and let me choose each one", "suggest")]
WORDS = [("Joins (default): the recognizer decides which ink pieces form a word; best for cramped writing", "joins"),
         ("Gaps: cut words at wide spaces; best when the writer leaves clear gaps between words", "gaps")]

SYSTEM_FONT = ["-apple-system", "BlinkMacSystemFont", "Segoe UI", "Helvetica Neue", "Arial", "sans-serif"]
THEME = gr.themes.Soft(primary_hue="blue", secondary_hue="sky", neutral_hue="slate", radius_size="lg",
                       text_size="md", spacing_size="md", font=SYSTEM_FONT,
                       font_mono=["SF Mono", "Menlo", "Consolas", "monospace"]).set(
    body_background_fill="#f4f6f9", body_background_fill_dark="#0f1218",
    block_background_fill="#ffffff", block_background_fill_dark="#171b23",
    block_border_width="1px", block_shadow="0 1px 3px rgba(16, 24, 40, 0.06)", block_shadow_dark="none",
    button_large_padding="14px 22px", button_large_text_size="18px",
)
CSS = """
:root { --sd-font: "SF Arabic", "Geeza Pro", "Damascus", "Noto Naskh Arabic", "Segoe UI", serif;
        --ink: #1f4fbf; --chg-bg: #ffe58a; --chg-fg: #3b2f00; --sug: #2f6fd6;
        --warn-bg: #fff4e5; --warn-border: #e8a33d; --warn-fg: #5c3a00;
        --info-bg: #eef4ff; --info-border: #8fb0ef; --info-fg: #17346e; }
.dark { --ink: #8fb0f5; --chg-bg: #6b5a12; --chg-fg: #fff6cf; --sug: #8fb0f5;
        --warn-bg: #3a2a10; --warn-border: #c98a2b; --warn-fg: #ffe2b0;
        --info-bg: #17243d; --info-border: #3e5d9c; --info-fg: #d6e3ff; }
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }
.app-head h1 { font-size: 28px; margin: 4px 0 2px; }
.app-head p { margin: 0; color: var(--body-text-color-subdued); }
.stepper { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; list-style: none; padding: 0;
           margin: 12px 0 4px; }
.stepper .st { margin: 0 !important; line-height: 1.2; }
.stepper .st { display: flex; align-items: center; gap: 8px; padding: 6px 14px 6px 6px; border-radius: 999px;
               border: 1px solid var(--border-color-primary); color: var(--body-text-color-subdued); font-size: 15px; }
.stepper .dot { width: 26px; height: 26px; border-radius: 50%; display: inline-grid; place-items: center;
                font-weight: 600; font-size: 14px; background: var(--background-fill-secondary); }
.stepper .active { border-color: var(--ink); color: var(--body-text-color); font-weight: 600; }
.stepper .active .dot { background: var(--ink); color: #fff; }
.stepper .done .dot { background: #2e8b57; color: #fff; }
.stepper .done { color: var(--body-text-color); }
.stepper .skip { opacity: .55; border-style: dashed; }
.hint { font-size: 14px; color: var(--body-text-color-subdued); margin: 2px 0 6px; }
.empty { border: 2px dashed var(--border-color-primary); border-radius: 14px; padding: 48px 24px; text-align: center;
         color: var(--body-text-color-subdued); }
.empty .big { font-size: 20px; font-weight: 600; color: var(--body-text-color); margin-bottom: 6px; }
.summary p { font-size: 15px; color: var(--body-text-color-subdued); margin: 0; }
.final-text textarea { direction: rtl; text-align: right; font-family: var(--sd-font) !important; font-size: 26px !important;
                       line-height: 2 !important; }
.sd { direction: rtl; text-align: right; font-family: var(--sd-font); font-size: 22px; line-height: 2; }
.sd .chg, .key .chg { background: var(--chg-bg); color: var(--chg-fg); border-radius: 4px; padding: 0 4px; }
.sd .sug, .key .sug { border-bottom: 3px solid var(--sug); }
.cmp { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }
.cmp section { border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 10px 14px; min-width: 0; }
.cmp h4 { margin: 0 0 4px; font-size: 14px; color: var(--body-text-color-subdued); font-weight: 600; }
.key { font-size: 14px; color: var(--body-text-color-subdued); margin: 8px 2px 0; }
.legend { display: flex; flex-wrap: wrap; gap: 16px; font-size: 14px; color: var(--body-text-color-subdued); }
.legend .lg { display: inline-flex; align-items: center; gap: 6px; }
.legend i { width: 16px; height: 12px; border: 3px solid; border-radius: 2px; display: inline-block; }
.banner { border-radius: 12px; padding: 12px 16px; font-size: 15px; line-height: 1.5; }
.banner p { margin: 6px 0; } .banner ul { margin: 4px 0 6px 20px; }
.banner.warn { background: var(--warn-bg); border: 1px solid var(--warn-border); color: var(--warn-fg); }
.banner.info { background: var(--info-bg); border: 1px solid var(--info-border); color: var(--info-fg); }
.banner details { font-size: 13px; opacity: .85; }
.card { border: 1px solid var(--border-color-primary) !important; border-radius: 12px !important; padding: 8px !important;
        background: var(--block-background-fill) !important; }
.card .reading textarea { direction: rtl; text-align: right; font-size: 20px !important;
        font-family: var(--sd-font) !important; }
.card .pick input { font-size: 18px !important; font-family: var(--body-text-font), var(--sd-font) !important; }
.save-note { font-size: 13px; color: var(--body-text-color-subdued); }
.screen.setup { max-width: 640px; margin: 0 auto; }
.status { min-height: 48px; }
.status-text { font-size: 14px; color: var(--body-text-color-subdued); text-align: center; margin: 4px 0; }
.topbar { align-items: center; gap: 12px; }
.topbar .summary p { font-size: 15px; }
.final-text textarea { field-sizing: content; min-height: 0 !important; }
.review { border-top: 1px solid var(--border-color-primary); padding-top: 8px; }
"""


def write_txt(page: dict, image_name: str) -> str:
    out = Path(tempfile.mkdtemp(prefix="sindhi_ocr_")) / f"{Path(image_name).stem}.txt"
    out.write_text(page["text"] + "\n", encoding="utf-8")
    return str(out)


def card_updates(binary, page) -> list:
    """Per prepared card: card visibility, word image, reading, choices."""
    sugg = page["suggestions"][:MAX_ROWS] if page and page["correct_mode"] == "suggest" else []
    ups = []
    for r in range(MAX_ROWS):
        if r < len(sugg):
            sg = sugg[r]
            crop = cv2.cvtColor(word_crop(binary, sg["box"]), cv2.COLOR_GRAY2RGB)
            ups += [gr.update(visible=True), crop, sg["reading"],
                    gr.update(choices=present.row_choices(sg), value="0")]
        else:
            ups += [gr.update(visible=False), None, "", gr.update(choices=[("Keep original", "0")], value="0")]
    return ups


# Everything screen 2 shows, in this order (plus one card's 4 parts per prepared card). `cleared()` and
# `process()` both return values in this order, so nothing from an earlier page can survive a new upload.
RESULT_FIELDS = ("setup_screen", "results_screen", "warn", "summary", "final", "download", "review", "review_head",
                 "saved", "original", "pageimg", "show_boxes", "compare", "state")


def cleared() -> list:
    """Screen 1 shown, screen 2 hidden and emptied, no stored page."""
    return ([gr.update(visible=True), gr.update(visible=False), gr.update(value="", visible=False), "",
             gr.update(value="", lines=1), gr.update(value=None, visible=False), gr.update(visible=False), "", "",
             None, None, True, "", None]
            + card_updates(None, None))


def on_upload(image_path):
    """A new photo (or a cleared one) makes every earlier result invalid."""
    return [present.stepper_html(2 if image_path else 1, {1} if image_path else set())] + cleared()


def read_another():
    """Back to screen 1 with the upload cleared."""
    return [None, present.stepper_html(1, set())] + cleared()


def same_photo(state, image_path) -> bool:
    return bool(state) and state.get("image_path") == image_path


def process(image_path, mode: str, words: str, progress=gr.Progress()):
    try:
        present.check_image(image_path)
    except ValueError as e:
        raise gr.Error(str(e), title="Can't read this file")
    t0 = time.time()
    progress(*present.PROGRESS["clean"])
    rgb = load_rgb(image_path)
    binary = clean_page(rgb)
    page = run_pipeline.run([binary], DEVICE, correct=mode, words=words,
                            progress=lambda stage: progress(*present.PROGRESS[stage]))[0]
    seconds = time.time() - t0
    progress(*present.PROGRESS["done"])
    name = Path(image_path).name
    plain = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    state = {"binary": binary, "page": page, "image_name": name, "image_path": image_path, "plain": plain,
             "seconds": seconds}
    suggest = page["correct_mode"] == "suggest"
    needs_review = suggest and bool(page["suggestions"])
    step = present.stepper_html(3 if needs_review else 4, {1, 2}, review_used=suggest)
    warn = present.warning_html(page)
    return ([STATUS_IDLE, step, gr.update(visible=False), gr.update(visible=True),
             gr.update(value=warn, visible=bool(warn)), present.top_summary(page, seconds),
             gr.update(value=present.final_text(page), lines=present.text_rows(page)),
             gr.update(value=write_txt(page, name), visible=True),
             gr.update(visible=suggest), present.review_title(page, MAX_ROWS), "",
             rgb, present.draw_boxes(binary, page), True, present.compare_html(page), state]
            + card_updates(binary, page))


def apply(state, image_path, save: bool, *picks):
    if not state or state["page"]["correct_mode"] != "suggest":
        raise gr.Error("Read a page in Suggest mode first.", title="Nothing to apply")
    if not same_photo(state, image_path):
        raise gr.Error("The photo has changed since this page was read. Please read it again.",
                       title="Results are out of date")
    page = state["page"]
    sugg = page["suggestions"][:MAX_ROWS]
    choices = {sg["index"]: int(p or 0) for sg, p in zip(sugg, picks)}
    apply_page_choices(page, choices)
    saved = ""
    if save and sugg:
        folder = save_feedback(state["binary"], state["image_name"], dict(page, suggestions=sugg), choices)
        saved = (f"✓ Saved {len(sugg)} reviewed word{'s' * (len(sugg) != 1)} to `{folder}`. "
                 "They are not used for training automatically.")
    elif save:
        saved = "Nothing to save: this page has no suggestions."
    return (present.stepper_html(4, {1, 2, 3}),
            gr.update(value=present.final_text(page), lines=present.text_rows(page)),
            gr.update(value=write_txt(page, state["image_name"]), visible=True),
            present.draw_boxes(state["binary"], page), True, present.compare_html(page),
            present.top_summary(page, state["seconds"]), saved, state)


def toggle_boxes(state, image_path, show: bool):
    if not same_photo(state, image_path):
        return None
    return present.draw_boxes(state["binary"], state["page"]) if show else state["plain"]


def bulk(state, accept: bool):
    vals = present.bulk_choices(state["page"] if state else None, MAX_ROWS, accept)
    return [gr.update(value=v) for v in vals]


STATUS_IDLE = ""   # the progress bar appears in this space while reading


def build() -> gr.Blocks:
    with gr.Blocks(title="Sindhi handwriting OCR") as demo:
        state = gr.State(None)
        gr.HTML('<div class="app-head"><h1>Sindhi Handwriting OCR</h1>'
                "<p>Turn a photo of a handwritten Sindhi page into text. Everything runs on this computer.</p></div>")
        stepper = gr.HTML(present.stepper_html(1, set()))

        # ---- screen 1: set up
        with gr.Column(visible=True, elem_classes=["screen", "setup"]) as setup_screen:
            img = gr.Image(type="filepath", sources=["upload"], label="1. Page photo", height=320, buttons=[],
                           placeholder="Drop a page photo here, or click to choose (JPG or PNG)")
            gr.HTML('<p class="hint">Best results: a dark pen on plain, unlined paper; an upright, sharp photo '
                    "taken straight from above in even light. Pencil and ruled or dotted paper are not "
                    "supported.</p>")
            example = gr.Button("Try an example page", size="md", variant="secondary")
            mode = gr.Radio(MODES, value="auto", label="Correction")
            with gr.Accordion("Advanced settings", open=False):
                words = gr.Radio(WORDS, value="joins", label="Word finding")
            read = gr.Button("2. Read page", variant="primary", size="lg")
            gr.HTML('<p class="status-text">Reading takes about 10–60 seconds.</p>')
            status = gr.HTML(STATUS_IDLE, elem_classes=["status"])

        # ---- screen 2: results
        with gr.Column(visible=False, elem_classes=["screen"]) as results_screen:
            with gr.Row(equal_height=True, elem_classes=["topbar"]):
                back = gr.Button("← Read another page", size="sm", variant="secondary", scale=0, min_width=190)
                summary = gr.Markdown(elem_classes=["summary"])
            warn = gr.HTML(visible=False)
            final = gr.Textbox(label="Final text", lines=1, max_lines=40, interactive=False, rtl=True,
                               buttons=["copy"], elem_classes=["final-text"])
            download = gr.DownloadButton("4. Download text (.txt)", visible=False, variant="secondary", size="md")
            with gr.Column(visible=False, elem_classes=["review"]) as review:
                review_head = gr.Markdown()
                with gr.Row():
                    accept_all = gr.Button("Accept all suggestions", size="sm")
                    keep_all = gr.Button("Keep all originals", size="sm")
                picks, card_parts = [], []
                for r in range(MAX_ROWS):
                    with gr.Row(visible=False, equal_height=True, elem_classes=["card"]) as card:
                        crop = gr.Image(show_label=False, interactive=False, height=80, scale=2, min_width=140,
                                        buttons=[])
                        reading = gr.Textbox(label="Read as", interactive=False, scale=2, min_width=120,
                                             elem_classes=["reading"])
                        pick = gr.Dropdown(choices=[("Keep original", "0")], value="0", label="Your choice",
                                           scale=3, min_width=180, elem_classes=["pick"])
                    picks.append(pick)
                    card_parts += [card, crop, reading, pick]
                save = gr.Checkbox(value=False, label="Save my choices to help improve the recognizer later")
                gr.HTML(f'<p class="save-note">If ticked, <b>Apply choices</b> also stores each reviewed word '
                        f"(its image, the reading and your choice) in <code>{feedback_root()}</code> on this "
                        "computer. Nothing is sent anywhere, and the data is never used for training "
                        "automatically: a person has to check it first. Delete the folder at any time to "
                        "remove it.</p>")
                apply_btn = gr.Button("3. Apply choices", variant="primary")
                saved = gr.Markdown()
            with gr.Tabs():
                with gr.Tab("Pages"):
                    with gr.Row(equal_height=False):
                        gr.HTML(present.legend_html())
                        show_boxes = gr.Checkbox(value=True, label="Show boxes", scale=0, min_width=140)
                    pageimg = gr.Image(label="Cleaned page (use ⛶ to zoom)", interactive=False,
                                       buttons=["fullscreen", "download"])
                    original = gr.Image(label="Your photo (use ⛶ to zoom)", interactive=False,
                                        buttons=["fullscreen", "download"])
                with gr.Tab("Compare"):
                    compare = gr.HTML()

        fields = dict(setup_screen=setup_screen, results_screen=results_screen, warn=warn, summary=summary,
                      final=final, download=download, review=review, review_head=review_head, saved=saved,
                      original=original, pageimg=pageimg, show_boxes=show_boxes, compare=compare, state=state)
        result_outputs = [fields[n] for n in RESULT_FIELDS] + card_parts
        img.change(on_upload, img, [stepper] + result_outputs)
        example.click(lambda: str(EXAMPLE), None, img)
        back.click(read_another, None, [img, stepper] + result_outputs)
        (read.click(lambda: gr.update(interactive=False, value="Reading…"), None, read)
             .then(process, [img, mode, words], [status, stepper] + result_outputs, api_name="process",
                   concurrency_limit=1, show_progress_on=[status])
             .then(lambda: gr.update(interactive=True, value="2. Read page"), None, read))
        apply_btn.click(apply, [state, img, save] + picks,
                        [stepper, final, download, pageimg, show_boxes, compare, summary, saved, state],
                        api_name="apply", concurrency_limit=1)
        show_boxes.change(toggle_boxes, [state, img, show_boxes], pageimg)
        accept_all.click(lambda s: bulk(s, True), state, picks)
        keep_all.click(lambda s: bulk(s, False), state, picks)
    return demo


def main() -> None:
    global DEVICE
    ap = argparse.ArgumentParser(description="Local web frontend for the Sindhi OCR pipeline")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    for g in ("step1_preprocessing", "step2_cnn_transformer", "step3_tesseract", "step4_xlmroberta"):
        require_gate(g, allow_user_accepted=True)
    DEVICE = get_device("auto")
    build().launch(server_name="127.0.0.1", server_port=args.port, share=False, theme=THEME, css=CSS,
                   show_error=True, allowed_paths=[str(EXAMPLE.parent)])


if __name__ == "__main__":
    main()
