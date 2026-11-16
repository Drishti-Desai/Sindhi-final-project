"""What the web app shows (pipeline/present.py). No model and no Gradio needed.

Run: .venv/bin/python -m pytest tests/test_present.py -q
"""
import numpy as np
import pytest
from PIL import Image

from pipeline import present
from pipeline.page import finish_page


def page(mode="auto", actions=("keep", "replace"), warnings=(), suggestions=()):
    p = {"lines": [{"box": [0, 0, 100, 20], "words": [{"box": [60, 2, 90, 18], "text": "ڪهڙا"},
                                                      {"box": [10, 2, 50, 18], "text": "ٽا؟"}]}],
         "tokens": ["ڪهڙا", "ٽا؟"], "correct_mode": mode, "warnings": list(warnings),
         "suggestions": list(suggestions)}
    out = ["ڪهڙا", "ٿا؟"] if actions[1] != "keep" else ["ڪهڙا", "ٽا؟"]
    return finish_page(p, out, list(actions))


def test_stepper_marks_done_active_and_skipped_review():
    h = present.stepper_html(4, {1, 2}, review_used=False)
    assert h.count('class="st done"') == 2 and 'class="st active"' in h and "Review (Suggest only)" in h
    assert 'class="st skip"' not in present.stepper_html(3, {1, 2}, review_used=True)


def test_check_image_gives_friendly_messages(tmp_path):
    good = tmp_path / "page.png"
    Image.new("RGB", (800, 600), "white").save(good)
    present.check_image(str(good))
    small = tmp_path / "small.png"
    Image.new("RGB", (300, 60), "white").save(small)
    with pytest.raises(ValueError, match="too small"):
        present.check_image(str(small))
    bad = tmp_path / "notes.png"
    bad.write_text("not an image")
    with pytest.raises(ValueError, match="JPG or PNG"):
        present.check_image(str(bad))
    with pytest.raises(ValueError, match="choose a page photo"):
        present.check_image(None)


def test_compare_highlights_changes_and_pending_suggestions():
    h = present.compare_html(page("auto"))
    assert '<span class="chg">ٿا؟</span>' in h and "Corrected text" in h
    sg = [{"index": 1, "reading": "ٽا؟", "candidates": [{"text": "ٿا؟", "consumes": 1}]}]
    h = present.compare_html(page("suggest", ("keep", "keep"), suggestions=sg))
    assert '<span class="sug">ٽا؟</span>' in h and '<span class="chg">' not in h.split('<p class="key">')[0]


def test_summary_line():
    assert present.summary_line(page("auto"), 9.83) == "1 line · 2 words · 1 changed by the correction · 9.8 s"
    assert present.summary_line(page("off", ("keep", "keep"))) == "1 line · 2 words · correction off"


def test_warning_banner_only_when_needed():
    assert present.warning_html(page()) == ""
    w = present.warning_html(page(warnings=["40 text-line pieces were found", "generic advice"]))
    assert "may not have been read reliably" in w and "40 text-line pieces" in w and "generic advice" not in w
    empty = page()
    empty["tokens"] = []
    assert "No handwriting was found" in present.warning_html(empty)


def test_review_choices_and_bulk_buttons():
    sg = {"index": 1, "reading": "ٽا؟", "next_reading": "ڪري",
          "candidates": [{"text": "ٿا؟", "consumes": 1}, {"text": "تاڪري", "consumes": 2}]}
    opts = present.row_choices(sg)
    assert opts[0] == ("Keep original: ٽا؟", "0") and opts[1] == ("1. ٿا؟", "1") and "next word ڪري" in opts[2][0]
    p = page("suggest", ("keep", "keep"), suggestions=[sg, dict(sg, index=0)])
    assert present.bulk_choices(p, 4, accept=True) == ["1", "1", "0", "0"]
    assert present.bulk_choices(p, 4, accept=False) == ["0"] * 4
    assert present.bulk_choices(None, 2, accept=True) == ["0", "0"]


def test_draw_boxes_colours_changed_words_green():
    binary = np.full((20, 100), 255, np.uint8)
    vis = present.draw_boxes(binary, page("auto"))
    assert tuple(vis[2, 10]) == present.COLOURS["changed"] and tuple(vis[2, 60]) == present.COLOURS["word"]


def test_progress_callback_reports_stages(monkeypatch):
    import run_pipeline
    import pipeline.page as pg
    stages = []
    monkeypatch.setattr(run_pipeline, "load_recognizer", lambda device: None)
    monkeypatch.setattr(pg, "read_page", lambda *a, stats=None, **k: (stats.update(line_boxes=[]), [])[1])
    monkeypatch.setattr(pg, "free", lambda device: None)
    out = run_pipeline.run([np.full((10, 10), 255, np.uint8)], None, correct="off", progress=stages.append)
    assert stages == ["read"] and out[0]["text"] == ""
