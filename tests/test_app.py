"""The web app's screen switching and stale-result protection (app.py). No model is loaded.

A new upload or "Read another page" must clear every earlier result, and Apply / the boxes toggle must refuse a page
that was read from a different photo, so results can never show for another image than the current one.

Run: .venv/bin/python -m pytest tests/test_app.py -q
"""
import numpy as np
import pytest

gr = pytest.importorskip("gradio")
import app  # noqa: E402

from pipeline.page import finish_page  # noqa: E402

N = len(app.RESULT_FIELDS)


def val(u):
    return u.get("value") if isinstance(u, dict) else u


def stored_page(image_path="/tmp/a.png"):
    page = {"lines": [{"box": [0, 0, 100, 20], "words": [{"box": [10, 2, 50, 18], "text": "ٽا؟"}]}],
            "tokens": ["ٽا؟"], "correct_mode": "suggest", "words_mode": "joins", "warnings": [],
            "suggestions": [{"index": 0, "reading": "ٽا؟", "box": [10, 2, 50, 18],
                             "candidates": [{"text": "ٿا؟", "words": ["ٿا؟"], "kind": "replace", "consumes": 1}]}]}
    finish_page(page, ["ٽا؟"], ["keep"])
    binary = np.full((20, 100), 255, np.uint8)
    return {"binary": binary, "page": page, "image_name": "a.png", "image_path": image_path,
            "plain": np.stack([binary] * 3, -1), "seconds": 1.0}


def check_cleared(fields):
    f = dict(zip(app.RESULT_FIELDS, fields[:N]))
    assert f["setup_screen"]["visible"] is True and f["results_screen"]["visible"] is False
    assert f["state"] is None and val(f["final"]) == "" and val(f["download"]) is None
    assert f["original"] is None and f["pageimg"] is None and f["compare"] == "" and val(f["warn"]) == ""
    assert f["review"]["visible"] is False and f["summary"] == ""
    cards = fields[N:]
    assert len(cards) == 4 * app.MAX_ROWS
    assert all(cards[4 * r]["visible"] is False and cards[4 * r + 1] is None for r in range(app.MAX_ROWS))


def test_new_upload_clears_all_previous_results():
    out = app.on_upload("/tmp/b.png")
    assert "Read" in out[0]
    check_cleared(out[1:])


def test_read_another_page_clears_the_upload_and_results():
    out = app.read_another()
    assert out[0] is None                      # the upload is emptied
    assert 'class="st active"' in out[1] and ">1<" in out[1]
    check_cleared(out[2:])


def test_results_from_another_photo_are_refused():
    st = stored_page("/tmp/a.png")
    with pytest.raises(gr.Error):
        app.apply(st, "/tmp/b.png", False, "1")
    assert app.toggle_boxes(st, "/tmp/b.png", True) is None
    assert app.toggle_boxes(st, "/tmp/a.png", False) is st["plain"]
    out = app.apply(st, "/tmp/a.png", False, "1")
    assert val(out[1]) == "ٿا؟"


def test_top_summary_names_the_settings():
    from pipeline import present
    page = stored_page()["page"]
    assert present.top_summary(page, 2.0).startswith("Suggest · word finding: joins · 1 line · 1 word")
    assert present.text_rows(page) == 1
