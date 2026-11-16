"""Unit tests for Step 2 building blocks (no training).

Run: .venv/bin/python -m pytest tests/test_step2.py -q
"""
import numpy as np
import torch

from cnn_transformer.letters import augment, fit_to_square
from cnn_transformer.models import LetterCNN, WordRecognizer, load_trunk_from_letter_cnn
from cnn_transformer.words import BOS, EOS, LetterBank, Vocab, collate, to_tensor_image
from common.metrics import cer, edit_distance, wer
from common.text import normalize, strip_punct


def test_normalize_unifies_variants():
    assert normalize("سگھيو ی ك") == "سگهيو ي ڪ"
    assert normalize("آءِ", diacritics=False) == "آء"
    assert normalize("  a   b \n\n c ") == "a b\nc"
    assert strip_punct("،ميچ فڪسنگ.") == "ميچ فڪسنگ"


def test_metrics():
    assert edit_distance("abc", "abc") == 0 and edit_distance("abc", "") == 3
    assert abs(cer(["abcd"], ["abxd"]) - 0.25) < 1e-9
    assert wer(["a b c"], ["a c"]) == 1 / 3


def test_fit_to_square_keeps_aspect():
    img = np.full((40, 10), 255, np.uint8)
    img[5:35, 4:6] = 0                       # tall thin stroke (alif-like)
    out = fit_to_square(img, 64)
    ys, xs = np.nonzero(out)
    assert out.shape == (64, 64) and (ys.max() - ys.min()) > 4 * (xs.max() - xs.min())


def test_augment_keeps_size_and_ink():
    ink = np.zeros((64, 64), np.uint8)
    ink[20:44, 30:34] = 255
    a = {"rotate_deg": 12, "shear_deg": 10, "scale": [0.8, 1.1], "translate": 0.08, "stroke_p": 1, "elastic_p": 1}
    out = augment(ink, np.random.default_rng(0), a)
    assert out.shape == ink.shape and out.sum() > 0


def test_vocab_roundtrip():
    v = Vocab.build(["ميچ", "فڪسنگ"])
    ids = v.encode("ميچ5")
    assert ids[0] == BOS and ids[-1] == EOS and v.decode(ids[1:]) == "ميچ5"


def test_word_image_is_flipped_to_reading_order():
    ink = np.zeros((20, 60), np.uint8)
    ink[:, 50:58] = 255                      # ink near the RIGHT edge (= start of an RTL word)
    t = to_tensor_image(ink, 64, 512)
    cols = np.nonzero(t.sum(0))[0]
    assert t.shape[0] == 64 and cols.mean() < t.shape[1] / 2   # now at the LEFT = first time steps


def _bank():
    b = LetterBank.__new__(LetterBank)
    tall = np.zeros((40, 6), np.uint8); tall[:] = 255     # "alif": tall
    wide = np.zeros((12, 40), np.uint8); wide[:] = 255    # "beh": wide and short
    b.crops = {1: [tall], 2: [wide], 13: [wide]}
    b.height_ratio = {1: 1.0, 2: 0.5, 13: 0.5}
    b.char_to_class = {"ا": 1, "ب": 2, "جه": 13}
    b.multi = ["جه"]
    return b


def test_compose_places_first_letter_on_the_right():
    b = _bank()
    rng = np.random.default_rng(1)
    img = b.compose("اب", rng)              # ink map: 255 = ink
    ink = img > 128
    cols = np.nonzero(ink.any(0))[0]
    right = ink[:, cols[-1] - 2:cols[-1] + 1]
    rows_right = np.nonzero(right.any(1))[0]
    # the right-most ink is the tall alif, so it spans many more rows than the short beh
    assert rows_right.max() - rows_right.min() > 25


def test_tokenize_prefers_digraphs_and_rejects_unknown():
    b = _bank()
    assert b.tokenize("جهب") == [13, 2]     # جه is one class (digraph), then ب
    assert b.tokenize("ابx") is None


def test_collate_and_recognizer_shapes():
    v = Vocab.build(["اب", "ب"])
    batch = [(np.zeros((64, 40), np.float32), "اب"), (np.zeros((64, 90), np.float32), "ب")]
    x, widths, y, labels = collate(batch, v)
    assert x.shape == (2, 1, 64, 96) and widths.tolist() == [40, 90] and y.shape[0] == 2
    m = WordRecognizer(len(v), d_model=64, enc_layers=1, dec_layers=1, heads=2, ff=128)
    logits = m(x, widths, y[:, :-1])
    assert logits.shape == (2, y.shape[1] - 1, len(v))
    seq, probs = m.greedy(x, widths, BOS, EOS, max_len=5)
    assert seq.shape[0] == 2


def test_trunk_transfer_from_letter_cnn():
    cnn = LetterCNN(52, aux=(22, 5, 5))
    rec = WordRecognizer(10)
    n = load_trunk_from_letter_cnn(rec, cnn.state_dict())
    assert n > 50
    k = "stem.0.weight"
    assert torch.equal(rec.trunk.state_dict()[k], cnn.trunk.state_dict()[k])
