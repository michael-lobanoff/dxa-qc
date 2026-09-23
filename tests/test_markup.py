"""Reading the ROI markup a densitometer prints on the image, and proposing a correction (ТЗ 2.6).

The phantom below is synthetic on purpose: the organisers' export contains no printed markup at all,
so a test that needed their data could not exist. The feature itself was measured on a public set
(see docs/night_report.md), and this test only guards the behaviour: the box is read, a separator that
sits in a disc is left alone, and one moved onto a vertebral body is proposed to be moved back.
"""
import cv2
import numpy as np
import pytest

from dxaqc import markup as M

BODIES = ((40, 92), (110, 162), (180, 232), (250, 302))   # vertebral bodies, gaps between them are discs
SEPARATORS = (101, 171, 241)                              # printed lines, each in the middle of a disc
BOX = (30, 34, 220, 308)


def phantom(separators=SEPARATORS):
    """A lumbar column with bright bodies, dark discs and the scanner's box burned into the picture."""
    rng = np.random.default_rng(0)
    img = (rng.normal(24, 4, (400, 250))).astype(np.float32)
    for y0, y1 in BODIES:
        img[y0:y1, 92:160] += 150
    img[34:308, 92:160] += 30                             # soft tissue behind the whole column
    img = np.clip(img, 0, 255).astype(np.uint8)
    x0, y0, x1, y1 = BOX
    for x in (x0, x1):
        cv2.line(img, (x, y0), (x, y1), 255, 1)
    for y in (y0, y1, *separators):
        cv2.line(img, (x0, y), (x1, y), 255, 1)
    return img


def test_reads_the_printed_box():
    m = M.read(phantom())
    assert m is not None
    assert len(m["levels"]) == 4                          # L1-L4
    assert len(m["separators"]) == 3
    for got, want in zip(m["separators"], SEPARATORS):
        assert abs(got - want) <= 2
    assert M.review(phantom(), m)["in_frame"]


def test_correct_markup_is_left_alone():
    img = phantom()
    r = M.review(img, M.read(img))
    assert r["corrections"] == []
    assert r["max_error_mm"] < M.CORRECT_MM


def test_separator_on_a_vertebral_body_is_proposed_to_move():
    """The middle separator is drawn 20 px low, on the body of the next vertebra."""
    img = phantom(separators=(101, 191, 241))
    m = M.read(img)
    r = M.review(img, m)
    assert len(r["corrections"]) == 1
    c = r["corrections"][0]
    assert c["index"] == 2 and c["shift_mm"] < 0          # move it up, back into the disc
    assert BODIES[1][1] <= c["to"] <= BODIES[2][0]        # the proposal lands inside the disc
    assert "разделитель №2" in M.describe(r)


def test_image_without_markup_reads_as_none():
    img = phantom()
    img[:, :] = np.where(img == 255, 40, img)             # erase the printed lines, keep the anatomy
    assert M.read(img) is None
    assert M.describe(M.review(img, None)) is None
