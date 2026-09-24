"""Smoke tests for the QC service: organizer samples, odd DICOM encodings, broken inputs, reproducibility.

Run: .venv/bin/python -m pytest -q tests  (needs the trained models in models/ and data/test_sample)
"""
from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.uid import ExplicitVRLittleEndian

from dxaqc.service import QCService, read_dicom_image

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = sorted((ROOT / "data/test_sample").glob("*.dcm"))

# The organisers' sample DICOMs are medical data and are not in the repository: put them in
# data/test_sample to run the tests that need them; everything else runs without.
needs_samples = pytest.mark.skipif(not SAMPLES, reason="нужны тестовые DICOM организаторов в data/test_sample")
EXPECTED_REGION = {"ПОП": "Поясничный отдел позвоночника", "ППОБ": "Проксимальный отдел бедра",
                   "ЛПОБ": "Проксимальный отдел бедра"}


@pytest.fixture(scope="module")
def service():
    return QCService(ROOT / "models")


def _variant(src, tmp_path, name, transform):
    ds = pydicom.dcmread(src)
    transform(ds)
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    out = tmp_path / name
    ds.save_as(out, enforce_file_format=True)
    return out


@needs_samples
def test_organizer_samples_regions(service):
    for p in SAMPLES:
        row = service.process_file(p)
        assert row["processing_status"] == "Success", row["error"]
        assert row["anatomical_region"] == EXPECTED_REGION[p.stem.split("_")[1]]
        assert row["quality_class"] in (0, 1) and 0 <= row["quality_prob"] <= 1
        assert row["image_uid"] and row["study_uid"]


@needs_samples
def test_sixteen_bit_and_monochrome1_give_same_verdict(service, tmp_path):
    src = SAMPLES[0]
    base = service.process_file(src)

    def to16(ds):
        a = ds.pixel_array.astype(np.uint16) * 257
        ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
        ds.PixelData = a.tobytes()

    def to_mono1(ds):
        ds.PixelData = (255 - ds.pixel_array).astype(np.uint8).tobytes()
        ds.PhotometricInterpretation = "MONOCHROME1"

    for name, tr in (("x16.dcm", to16), ("mono1.dcm", to_mono1)):
        p = _variant(src, tmp_path, name, tr)
        img, _ = read_dicom_image(p)
        assert img.dtype == np.uint8 and abs(float(img.mean()) - float(read_dicom_image(src)[0].mean())) < 3
        row = service.process_file(p)
        assert row["processing_status"] == "Success" and row["anatomical_region"] == base["anatomical_region"]
        assert row["quality_class"] == base["quality_class"]


@pytest.mark.parametrize("content", [b"", b"not a dicom at all" * 10, b"\0" * 128 + b"DICM" + b"\x02\x00" * 50])
def test_broken_files_are_failures_not_crashes(service, tmp_path, content):
    p = tmp_path / "broken.dcm"
    p.write_bytes(content)
    row = service.process_file(p)
    assert row["processing_status"] == "Failure" and row["error"]


@needs_samples
def test_dicom_without_pixels_is_failure(service, tmp_path):
    ds = pydicom.dcmread(SAMPLES[0])
    del ds.PixelData
    p = tmp_path / "nopixels.dcm"
    ds.save_as(p, enforce_file_format=True)
    assert service.process_file(p)["processing_status"] == "Failure"


@needs_samples
def test_reproducible(service):
    a = [service.process_file(p) for p in SAMPLES]
    b = [service.process_file(p) for p in SAMPLES]
    for r1, r2 in zip(a, b):
        assert (r1["quality_class"], r1["violation_codes"], r1["details"]) == (r2["quality_class"], r2["violation_codes"], r2["details"])
        assert abs(r1["quality_prob"] - r2["quality_prob"]) < 1e-6


def test_pixel_spacing_from_exposed_area():
    """The pixel size comes from the DICOM Exposed Area tag, with a plausibility check (pixels.py)."""
    import pydicom

    from dxaqc.pixels import DEFAULT_MM_PER_PX, pixel_spacing_mm

    ds = pydicom.Dataset()
    ds.ExposedArea = [180, 175]
    (sx, sy), src = pixel_spacing_mm(ds, (289, 300))
    assert src == "ExposedArea" and abs(sx - 0.600) < 0.01 and abs(sy - 0.606) < 0.01
    ds.ExposedArea = [520, 595]                      # whole scan table, repeated for every image
    (sx, sy), src = pixel_spacing_mm(ds, (289, 300))
    assert src == "default" and sx == sy == DEFAULT_MM_PER_PX
    assert pixel_spacing_mm(pydicom.Dataset(), (289, 300))[1] == "default"


def test_pixel_spacing_override(monkeypatch):
    """The organisers quote an anisotropic 0.6 x 1.05 mm pixel; one variable switches the service to it."""
    import pydicom

    from dxaqc.pixels import pixel_spacing_mm

    monkeypatch.setenv("DXAQC_PIXEL_MM", "0.6,1.05")
    (sx, sy), src = pixel_spacing_mm(pydicom.Dataset(), (289, 300))
    assert (sx, sy) == (0.6, 1.05) and src == "DXAQC_PIXEL_MM"


@needs_samples
def test_vertebrae_pitch_is_anatomical(service):
    """The vertebral train gives a plausible pitch (body + disc) on a real spine scan."""
    from dxaqc.vertebrae import find

    spine = next(p for p in SAMPLES if "ПОП" in p.name)
    img, _ = read_dicom_image(spine)
    points, _ = service.detector(img, "spine")
    v = find(img, points, 0.603)
    assert v is not None and 4 <= v["n"] <= 8
    assert 26 <= v["pitch_px"] * 0.603 <= 40
    assert np.all(np.diff([p[1] for p in v["points"]]) > 0)   # ordered top to bottom


@needs_samples
def test_refuses_foreign_images(service, tmp_path):
    """A blank frame, noise and a rotated scan are refused instead of getting a quality verdict."""
    import numpy as np

    rng = np.random.default_rng(0)
    base = pydicom.dcmread(SAMPLES[0]).pixel_array
    cases = {"blank": np.full(base.shape, 40, np.uint8),
             "noise": rng.integers(0, 255, base.shape, dtype=np.uint8),
             "rot90": np.ascontiguousarray(np.rot90(base)).astype(np.uint8)}
    for name, arr in cases.items():
        path = _variant(SAMPLES[0], tmp_path, f"{name}.dcm",
                        lambda ds, a=arr: (setattr(ds, "PixelData", np.ascontiguousarray(a).tobytes()),
                                           setattr(ds, "Rows", a.shape[0]), setattr(ds, "Columns", a.shape[1])))
        row = service.process_file(path)
        assert row["processing_status"] == "Success", name
        assert row["violation_codes"] == "unsupported", (name, row["violation_codes"])
        assert row["quality_class"] == 0 and row["anatomical_region"] == "не определена"


@needs_samples
def test_real_scans_are_not_refused(service):
    """The guard must not reject the organisers' own samples."""
    for path in SAMPLES:
        assert service.process_file(path)["violation_codes"] != "unsupported", path.name


@needs_samples
def test_refuses_plain_radiographs(service, tmp_path):
    """A plain lumbar radiograph is not densitometry: refuse by pixel size and by image size.

    Real files (Philips CR, 0.14 mm pixels, ~2900 px) were accepted as DXA before these checks —
    62 of 66 got a full quality verdict. See docs/night_report.md.
    """
    import numpy as np

    from dxaqc.region import MAX_SIDE_PX, RegionClassifier

    big = np.zeros((1600, 1200), np.uint8)
    big[::7] = 200                                    # structure, so it is not refused as noise
    assert max(big.shape) > MAX_SIDE_PX
    ok, why, _ = RegionClassifier.load(ROOT / "models/region.pkl").check(big)
    assert not ok and "рентген" in why

    path = _variant(SAMPLES[0], tmp_path, "radiograph.dcm",
                    lambda ds: setattr(ds, "PixelSpacing", [0.143, 0.143]))
    row = service.process_file(path)
    assert row["violation_codes"] == "unsupported" and row["processing_status"] == "Success"
    # violation_type is a closed vocabulary: the refusal reason goes to details
    assert row["violation_type"] == "" and "не денситометрия" in row["details"]


@needs_samples
def test_screening_policy_flags_at_least_as_much(service):
    """The screening operating point may only add violations, never drop one the balanced point found."""
    screening = QCService(ROOT / "models", policy="screening")
    for p in SAMPLES:
        base, scr = service.process_file(p), screening.process_file(p)
        assert base["processing_status"] == scr["processing_status"] == "Success"
        found = set(filter(None, base["violation_codes"].split(";")))
        found_s = set(filter(None, scr["violation_codes"].split(";")))
        assert found <= found_s, (p.name, found, found_s)
        assert scr["quality_class"] >= base["quality_class"]


@needs_samples
def test_overlay_text_is_russian(service, tmp_path):
    """The picture a doctor sees must carry the same Russian wording as the table (ТЗ 2.6)."""
    from PIL import Image
    from dxaqc.service import read_dicom_image
    from dxaqc.pixels import pixel_spacing_mm
    from dxaqc.visualize import render_overlay
    img, ds = read_dicom_image(SAMPLES[0])
    mm, _ = pixel_spacing_mm(ds, img.shape)
    rgb = render_overlay(img, service.analyse(img, mm))
    assert rgb.shape[0] > img.shape[0] * 3 and rgb.ndim == 3       # header band on top of the 3x image
    out = tmp_path / "overlay.png"
    Image.fromarray(rgb).save(out)
    assert out.stat().st_size > 10_000


@needs_samples
def test_bone_cortex_is_not_printed_markup(service):
    """The lateral cortex of the femoral shaft is long, straight and bright — but it is bone, not the
    ROI markup a densitometer prints. A drawn line, on the other hand, must be recognised."""
    import cv2
    from dxaqc.artifacts import overlay_mask
    from dxaqc.service import read_dicom_image
    hip = next(p for p in SAMPLES if "ПОБ" in p.stem)
    img, _ = read_dicom_image(hip)
    assert overlay_mask(img)[1] == 0
    drawn = img.copy()
    for x in (int(img.shape[1] * 0.25), int(img.shape[1] * 0.75)):
        cv2.line(drawn, (x, 5), (x, img.shape[0] - 5), 255, 1)
    for y in (int(img.shape[0] * 0.3), int(img.shape[0] * 0.7)):
        cv2.line(drawn, (5, y), (img.shape[1] - 5, y), 255, 1)
    assert overlay_mask(drawn)[1] >= 3


@needs_samples
def test_report_columns_match_the_spec(service, tmp_path):
    """ТЗ 2.5 fixes the report columns; ours adds a few of its own. The batch layer filters the row to a
    fixed list, so a field added in the service silently disappears from the file unless it is listed
    there too — that is exactly how `projection` went missing once."""
    from dxaqc.batch import COLUMNS, run_batch
    required = ["path_to_study", "study_uid", "image_uid", "anatomical_region", "quality_class",
                "violation_type", "processing_status", "time_of_processing"]
    df = run_batch(SAMPLES[0].parent, service, None)
    assert [c for c in required if c not in df.columns] == []
    # every field the service fills must survive into the table
    row = service.process_file(SAMPLES[0])
    assert [k for k in row if not k.startswith("_") and k not in COLUMNS] == []
    assert set(df.projection) <= {"переднезадняя (AP)", ""}


@needs_samples
def test_reads_compressed_dicom(service, tmp_path):
    """The organisers' export is uncompressed, but a PACS usually hands out JPEG Lossless or JPEG-LS.
    Without a decoder pydicom raises and every such file would become a Failure row, so the decoders
    are pinned in requirements and this test states which transfer syntaxes must stay readable."""
    from pydicom import uid
    from pydicom.pixels import get_decoder
    for u in (uid.JPEGBaseline8Bit, uid.JPEGLossless, uid.JPEGLosslessSV1, uid.JPEGLSLossless,
              uid.JPEG2000Lossless, uid.RLELossless):
        assert get_decoder(u).is_available, f"нет декодера для {u.name}"

    # a real round trip: RLE Lossless is the one pydicom can also write
    import numpy as np
    import pydicom
    ds = pydicom.dcmread(SAMPLES[0], force=True)
    before, _ = read_dicom_image(SAMPLES[0])
    ds.compress(uid.RLELossless)
    out = tmp_path / "rle.dcm"
    ds.save_as(out, enforce_file_format=True)
    after, _ = read_dicom_image(out)
    assert after.shape == before.shape
    assert np.abs(after.astype(int) - before.astype(int)).max() == 0
