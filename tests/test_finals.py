"""Доработки финала: отказ при размытии, сколиоз, пояснения по критериям ТЗ, сводка для лаборанта."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from dxaqc.batch import study_summaries
from dxaqc.features import spine_bend_angle
from dxaqc.quality import SHARPNESS_MIN, sharpness, too_blurred
from dxaqc.service import QCService, explain, read_dicom_image

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = sorted((ROOT / "data/test_sample").glob("*.dcm"))
needs_samples = pytest.mark.skipif(not SAMPLES, reason="нужны тестовые DICOM организаторов в data/test_sample")
SPINE = next((p for p in SAMPLES if "ПОП" in p.name), None)


@pytest.fixture(scope="module")
def service():
    return QCService(ROOT / "models")


def test_sharpness_drops_with_blur():
    """Резкость не зависит от контраста, но падает при размытии и при смазе вдоль одной оси."""
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur((rng.random((300, 300)) * 255).astype(np.float32), (0, 0), 0.8)
    k = np.zeros((15, 15), np.float32)
    k[:, 7] = 1 / 15
    s = sharpness(img)
    assert abs(sharpness(img * 0.5 + 40) - s) < 1e-3 * max(s, 1)
    assert sharpness(cv2.GaussianBlur(img, (0, 0), 3)) < s / 3
    assert sharpness(cv2.filter2D(img, -1, k)) < s / 3


@needs_samples
def test_real_sample_is_sharp_and_blurred_copy_is_refused(service, tmp_path):
    img, ds = read_dicom_image(SPINE)
    assert not too_blurred(img, "spine")[0]
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    ds.PixelData = blurred.astype(ds.pixel_array.dtype).tobytes()
    path = tmp_path / "blurred.dcm"
    ds.save_as(path)
    row = service.process_file(path)
    assert row["processing_status"] == "Success" and row["violation_codes"] == "blurred"
    # непригоден для интерпретации — класс 1, иначе система, читающая только эту колонку, сочтёт его годным
    assert row["quality_class"] == 1 and row["violation_type"] == ""
    assert "размыт" in row["details"] and row["anatomical_region"]
    assert sharpness(blurred) < SHARPNESS_MIN["spine"]


@needs_samples
def test_bend_angle_is_a_number_on_a_real_spine(service):
    img, _ = read_dicom_image(SPINE)
    pts, _, _ = service.detector(img, "spine", members=True)
    a = spine_bend_angle(img, pts)
    assert a is not None and 0 <= a < 90


def test_explain_cites_the_criterion_and_is_honest_near_the_limit():
    hi = explain(["v_axis"], {"tilt_deg": -7.4}, "spine")[0]
    lo = explain(["v_axis"], {"tilt_deg": 4.1}, "spine")[0]
    assert "7.4°" in hi and "при допустимых 5°" in hi
    assert "у границы" in lo                         # не пишем «нарушено, потому что больше 5°» при 4.1°
    assert not any("ТЗ" in s for s in explain(["v_axis", "v_pos", "v_artifact"], {"tilt_deg": 7.4, "scoliosis": True}, "spine")
                   + explain(["v_roi", "v_posrot"], {"rotation": 0.6}, "hip_left"))   # пояснения — для врача, без ссылок на документ
    roi = explain(["v_roi"], {"margin_top": 12, "margin_bottom": 40, "margin_lateral": 25}, "hip_left")[0]
    assert "сверху 12 мм при норме 30 мм" in roi and "снизу" not in roi
    cut = explain(["v_roi"], {}, "hip_left")[0]
    assert "вне кадра" in cut


def test_study_summary_groups_by_folder_and_drops_duplicates():
    rows = [
        # позвоночник и бёдра одного пациента в одной папке, но под разными StudyInstanceUID — как в образце
        {"file": "s1/a.dcm", "path_to_study": "s1", "study_uid": "1.1", "_region": "spine",
         "processing_status": "Success", "quality_class": 1, "violation_codes": "v_axis", "review": ""},
        {"file": "s1/b.dcm", "path_to_study": "s1", "study_uid": "1.2", "_region": "hip_left",
         "processing_status": "Success", "quality_class": 1, "violation_codes": "v_posrot", "review": ""},
        {"file": "s1/b_copy.dcm", "path_to_study": "s1", "study_uid": "1.2", "_region": "hip_left",
         "processing_status": "Success", "quality_class": 1, "violation_codes": "v_posrot", "review": ""},
        {"file": "s1/c.dcm", "path_to_study": "s1", "study_uid": "1.2", "_region": "hip_right",
         "processing_status": "Success", "quality_class": 0, "violation_codes": "", "review": "спорный: …"},
        {"file": "s2/x.dcm", "path_to_study": "s2", "study_uid": "2.1", "_region": "spine",
         "processing_status": "Success", "quality_class": 1, "violation_codes": "blurred", "review": ""},
        {"file": "s2/y.dcm", "path_to_study": "s2", "study_uid": "2.1", "_region": None,
         "processing_status": "Success", "quality_class": 0, "violation_codes": "unsupported", "review": ""},
    ]
    study_summaries(rows)
    s1 = rows[0]["study_summary"]
    assert all(r["study_summary"] == s1 for r in rows[:4])
    assert s1.count("левое бедро") == 1                 # дубль снимка не повторяется
    assert "переснять: позвоночник (наклон оси); левое бедро (ротация)" in s1
    assert "правое бедро — спорный" in s1
    assert "переснять: позвоночник (снимок размыт)" in rows[4]["study_summary"]
    assert "не оценено: y.dcm — не денситометрия" in rows[5]["study_summary"]


@needs_samples
def test_overlay_outlines_what_the_segmentation_net_found(service):
    """Предмет, который нашла только сеть сегментации (кольцо, пуговица), обведён на картинке для врача."""
    from dxaqc.visualize import RED, render_overlay
    img, _ = read_dicom_image(SPINE)
    r = service.analyse(img)
    assert r["seg_mask"].shape == img.shape
    r["violations"] = ["v_artifact"]
    r["seg_mask"] = np.zeros(img.shape, bool)
    base = (render_overlay(img, r) == RED).all(-1).sum()
    r["seg_mask"][100:130, 30:60] = True
    assert (render_overlay(img, r) == RED).all(-1).sum() > base
