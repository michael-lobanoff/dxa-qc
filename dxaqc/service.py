"""The QC service: DICOM image -> region -> landmarks -> measurements -> per-violation decisions."""
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import pydicom

from .decision import NAMES_RU, hip_measurements, spine_measurements
from .detector import KeypointDetector
from .hipcrop import hip_crop
from .hog import hog
from .pixels import pixel_spacing_mm
from .region import RegionClassifier

REGION_RU = {"spine": "поясничный отдел позвоночника", "hip_left": "проксимальный отдел левого бедра",
             "hip_right": "проксимальный отдел правого бедра"}
IMPLANT_METAL_FRACTION = 0.015  # saturated share: implants 0.018-0.12 on the training set, normal hips < 0.012


def read_dicom_image(path):
    """Return (uint8 grayscale image, dataset). Handles MONOCHROME1, rescale, >8-bit data, RGB and multi-frame."""
    ds = pydicom.dcmread(path, force=True)
    arr = ds.pixel_array
    if arr.ndim == 3 and int(getattr(ds, "SamplesPerPixel", 1)) == 3:
        arr = cv2.cvtColor(arr.astype(np.uint8) if arr.dtype != np.uint8 else arr, cv2.COLOR_RGB2GRAY)
    elif arr.ndim == 3:  # multi-frame: DXA exports are single-frame, take the first
        arr = arr[0]
    a = arr.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1) or 1) + float(getattr(ds, "RescaleIntercept", 0) or 0)
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        a = a.max() - a
    if a.max() > 255 or a.min() < 0:
        lo, hi = np.percentile(a, [0.5, 99.5])
        a = (a - lo) / max(hi - lo, 1e-6) * 255
    return np.clip(a, 0, 255).astype(np.uint8), ds


class QCService:
    def __init__(self, model_dir="models", device=None):
        model_dir = Path(model_dir)
        self.region = RegionClassifier.load(model_dir / "region.pkl")
        self.detector = KeypointDetector(model_dir, device)
        with open(model_dir / "decision.pkl", "rb") as f:
            d = pickle.load(f)
        self.decision = {"spine": d["spine"], "hip": d["hip"]}
        self.rotation, self.hog_cell = d["rotation_rf"], d["hog_cell"]

    def analyse(self, img, mm_per_px=None):
        """mm_per_px: pixel size of this image (see pixels.pixel_spacing_mm); None = the export default."""
        region, region_conf = self.region.predict(img)
        region = str(region)
        points, conf = self.detector(img, region)
        out = {"region": region, "region_conf": region_conf, "points": points, "point_conf": conf,
               "mm_per_px": mm_per_px}
        if region == "spine":
            meas = spine_measurements(img, points, conf, mm_per_px)
        else:
            if float((img >= img.max() - 2).mean()) > IMPLANT_METAL_FRACTION:
                out.update(implant=True, measurements={}, probs={}, violations=[], score=None)
                return out
            meas = hip_measurements(img, points, conf, region, mm_per_px)
            crop, _ = hip_crop(img, points, region)
            meas["rotation"] = float(self.rotation.predict_proba(hog(crop, self.hog_cell)[None])[0, 1])
        probs, decs, score = self.decision["spine" if region == "spine" else "hip"].predict(meas)
        violations = [vt for vt, d in decs.items() if d]
        # ТЗ 2.3 (upper coverage: mid-Th12 must be in the frame) is checked by rule, not by a learned
        # model: no study in the training set was marked bad for it, so there is nothing to fit.
        if region == "spine" and meas.get("top_coverage_ok") is False and "v_pos" not in violations:
            violations.append("v_pos")
            score = max(score, 0.9)
        out.update(implant=False, measurements=meas, probs=probs, violations=violations, score=score)
        return out

    def process_file(self, path, study_dir="", vis_dir=None):
        """One results row. With vis_dir, also writes <image_uid>.png, a Secondary Capture <image_uid>_qc.dcm
        and a Basic Text SR <image_uid>_sr.dcm with the findings."""
        t0 = time.time()
        row = {"path_to_study": study_dir, "file": str(path), "study_uid": "", "image_uid": "", "anatomical_region": "",
               "quality_class": None, "quality_score": None, "violation_type": "", "violation_codes": "",
               "processing_status": "Failure", "time_of_processing": None, "details": "", "error": "",
               "mm_per_px": None, "mm_per_px_source": ""}
        try:
            img, ds = read_dicom_image(path)
            row["study_uid"] = str(getattr(ds, "StudyInstanceUID", ""))
            row["image_uid"] = str(getattr(ds, "SOPInstanceUID", ""))
            mm, mm_src = pixel_spacing_mm(ds, img.shape)
            r = self.analyse(img, mm)
            row["mm_per_px"] = round(mm, 4)
            row["mm_per_px_source"] = mm_src
            row["anatomical_region"] = REGION_RU[r["region"]]
            if r["implant"]:
                row.update(quality_class=0, violation_type="эндопротез: оценка качества не проводится", violation_codes="implant")
            else:
                row.update(quality_class=int(bool(r["violations"])), quality_score=round(r["score"], 4),
                           violation_type="; ".join(NAMES_RU[v] for v in r["violations"]),
                           violation_codes=";".join(r["violations"]))
                m = r["measurements"]
                if r["region"] == "spine":
                    bits = []
                    if m.get("tilt_deg") is not None:
                        bits.append(f"наклон оси {m['tilt_deg']:.1f}°")
                    if m.get("span_vert") is not None:
                        bits.append(f"над гребнями {m['span_vert']:.1f} позвонка")
                    if m.get("vert_pitch_mm"):
                        bits.append(f"высота позвонка {m['vert_pitch_mm']:.0f} мм")
                    row["details"] = "; ".join(bits)
                else:
                    ru = {"top": "сверху", "bottom": "снизу", "lateral": "сбоку"}
                    row["details"] = "; ".join(f"поле {ru[k]} {m[f'margin_{k}']:.0f} мм" for k in ("top", "bottom", "lateral")
                                               if m.get(f"margin_{k}") is not None)
            row["processing_status"] = "Success"
            if vis_dir is not None:
                from PIL import Image
                from .visualize import render_overlay, to_secondary_capture
                rgb = render_overlay(img, r)
                stem = Path(vis_dir) / (row["image_uid"] or Path(path).stem)
                stem.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb).save(f"{stem}.png")
                to_secondary_capture(rgb, ds).save_as(f"{stem}_qc.dcm", enforce_file_format=True)
                from .sr import to_structured_report
                to_structured_report(row, ds).save_as(f"{stem}_sr.dcm", enforce_file_format=True)
        except Exception as e:  # every failure is reported, never raised
            row["error"] = f"{type(e).__name__}: {e}"
        row["time_of_processing"] = round(time.time() - t0, 3)
        return row
