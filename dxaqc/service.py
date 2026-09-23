"""The QC service: DICOM image -> region -> landmarks -> measurements -> per-violation decisions."""
import os
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import pydicom

from .decision import NAMES_RU, hip_measurements, spine_measurements
from .markup import describe as describe_markup
from .artifactnet import ArtifactSegmenter
from .detector import KeypointDetector
from .hipcrop import hip_crop, rotation_features
from .pixels import DXA_MIN_MM_PER_PX, pixel_spacing_mm
from .region import RegionClassifier

# разъяснения V2, вопрос 15: only these two values, the side is not reported (it is still detected
# internally, because the lateral margin of the hip depends on which side it is).
REGION_RU = {"spine": "Поясничный отдел позвоночника", "hip_left": "Проксимальный отдел бедра",
             "hip_right": "Проксимальный отдел бедра"}
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
    def __init__(self, model_dir="models", device=None, policy=None):
        """policy: 'balanced' (default, F1-optimal per violation type — what the reported metrics are
        measured at) or 'screening' (per-type thresholds that catch ~80 % of that violation, fewer misses
        and more false alarms). Override per run with DXAQC_POLICY."""
        self.policy = policy or os.environ.get("DXAQC_POLICY", "balanced")
        model_dir = Path(model_dir)
        self.region = RegionClassifier.load(model_dir / "region.pkl")
        self.detector = KeypointDetector(model_dir, device)
        self.segmenter = ArtifactSegmenter(model_dir, self.detector.device)
        with open(model_dir / "decision.pkl", "rb") as f:
            d = pickle.load(f)
        self.decision = {"spine": d["spine"], "hip": d["hip"]}
        self.rotation, self.hog_cell = d["rotation_rf"], d["hog_cell"]
        self.rotation.n_jobs = 1   # at most three crops per call: worker threads cost more than they save
        self.rotation_blend = float(d.get("rotation_blend", 0.0))   # weight of the other hip
        self.rotation_members = bool(d.get("rotation_members", False))   # average over the ensemble's crops
        self.image_axis = bool(d.get("image_axis", False))   # crop alignment the rotation model was trained on

    def analyse(self, img, mm_per_px=None, rotation_override=None, policy=None):
        """mm_per_px: pixel size of this image (see pixels.pixel_spacing_mm); None = the export default.
        rotation_override: rotation probability already blended with the other hip of the study."""
        ok, why, stats = self.region.check(img)
        if not ok:
            # Not a DXA spine / proximal femur: refuse instead of inventing a quality verdict.
            return {"region": None, "unsupported": True, "reason": why, "checks": stats,
                    "points": {}, "measurements": {}, "probs": {}, "violations": [], "score": None,
                    "implant": False, "quality_class": 0, "mm_per_px": mm_per_px}
        region, region_conf = self.region.predict(img)
        region = str(region)
        points, conf, members = self.detector(img, region, members=True)
        out = {"region": region, "region_conf": region_conf, "points": points, "point_conf": conf,
               "mm_per_px": mm_per_px, "unsupported": False, "checks": stats}
        if region == "spine":
            meas = spine_measurements(img, points, conf, mm_per_px, self.segmenter.area(img))
        else:
            # Q&A 17.09: a hip with an endoprosthesis is judged like any other hip (positioning and field
            # are still the radiographer's job); the implant is only reported as a note in `details`.
            implant = float((img >= img.max() - 2).mean()) > IMPLANT_METAL_FRACTION
            meas = hip_measurements(img, points, conf, region, mm_per_px)
            meas["implant"] = implant
            # Rotation reads a crop aligned by the landmarks, and is sensitive to which detector placed
            # them. With an ensemble, the forest looks at the crop of every member and the answers are
            # averaged: on new detectors 0.826 AUC, against 0.809 for one crop from the averaged points.
            sets = [p for p, _ in members] if self.rotation_members and len(members) > 1 else [points]
            feats = np.stack([rotation_features(hip_crop(img, p, region, image_axis=self.image_axis)[0]) for p in sets])
            meas["rotation_own"] = float(self.rotation.predict_proba(feats)[:, 1].mean())
            meas["rotation"] = meas["rotation_own"] if rotation_override is None else float(rotation_override)
        model = self.decision["spine" if region == "spine" else "hip"]
        policy = policy or self.policy
        probs, decs, score = model.predict(meas, policy)
        quality_class, violations = model.verdict(probs, decs, score, policy)
        # An image with the densitometer's own ROI boxes printed on it cannot be judged for foreign
        # bodies: the printed lines and labels are thin and bright exactly like metal. Say so instead.
        if meas.get("printed_markup") and "v_artifact" in violations:
            violations.remove("v_artifact")
            quality_class = int(bool(violations))
        # ТЗ 2.3 (upper coverage: mid-Th12 must be in the frame) is checked by rule, not by a learned
        # model: no study in the training set was marked bad for it, so there is nothing to fit.
        if region == "spine" and meas.get("top_coverage_ok") is False:
            quality_class, score = 1, max(score, 0.9)
            if "v_pos" not in violations:
                violations.append("v_pos")
        out.update(implant=bool(meas.get("implant", False)), measurements=meas, probs=probs, violations=violations, score=score,
                   quality_class=quality_class)
        return out

    def process_file(self, path, study_dir="", vis_dir=None, rotation_override=None, policy=None):
        """One results row. With vis_dir, also writes <image_uid>.png, a Secondary Capture <image_uid>_qc.dcm
        and a Basic Text SR <image_uid>_sr.dcm with the findings."""
        t0 = time.time()
        row = {"path_to_study": study_dir, "file": str(path), "study_uid": "", "image_uid": "", "anatomical_region": "",
               "quality_class": None, "quality_prob": None, "violation_type": "", "violation_codes": "",
               "processing_status": "Failure", "time_of_processing": None, "details": "", "error": "",
               "mm_per_px": None, "mm_per_px_source": ""}
        try:
            img, ds = read_dicom_image(path)
            row["study_uid"] = str(getattr(ds, "StudyInstanceUID", ""))
            row["image_uid"] = str(getattr(ds, "SOPInstanceUID", ""))
            mm, mm_src = pixel_spacing_mm(ds, img.shape)
            if mm_src in ("PixelSpacing", "ImagerPixelSpacing") and min(mm) < DXA_MIN_MM_PER_PX:
                # a real spacing tag this fine means a radiograph (0.12-0.14 mm), not DXA (0.6 mm)
                # violation_type is a closed vocabulary (разъяснения V2), so the reason goes to details
                row.update(anatomical_region="не определена", quality_class=0, processing_status="Success",
                           violation_type="", violation_codes="unsupported",
                           details=f"оценка не проводится: размер пикселя {min(mm):.2f} мм — это не денситометрия "
                                   f"({mm[0]:.3f}x{mm[1]:.3f} мм из тега {mm_src})")
                row["time_of_processing"] = round(time.time() - t0, 3)
                return row
            r = self.analyse(img, mm, rotation_override, policy)
            row["mm_per_px"] = f"{mm[0]:.3f}x{mm[1]:.3f}" if mm[0] != mm[1] else round(mm[0], 4)
            row["mm_per_px_source"] = mm_src
            if r.get("unsupported"):
                row.update(anatomical_region="не определена", quality_class=0, violation_type="", violation_codes="unsupported",
                           details=f"оценка не проводится: {r['reason']} ("
                                   + "; ".join(f"{k} {v:.2f}" for k, v in r["checks"].items()) + ")",
                           processing_status="Success")
                row["time_of_processing"] = round(time.time() - t0, 3)
                return row
            row["anatomical_region"] = REGION_RU[r["region"]]
            row["_region"] = r["region"]
            row["_rotation"] = r.get("measurements", {}).get("rotation_own")
            row.update(quality_class=r["quality_class"], quality_prob=round(r["score"], 4),
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
                if m.get("printed_markup"):
                    bits.append("на снимке впечатана разметка аппарата: инородные тела не оцениваются")
                note = describe_markup(m.get("markup_review"))
                if note:
                    bits.append(note)
                row["details"] = "; ".join(bits)
            else:
                ru = {"top": "сверху (от большого вертела)", "bottom": "снизу", "lateral": "сбоку"}
                bits = [f"поле {ru[k]} {m[f'margin_{k}']:.0f} мм" for k in ("top", "bottom", "lateral")
                        if m.get(f"margin_{k}") is not None]
                if m.get("shaft_angle") is not None:
                    bits.append(f"ось бедра {m['shaft_angle']:.0f}° к вертикали")
                if m.get("implant"):
                    bits.append("эндопротез (укладка оценена как у обычного бедра)")
                if m.get("printed_markup"):
                    bits.append("на снимке впечатана разметка аппарата")
                row["details"] = "; ".join(bits)
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
