"""Overlay of the QC result on the scan (ТЗ 2.6: additional series with the detected violation)."""
import datetime

import cv2
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from .artifacts import artifact_mask
from .decision import NAMES_RU
from .geometry import MM_PER_PX

GREEN, RED, YELLOW, CYAN, WHITE = (90, 209, 122), (255, 90, 90), (255, 216, 76), (76, 194, 255), (255, 255, 255)
SCALE = 3


def _pt(p):
    return int(round(p[0] * SCALE)), int(round(p[1] * SCALE))


def _with_header(img, lines, color):
    """Text goes into a black band above the scan so it never covers anatomy."""
    band = np.zeros((22 * len(lines) + 12, img.shape[1], 3), np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(band, line, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return np.vstack([band, img])


def render_overlay(img, result):
    """RGB uint8 image (3x upscaled) with landmarks, measurements and the verdict. Text is ASCII: OpenCV fonts
    have no Cyrillic, the Russian wording goes to the report and the DICOM SR/metadata instead."""
    h, w = img.shape
    rgb = cv2.cvtColor(cv2.resize(img, (w * SCALE, h * SCALE), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB)
    p, region = result["points"], result["region"]
    bad = bool(result.get("quality_class"))
    if region == "spine":
        if p.get("col_top") and p.get("col_bottom"):
            cv2.line(rgb, _pt(p["col_top"]), _pt(p["col_bottom"]), RED if "v_axis" in result["violations"] else CYAN, 2, cv2.LINE_AA)
            cv2.line(rgb, (_pt(p["col_bottom"])[0], _pt(p["col_top"])[1]), _pt(p["col_bottom"]), WHITE, 1, cv2.LINE_AA)
        for k in ("crest_a", "crest_b"):
            if p.get(k):
                cv2.circle(rgb, _pt(p[k]), 7, GREEN, 2, cv2.LINE_AA)
        for v in result["measurements"].get("vert_points") or []:   # vertebral bodies: the ruler behind
            x, y = _pt(v)                                           # the "half of Th12 in frame" check
            cv2.line(rgb, (x - 45, y), (x + 45, y), GREEN, 1, cv2.LINE_AA)
        m = artifact_mask(img, p)
        if m.any():
            cnts, _ = cv2.findContours(cv2.resize(m.astype(np.uint8), (w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb, cnts, -1, RED if "v_artifact" in result["violations"] else YELLOW, 2)
        m2 = result["measurements"]
        tilt, sv = m2.get("tilt_deg"), m2.get("span_vert")
        lines = [("SPINE" if tilt is None else f"SPINE  tilt {tilt:+.1f} deg")
                 + ("" if sv is None else f"  above crests {sv:.1f} vert ({m2['vert_pitch_mm']:.0f} mm each)")]
    else:
        meas = result.get("measurements", {})
        for k, v in p.items():
            if v:
                cv2.circle(rgb, _pt(v), 5, CYAN, -1, cv2.LINE_AA)
        if p.get("gt_top"):   # top margin: from the tip of the greater trochanter (geometry.hip_margins_mm)
            x, y = _pt(p["gt_top"]); cv2.arrowedLine(rgb, (x, y), (x, 0), YELLOW, 1, cv2.LINE_AA, tipLength=0.05)
        low = p.get("lt") or p.get("shaft_p")
        if low:
            x, y = _pt(low); cv2.arrowedLine(rgb, (x, y), (x, h * SCALE - 1), YELLOW, 1, cv2.LINE_AA, tipLength=0.05)
        if p.get("gt_lat"):
            x, y = _pt(p["gt_lat"]); edge = 0 if region == "hip_right" else w * SCALE - 1
            cv2.arrowedLine(rgb, (x, y), (edge, y), YELLOW, 1, cv2.LINE_AA, tipLength=0.05)
        side = "LEFT" if region == "hip_left" else "RIGHT"
        lines = [f"HIP {side}  margins top/bottom/lateral: " + "/".join(
            "-" if meas.get(f"margin_{k}") is None else f"{meas[f'margin_{k}']:.0f}" for k in ("top", "bottom", "lateral")) + " mm"]
        if meas.get("rotation") is not None:
            lines.append(f"rotation score {meas['rotation']:.2f}")
    for r in result.get("measurements", {}).get("rois") or []:      # proposed measurement region (ТЗ 2.6)
        x0, y0, x1, y1 = (int(round(v * SCALE)) for v in r["box"])
        cv2.rectangle(rgb, (x0, y0), (x1, y1), CYAN, 1, cv2.LINE_AA)
        cv2.putText(rgb, r["level"], (x0 + 4, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, CYAN, 1, cv2.LINE_AA)
    if result.get("implant"):
        lines.append("endoprosthesis: positioning judged as for any hip")
    codes = {"v_axis": "axis tilt", "v_pos": "positioning (iliac crests)", "v_artifact": "foreign body",
             "v_roi": "field margins", "v_posrot": "positioning/rotation"}
    lines.append("QUALITY: " + (", ".join(codes[v] for v in result["violations"]) if bad else "OK")
                 + (f"  (p={result['score']:.2f})" if result.get("score") is not None else ""))
    color = RED if bad else GREEN
    return _with_header(rgb, lines, color)


def to_secondary_capture(rgb, ref: pydicom.Dataset, description="DXA QC overlay"):
    """Secondary Capture in the same study, new series; identifying attributes copied from the source image."""
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SpecificCharacterSet = "ISO_IR 192"   # copied attributes carry Russian text (StudyDescription)
    for tag in ("PatientID", "PatientName", "PatientBirthDate", "PatientSex", "StudyInstanceUID", "StudyDate",
                "StudyTime", "AccessionNumber", "StudyID", "StudyDescription"):
        if tag in ref:
            setattr(ds, tag, getattr(ref, tag))
    now = datetime.datetime.now()
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID, ds.SeriesNumber, ds.InstanceNumber = generate_uid(), 999, getattr(ref, "InstanceNumber", 1)
    ds.Modality, ds.ConversionType, ds.SeriesDescription = "OT", "WSD", description
    ds.ContentDate, ds.ContentTime = now.strftime("%Y%m%d"), now.strftime("%H%M%S")
    ds.ReferencedImageSequence = [pydicom.Dataset()]
    ds.ReferencedImageSequence[0].ReferencedSOPClassUID = getattr(ref, "SOPClassUID", "")
    ds.ReferencedImageSequence[0].ReferencedSOPInstanceUID = getattr(ref, "SOPInstanceUID", "")
    ds.SamplesPerPixel, ds.PhotometricInterpretation, ds.PlanarConfiguration = 3, "RGB", 0
    ds.Rows, ds.Columns = rgb.shape[:2]
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    ds.PixelData = np.ascontiguousarray(rgb).tobytes()
    return ds
