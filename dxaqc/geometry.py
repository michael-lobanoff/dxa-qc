"""Quality rules computed from anatomical keypoints.

The same functions label synthetic violations and run inside the QC engine,
so a synthetic sample is "bad" exactly when the production rule would say so.
Keypoints are dicts name -> [x, y] in image pixels, or None when the point is
outside the frame / not visible (see docs/annotation_scheme.md).
"""
import math

# Fallback pixel size for GE Lunar Prodigy exports; per image it is read from the DICOM Exposed Area
# tag (see pixels.py), which is why every rule below takes mm_per_px as a parameter.
MM_PER_PX = 0.607
MAX_TILT_DEG = 5.0          # ТЗ 2.3: допустимый наклон оси до 5°
# ТЗ 2.3: the scan must reach the middle of Th12, i.e. ~4.5 vertebrae above the L4/L5 level marked by
# the iliac crests. Measured per patient in vertebral pitches (vertebrae.find); the mm value below is
# only the fallback for images where the vertebral train is not found.
MIN_SPINE_SPAN_VERT = 3.2   # conservative: no training scan was called short at the top (min 3.5)
MIN_SPINE_SPAN_MM = 115.0   # ~4.5 vertebrae at the measured pitch of ~30 mm
HIP_MARGIN_TOP_MM = 30.0    # ТЗ 2.3, рис. 6: 3 см над большим вертелом, 3 см под малым, 2 см сбоку
HIP_MARGIN_BOTTOM_MM = 30.0
HIP_MARGIN_LATERAL_MM = 20.0

# Annotation scheme v2: the spine is described by its centre line and the iliac crests. Individual
# vertebral levels are not annotated — on low-resolution DXA their names are ambiguous without the
# ribs, and an axis fitted through per-vertebra midlines was far noisier than this two-point axis
# (AUC 0.57 vs 0.89). The vertebrae are still located (vertebrae.py), but only as a ruler.
SPINE_POINTS = ["col_top", "col_bottom", "crest_a", "crest_b"]
HIP_POINTS = ["fh_c", "fh_top", "fn_c", "gt_top", "gt_lat", "lt", "isch", "shaft_p", "shaft_d"]


def axes_mm(spacing):
    """Accept a single mm/px value or an (x, y) pair; DXA exports may in principle be anisotropic."""
    if spacing is None:
        return MM_PER_PX, MM_PER_PX
    if isinstance(spacing, (int, float)):
        return float(spacing), float(spacing)
    return float(spacing[0]), float(spacing[1])


def inside(p, shape) -> bool:
    h, w = shape
    return p is not None and 0 <= p[0] < w and 0 <= p[1] < h


def first(kp, names):
    return next((kp[k] for k in names if kp.get(k) is not None), None)


def spine_tilt_deg(kp):
    """Signed angle between the vertical and the spine centre line (bottom → top).

    Positive when the top of the spine leans to the image right. Mirrors the expert
    measurement in ТЗ рис. 2 (line along the spine vs. the vertical).
    """
    top, bottom = kp.get("col_top"), kp.get("col_bottom")
    if top is None or bottom is None or bottom[1] - top[1] < 1:
        return None
    return math.degrees(math.atan2(top[0] - bottom[0], bottom[1] - top[1]))


def spine_span_px(kp):
    """Distance from the iliac crest level up to the top edge of the scan, in pixels."""
    crests = [kp[k][1] for k in ("crest_a", "crest_b") if kp.get(k) is not None]
    return None if not crests else min(crests)


def spine_span_mm(kp, mm_per_px=MM_PER_PX):
    span = spine_span_px(kp)
    return None if span is None else span * axes_mm(mm_per_px)[1]     # vertical distance


def spine_rules(kp, shape, mm_per_px=MM_PER_PX, pitch_px=None):
    """pitch_px: vertebral pitch from vertebrae.find, which turns the upper-coverage criterion into
    a count of vertebrae; without it the rule falls back to a distance in mm."""
    tilt = spine_tilt_deg(kp)
    crests_ok = inside(kp.get("crest_a"), shape) and inside(kp.get("crest_b"), shape)
    span, span_px = spine_span_mm(kp, mm_per_px), spine_span_px(kp)
    span_vert = None if (pitch_px is None or span_px is None or pitch_px <= 0) else span_px / pitch_px
    if span_vert is not None:
        top_ok = span_vert >= MIN_SPINE_SPAN_VERT
    else:
        top_ok = span is None or span >= MIN_SPINE_SPAN_MM  # unknown when the crests are missing
    return {
        "tilt_deg": tilt,
        "v_axis": None if tilt is None else abs(tilt) > MAX_TILT_DEG,
        "crests_in_frame": crests_ok,
        "span_mm": span,
        "span_vert": span_vert,
        "top_coverage_ok": top_ok,
        "v_pos": not (crests_ok and top_ok),
    }


def hip_margins_mm(kp, shape, side, mm_per_px=MM_PER_PX):
    """Distances from the femoral ROI to the scan edges. side: 'hip_left' | 'hip_right'.

    In the AP view the right hip has the femoral head on the image right, so its
    lateral side (greater trochanter) faces the image left.

    The top margin is measured from the tip of the GREATER TROCHANTER: ТЗ asks for 3 cm above it (the
    wording repeated at the organisers' Q&A, 17.09). It used to be taken from the top of the femoral
    head, and then 45 of 143 normal hips "violated" the 3 cm rule — the rule and the experts disagreed.
    From the trochanter only 1 of 143 does, and the ROI type improves on every detector checked
    (AUC 0.865 -> 0.930 on the shipped ensemble, 0.91 -> 0.93 on two detectors outside any selection).
    """
    h, w = shape
    sx, sy = axes_mm(mm_per_px)
    gt, low, lat = kp.get("gt_top"), first(kp, ["lt", "shaft_p"]), kp.get("gt_lat")
    return {
        "top": None if gt is None else gt[1] * sy,
        "bottom": None if low is None else (h - low[1]) * sy,
        "lateral": None if lat is None else (lat[0] if side == "hip_right" else w - lat[0]) * sx,
    }


def hip_rules(kp, shape, side, mm_per_px=MM_PER_PX):
    m = hip_margins_mm(kp, shape, side, mm_per_px)
    limits = {"top": HIP_MARGIN_TOP_MM, "bottom": HIP_MARGIN_BOTTOM_MM, "lateral": HIP_MARGIN_LATERAL_MM}
    # A structure pushed out of the frame means a zero margin.
    short = {k: (m[k] is None or m[k] < limits[k]) for k in limits}
    visible = {k: inside(kp.get(k), shape) for k in ("gt_top", "fn_c", "isch")}
    return {
        "margins_mm": m,
        "v_roi": any(short.values()),
        "short_margins": [k for k, v in short.items() if v],
        "v_pos_structures": not all(visible.values()),
        "missing_structures": [k for k, v in visible.items() if not v],
    }
