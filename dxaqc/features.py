"""Measurements derived from anatomical keypoints (shared by evaluation, classifiers and the service)."""
import math

import numpy as np

from . import geometry as G


def _dist_to_line(p, a, b):
    """Signed distance from p to the line a->b (positive to the right of the direction a->b)."""
    (ax, ay), (bx, by) = a, b
    return ((bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax)) / max(math.hypot(bx - ax, by - ay), 1e-6)


def hip_features(p, shape, side, mm_per_px=None):
    """Positioning / rotation cues. Orientation is normalised so that +x points medially (towards the pelvis).

    Distances are computed on millimetre coordinates (x*sx, y*sy); angles stay in pixel coordinates,
    because that is what the expert sees on the displayed image and judges the criteria against.
    """
    sx, sy = G.axes_mm(mm_per_px)
    f = {}
    m = G.hip_margins_mm(p, shape, side, mm_per_px)
    f.update({f"margin_{k}": v for k, v in m.items()})
    s = -1 if side == "hip_left" else 1
    q = {k: (None if v is None else (s * v[0] * sx, v[1] * sy)) for k, v in p.items()}   # mm, medial = +x
    qpx = {k: (None if v is None else (s * v[0], v[1])) for k, v in p.items()}           # pixels, for angles
    sp, sd = q.get("shaft_p"), q.get("shaft_d")
    f["lt_absent"] = float(q.get("lt") is None)
    f["missing"] = float(sum(q.get(k) is None for k in ("gt_top", "fn_c", "isch")))
    if sp and sd:
        # ISCD / JNMT positioning guidance: the long axis of the femur should be parallel to the long
        # axis of the table, i.e. vertical on the image. Reported as a measurement; on its own it
        # separates rotation labels only weakly (AUC 0.67) and adds nothing to the rotation model.
        f["shaft_angle"] = abs(math.degrees(math.atan2(qpx["shaft_d"][0] - qpx["shaft_p"][0],
                                                       qpx["shaft_d"][1] - qpx["shaft_p"][1])))
        # the shaft axis runs downwards; medial side = right of the upward direction sd->sp
        if q.get("lt"):
            f["lt_protrusion"] = -_dist_to_line(q["lt"], sd, sp)
        if q.get("fh_c"):
            f["neck_offset"] = -_dist_to_line(q["fh_c"], sd, sp)   # femoral offset in projection
        if q.get("fh_c") and q.get("fn_c"):
            shaft = np.subtract(sp, sd); neck = np.subtract(q["fh_c"], q["fn_c"])
            cos = np.dot(shaft, neck) / (np.linalg.norm(shaft) * np.linalg.norm(neck) + 1e-6)
            f["neck_shaft_angle"] = 180 - math.degrees(math.acos(np.clip(cos, -1, 1)))
    if q.get("fh_c") and q.get("gt_lat"):
        f["head_gt_width"] = abs(q["fh_c"][0] - q["gt_lat"][0])
    return f


def spine_features(p, shape, mm_per_px=None):
    r = G.spine_rules(p, shape, G.MM_PER_PX if mm_per_px is None else mm_per_px)
    return {"abs_tilt": None if r["tilt_deg"] is None else abs(r["tilt_deg"]), "crests_missing": float(not r["crests_in_frame"]),
            "span_mm": r["span_mm"]}


def pelvis_relative_tilt(points, mm_per_px=None):
    """Angle between the column axis and the perpendicular to the line joining the iliac crests, degrees.

    The tilt of ТЗ 2.3 is measured against the vertical of the frame, i.e. the axis of the table. This
    measures something different on purpose: how much the spine leans **relative to the pelvis**. When the
    whole patient lies askew, spine and pelvis lean together and this stays small; when the spine alone
    curves away from the pelvis, it grows. The expert flags the first and not the second, so on its own
    this separates his labels backwards (AUC 0.46) — which is exactly why it is useful next to the plain
    tilt: it tells a crooked lay from a crooked spine.
    """
    sx, sy = G.axes_mm(mm_per_px)
    a, b = points.get("crest_a"), points.get("crest_b")
    t, bo = points.get("col_top"), points.get("col_bottom")
    if not a or not b or not t or not bo or abs(b[0] - a[0]) < 1 or abs(bo[1] - t[1]) < 1:
        return None
    pelvis = math.degrees(math.atan2((b[1] - a[1]) * sy, (b[0] - a[0]) * sx))
    column = math.degrees(math.atan2((bo[0] - t[0]) * sx, (bo[1] - t[1]) * sy))
    return abs(column + pelvis)


def pelvis_ratio(img, points, band=0.22, gap=1.6):
    """How much bright bone sits in the lower corners of the frame, relative to the column itself.

    ТЗ 2.3 wants the upper edges of the iliac wings in the frame, and a correctly framed lumbar scan
    shows them as two large bright areas in the lower corners. The shipped signal for this is the
    detector's confidence in the crest landmarks; it is accurate on the scanner it was trained on
    (0.99 when the crests are there, 0.31 when they are not) but does not travel: on a public set from
    another densitometer, cropping the crests away moves it from 0.97 only to 0.88. This ratio is
    measured from pixels alone and reacts on every one of those scans (0.50 -> 0.28, 100 % of images).
    """
    h, w = img.shape
    t, b = points.get("col_top"), points.get("col_bottom")
    if not t or not b:
        return None
    rows = np.arange(int(h * (1 - band)), h)
    if len(rows) < 5:
        return None
    g = img.astype(np.float32)
    denom = (b[1] - t[1]) or 1
    xs = t[0] + (b[0] - t[0]) * (rows - t[1]) / denom      # the axis, extrapolated into those rows
    half = max(w * 0.08, 12)                               # half-width of the lumbar column, pixels
    col, side = [], []
    for y, x in zip(rows, xs):
        lo, hi = int(max(x - half, 0)), int(min(x + half, w - 1))
        if hi > lo:
            col.append(g[y, lo:hi + 1].mean())
        vals = np.concatenate([g[y, :max(int(x - gap * half), 0)], g[y, min(int(x + gap * half), w - 1):]])
        if vals.size:
            side.append(vals.mean())
    if not col or not side:
        return None
    c, s = float(np.mean(col)), float(np.mean(side))
    return s / c if c else None


def spine_curvature(img, points, window=60, mm_per_px=None):
    """Largest lateral deviation (mm) of the spinal column centre from the straight col_top -> col_bottom line.

    Positioning tilt rotates a straight column; scoliosis bends it. Experts flag only the former.
    """
    import cv2
    from scipy.ndimage import median_filter

    mm_per_px = G.axes_mm(mm_per_px)[0]      # lateral deviation: horizontal scale
    t, b = points.get("col_top"), points.get("col_bottom")
    if not t or not b or b[1] - t[1] < 30:
        return None
    g = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 3)
    h, w = g.shape
    ys = np.arange(int(max(t[1], 0)) + 5, int(min(b[1], h - 1)) - 5)
    dev = []
    for y in ys:
        xl = t[0] + (b[0] - t[0]) * (y - t[1]) / (b[1] - t[1])
        lo, hi = int(max(xl - window, 0)), int(min(xl + window, w - 1))
        row = g[y, lo:hi + 1]
        if row.size < 10:
            continue
        thr = np.percentile(row, 20) + 0.5 * (row.max() - np.percentile(row, 20))
        on = np.where(row > thr)[0]
        dev.append(lo + (on.min() + on.max()) / 2 - xl)
    if len(dev) < 20:
        return None
    dev = median_filter(np.asarray(dev), size=15, mode="nearest")
    return float(np.abs(dev).max() * mm_per_px)
