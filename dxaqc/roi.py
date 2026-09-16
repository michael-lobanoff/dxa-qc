"""Proposed measurement regions (ТЗ 2.6: автоматическая коррекция разметки).

A densitometer measures bone density inside boxes it places itself: L1-L4 on the lumbar spine and a
neck/trochanter box on the femur. The organisers' exports are raw — no boxes are drawn on them — so
the service proposes where those boxes belong, for a specialist to confirm or correct.

Spine: the boxes are stacked between the midpoints of consecutive vertebral bodies (vertebrae.find),
anchored on the iliac crest line, which published DXA guidance places at the L4/L5 interspace
(AJR 2011; J Nucl Med Technol 2023). The proposal is therefore only as certain as that anchor —
it is a suggestion for a human, not a measurement.
"""
import numpy as np

LEVELS = ["L4", "L3", "L2", "L1"]      # upwards from the crest line
WIDTH_MARGIN = 1.15                     # box a bit wider than the vertebral body


def _body_width(img, centre, pitch, half=70):
    """Width of the bright vertebral body at one level, in pixels."""
    y, x = int(round(centre[1])), int(round(centre[0]))
    y0, y1 = max(y - int(pitch * 0.15), 0), min(y + int(pitch * 0.15) + 1, img.shape[0])
    if y1 - y0 < 3:
        return None
    row = img[y0:y1].astype(np.float32).mean(0)
    row = np.convolve(row, np.ones(5) / 5, "same")
    lo, hi = np.percentile(row, 10), row.max()
    on = np.where(row > lo + 0.5 * (hi - lo))[0]
    on = on[(on > x - half) & (on < x + half)]
    return float(on.max() - on.min()) if len(on) > 5 else None


def spine_rois(img, points, vert):
    """[{level, box: [x0, y0, x1, y1]}] for L1-L4, or [] when the anchor is missing.

    vert: the dict from vertebrae.find (body centres, top to bottom).
    """
    if not vert or len(vert["points"]) < 3:
        return []
    crests = [points[k][1] for k in ("crest_a", "crest_b") if points.get(k)]
    if not crests:
        return []
    centres = [np.asarray(p, float) for p in vert["points"]]
    pitch = vert["pitch_px"]
    mids = [(centres[i] + centres[i + 1]) / 2 for i in range(len(centres) - 1)]
    # extend by half a pitch at both ends so the lowest and highest bodies also get a boundary
    mids = [centres[0] - np.array([0, pitch / 2])] + mids + [centres[-1] + np.array([0, pitch / 2])]
    crest_y = min(crests)
    k = int(np.argmin([abs(m[1] - crest_y) for m in mids]))       # boundary nearest the crest line = L4/L5
    widths = [w for w in (_body_width(img, c, pitch) for c in centres) if w]
    width = (np.median(widths) if widths else 1.3 * pitch) * WIDTH_MARGIN
    out = []
    for i, level in enumerate(LEVELS):
        lo, hi = k - i, k - i - 1                                  # boundaries below and above this body
        if hi < 0 or lo >= len(mids):
            break
        y0, y1 = float(mids[hi][1]), float(mids[lo][1])
        cx = float((mids[hi][0] + mids[lo][0]) / 2)
        out.append({"level": level, "box": [cx - width / 2, y0, cx + width / 2, y1]})
    return out[::-1]


def hip_roi(points, shape):
    """One box around the measured part of the proximal femur (head, neck, trochanters)."""
    need = ("fh_top", "gt_top", "gt_lat", "isch")
    have = [points.get(k) for k in need if points.get(k)]
    low = points.get("lt") or points.get("shaft_p")
    if len(have) < 3 or not low:
        return None
    xs = [p[0] for p in have] + [low[0]]
    ys = [p[1] for p in have] + [low[1]]
    pad = 0.06 * shape[1]
    return {"level": "femur", "box": [max(min(xs) - pad, 0), max(min(ys) - pad, 0),
                                      min(max(xs) + pad, shape[1] - 1), min(max(ys) + pad, shape[0] - 1)]}
