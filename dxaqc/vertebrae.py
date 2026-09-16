"""Vertebral bodies along the lumbar column.

On these exports the trabecular centre of a vertebral body is darker than its cortical endplates, so
the column reads along its axis as a regular train of dark blocks. On a public DXA set from another
centre the chain does not lock onto the right phase at all (its rendering differs), and choosing the
polarity automatically by response strength did not fix it — measured against the ROI boxes that
scanner printed on its images, the phase stayed uniformly distributed. So this module is validated on
the organisers' exports only; see docs/night_report.md. Fitting that train gives the
vertebral pitch (body + disc) — a per-patient anatomical ruler — and the positions of the bodies.

The pitch is what the upper-coverage criterion of ТЗ 2.3 needs: the scan must reach the middle of
Th12, i.e. about 4.5 vertebrae above the L4/L5 level marked by the iliac crests. Expressed in
vertebrae this holds for any patient size and does not depend on the pixel size.

What this module deliberately does NOT do: name the levels (L1, L2 …) or measure the axis. Level
names are ambiguous on DXA without the ribs (the intercristal line crosses L4 or the L4/L5 disc
depending on the patient), and a tilt fitted through per-vertebra midlines proved much noisier than
the annotated two-point axis (ROC-AUC 0.57 vs 0.89 against the expert axis labels), so the axis rule
keeps using col_top / col_bottom.
"""
import numpy as np

from .geometry import axes_mm
from .pixels import DEFAULT_MM_PER_PX

# Anatomical bounds on the vertebral pitch (body + disc) of an adult lumbar spine. Searching in mm
# instead of px matters: a px-only range lets the fit lock onto both bodies and endplates at once
# (double frequency), which put 12 of 99 training scans at an impossible 21-24 mm pitch.
PITCH_MM = (26.0, 40.0)
LAM = 15.0                           # weight of the spacing-regularity prior
HALF = 16                            # half width of the profile band, px
PAD = 70                             # how far past the annotated column ends to look, px


def _smooth(a, w):
    if w % 2 == 0:
        w += 1
    k = np.ones(w) / w
    return np.convolve(np.pad(a, w // 2, mode="edge"), k, "same")[w // 2:w // 2 + len(a)]


def _kernel(period):
    """Dark vertebral body between two bright endplate bands."""
    h = int(round(period / 2))
    k = np.zeros(2 * h + 1)
    b = max(3, int(round(period * 0.30)))
    e = max(2, int(round(period * 0.12)))
    k[h - b:h + b + 1] = -1.0 / (2 * b + 1)
    k[:e] = k[-e:] = 0.5 / e
    return k, h


def _response(d, period):
    k, h = _kernel(period)
    return np.convolve(np.pad(d, h, mode="edge"), k[::-1], "same")[h:h + len(d)]


def _refine(t, ts, r, P, valid, iters=8):
    """Coordinate descent: strong filter response, spacing close to the current pitch."""
    t = list(map(float, t))
    for _ in range(iters):
        for i in range(len(t)):
            grid = np.arange(t[i] - 0.28 * P, t[i] + 0.28 * P, 0.5)
            grid = grid[(grid > ts[0]) & (grid < ts[-1])]
            if not len(grid):
                continue
            s = np.interp(grid, ts, r) + 3.0 * (np.interp(grid, ts, valid.astype(float)) - 1)
            if i > 0:
                s -= LAM * ((grid - t[i - 1] - P) / P) ** 2
            if i < len(t) - 1:
                s -= LAM * ((t[i + 1] - grid - P) / P) ** 2
            t[i] = float(grid[int(np.argmax(s))])
        if len(t) > 1:
            P = float(np.median(np.diff(t)))
    return np.array(t), P


def find(img, points, mm_per_px=None):
    """Vertebral bodies from the image and the annotated/predicted spine axis.

    Returns None when the axis is missing or no regular train is found, else
    {points: [[x, y], ...] top to bottom, pitch_px, irregularity, n}.
    """
    mm = DEFAULT_MM_PER_PX if not mm_per_px else axes_mm(mm_per_px)[1]   # pitch runs along the spine
    periods = np.arange(PITCH_MM[0] / mm, PITCH_MM[1] / mm, 1.0)
    top, bot = points.get("col_top"), points.get("col_bottom")
    if not top or not bot:
        return None
    top, bot = np.asarray(top, float), np.asarray(bot, float)
    v = bot - top
    L = float(np.hypot(*v))
    if L < 3 * periods[0]:
        return None
    u = v / L
    n = np.array([-u[1], u[0]])
    ts = np.arange(-PAD, L + PAD, 1.0)
    offs = np.arange(-HALF, HALF + 1.0)
    pts = top[None, None, :] + ts[:, None, None] * u + offs[None, :, None] * n
    inside = ((pts[..., 0] >= 0) & (pts[..., 0] < img.shape[1]) &
              (pts[..., 1] >= 0) & (pts[..., 1] < img.shape[0])).all(1)
    if inside.sum() < 3 * periods[0]:
        return None
    x = np.clip(np.rint(pts[..., 0]).astype(int), 0, img.shape[1] - 1)
    y = np.clip(np.rint(pts[..., 1]).astype(int), 0, img.shape[0] - 1)
    prof = img[y, x].astype(np.float32).mean(1)
    d0 = prof - _smooth(prof, 91)
    d0 = (d0 - d0.mean()) / (d0.std() + 1e-6)
    valid = inside & (_smooth(prof, 41) > np.percentile(prof[inside], 60) * 0.72)
    d = d0
    best = None
    for P in periods:                                   # comb initialisation
        r = _response(d, P)
        for ph in np.arange(0, P, 1.0):
            pos = np.arange(ts[0] + ph, ts[-1], P)
            pos = pos[np.interp(pos, ts, valid.astype(float)) > 0.99]
            if len(pos) < 4:
                continue
            s = float(np.interp(pos, ts, r).mean() + 0.05 * len(pos))
            if best is None or s > best[0]:
                best = (s, P, pos)
    if best is None:
        return None
    _, P, pos = best
    r = _response(d, P)
    t, P = _refine(pos, ts, r, P, valid)
    while len(t) > 3 and np.interp(t[0], ts, r) < 0.05:
        t = t[1:]
    while len(t) > 3 and np.interp(t[-1], ts, r) < 0.05:
        t = t[:-1]
    if len(t) < 4:
        return None
    sp = np.diff(t)
    return {"points": [(top + ti * u).tolist() for ti in t],
            "pitch_px": float(np.median(sp)),
            "irregularity": float(np.std(sp) / np.mean(sp)),
            "n": int(len(t))}
