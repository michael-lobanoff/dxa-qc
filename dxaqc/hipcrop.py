"""Pose-normalised crops of the proximal femur for the rotation / positioning classifier.

The crop is taken in the normalised orientation (femoral head on the image right): the shaft
axis is rotated to vertical and the head-to-shaft distance is scaled to a fixed length, so the
classifier sees the lesser trochanter, neck and greater trochanter in comparable places.
In-plane alignment only: the out-of-plane rotation we want to detect stays in the image.
"""
import math

import cv2
import numpy as np

from .hog import hog

CROP = 128
IMAGE_AXIS = True   # align the shaft by the axis traced on the image, not by two landmarks
HEAD_SHAFT_PX = 62  # distance fh_c -> shaft_p in the crop


def _get(p, k):
    v = p.get(k)
    return None if v is None else np.asarray(v, np.float32)


def normalise(img, points, side):
    """Mirror left hips so the head is on the right; returns (image, points)."""
    if side != "hip_left":
        return img, points
    w = img.shape[1]
    return img[:, ::-1].copy(), {k: (None if v is None else [w - 1 - v[0], v[1]]) for k, v in points.items()}


def shaft_axis_angle(img, sp, sd, halfwidth=40):
    """Tilt of the femoral shaft from vertical (degrees), traced on the image: the midpoint of the bright
    bone in every row between shaft_p and shaft_d, then a robust straight line through them.

    The two landmarks alone give the axis with a median error of 4.3° (p90 12°): shaft_p is the noisiest
    point of the detector, ~11 px. Traced on the bone, the angle hardly depends on where a detector put
    the points (0.25° between detectors, p90 1°), so the rotation crop stops depending on which detector
    placed it. Returns None when the shaft cannot be traced (the caller falls back to the landmarks).
    """
    if sp is None or sd is None or sd[1] - sp[1] < 30:
        return None
    g = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 2)
    h, w = g.shape
    mids = []
    for y in range(int(sp[1]), int(min(sd[1], h - 1))):
        xl = sp[0] + (sd[0] - sp[0]) * (y - sp[1]) / (sd[1] - sp[1])
        lo, hi = int(max(xl - halfwidth, 0)), int(min(xl + halfwidth, w - 1))
        row = g[y, lo:hi + 1]
        if row.size < 20:
            continue
        base = np.percentile(row, 20)
        on = np.where(row > base + 0.5 * (row.max() - base))[0]
        if len(on) >= 5:
            mids.append((lo + (on.min() + on.max()) / 2, y))
    if len(mids) < 20:
        return None
    m = np.array(mids)
    a, b = np.polyfit(m[:, 1], m[:, 0], 1)
    r = np.abs(m[:, 0] - (a * m[:, 1] + b))
    keep = r < 2.5 * (np.median(r) + 1e-6)
    a, _ = np.polyfit(m[keep, 1], m[keep, 0], 1)
    return math.degrees(math.atan2(-a, 1.0))


def crop_matrix(points, shape, angle=None):
    """angle: shaft tilt traced on the image (shaft_axis_angle); None = from the landmarks."""
    fh, sp, sd = _get(points, "fh_c"), _get(points, "shaft_p"), _get(points, "shaft_d")
    if fh is None or sp is None:
        h, w = shape  # failed scan: plain resize of the whole frame
        s = CROP / max(h, w)
        return np.array([[s, 0, (CROP - w * s) / 2], [0, s, (CROP - h * s) / 2]], np.float32)
    up = (sp - sd) if sd is not None else np.array([0, -1], np.float32)
    ang = math.degrees(math.atan2(up[0], -up[1])) if angle is None else angle   # tilt of the shaft from vertical
    scale = HEAD_SHAFT_PX / max(np.linalg.norm(fh - sp), 1.0)
    others = [v for v in (_get(points, "fn_c"), _get(points, "gt_top"), fh, sp) if v is not None]
    centre = np.mean(others, axis=0)
    m = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])), ang, scale)
    m[:, 2] += np.array([CROP / 2, CROP / 2 + 6]) - centre   # centre of the femur slightly above the middle
    return m.astype(np.float32)


ZOOM_BOX = (40, 120, 44, 124)      # lesser trochanter and the neck above it, inside the normalised crop
ZOOM_SIZE = 64
CELL_WIDE = 20                     # coarse grid over the whole femur
CELL_NEAR = 6                      # fine grid over the close-up


def rotation_features(crop):
    """Descriptor for the rotation classifier: a coarse grid over the whole femur plus a fine grid over
    a close-up of the region the criterion is about (neck and lesser trochanter).

    The two grid sizes were swept over 20 combinations and the winner confirmed on seeds it was not
    selected on: 0.852 against 0.828 for the earlier 16/8 pair. Shape at the scale of the whole bone
    plus texture where the trochanter is — the coarse/fine split matters more than either alone.
    The close-up was moved 10 px down (more lesser trochanter, less neck) once the crop was aligned by
    the traced shaft axis: chosen on two detector runs, confirmed on a third that took no part in the
    choice — rotation AUC 0.833 -> 0.863, PR-AUC 0.597 -> 0.734, F1 0.625 -> 0.668."""
    y0, y1, x0, x1 = ZOOM_BOX
    near = cv2.resize(crop[y0:y1, x0:x1], (ZOOM_SIZE, ZOOM_SIZE), interpolation=cv2.INTER_LINEAR)
    return np.concatenate([hog(crop, CELL_WIDE), hog(near, CELL_NEAR)])


def hip_crop(img, points, side, jitter=None, image_axis=None):
    """uint8 crop (CROP x CROP). jitter: optional (dx, dy, dang, dscale) augmentation.
    image_axis: align by the traced shaft axis (default IMAGE_AXIS); a fitted rotation model records which
    alignment it was trained on, and the service passes that on."""
    im, p = normalise(img, points, side)
    use = IMAGE_AXIS if image_axis is None else image_axis
    m = crop_matrix(p, im.shape, shaft_axis_angle(im, _get(p, "shaft_p"), _get(p, "shaft_d")) if use else None)
    if jitter is not None:
        dx, dy, dang, dscale = jitter
        a = cv2.getRotationMatrix2D((CROP / 2, CROP / 2), dang, dscale)
        a[:, 2] += (dx, dy)
        m = (np.vstack([a, [0, 0, 1]]) @ np.vstack([m, [0, 0, 1]]))[:2].astype(np.float32)
    out = cv2.warpAffine(np.ascontiguousarray(im), m, (CROP, CROP), flags=cv2.INTER_LINEAR, borderValue=0)
    return out, m
