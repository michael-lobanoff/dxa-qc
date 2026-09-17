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


def crop_matrix(points, shape):
    fh, sp, sd = _get(points, "fh_c"), _get(points, "shaft_p"), _get(points, "shaft_d")
    if fh is None or sp is None:
        h, w = shape  # failed scan: plain resize of the whole frame
        s = CROP / max(h, w)
        return np.array([[s, 0, (CROP - w * s) / 2], [0, s, (CROP - h * s) / 2]], np.float32)
    up = (sp - sd) if sd is not None else np.array([0, -1], np.float32)
    ang = math.degrees(math.atan2(up[0], -up[1]))            # tilt of the shaft from vertical
    scale = HEAD_SHAFT_PX / max(np.linalg.norm(fh - sp), 1.0)
    others = [v for v in (_get(points, "fn_c"), _get(points, "gt_top"), fh, sp) if v is not None]
    centre = np.mean(others, axis=0)
    m = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])), ang, scale)
    m[:, 2] += np.array([CROP / 2, CROP / 2 + 6]) - centre   # centre of the femur slightly above the middle
    return m.astype(np.float32)


ZOOM_BOX = (30, 110, 44, 124)      # neck and lesser trochanter inside the normalised crop
ZOOM_SIZE = 64
CELL_WIDE = 20                     # coarse grid over the whole femur
CELL_NEAR = 6                      # fine grid over the close-up


def rotation_features(crop):
    """Descriptor for the rotation classifier: a coarse grid over the whole femur plus a fine grid over
    a close-up of the region the criterion is about (neck and lesser trochanter).

    The two grid sizes were swept over 20 combinations and the winner confirmed on seeds it was not
    selected on: 0.852 against 0.828 for the earlier 16/8 pair. Shape at the scale of the whole bone
    plus texture where the trochanter is — the coarse/fine split matters more than either alone."""
    y0, y1, x0, x1 = ZOOM_BOX
    near = cv2.resize(crop[y0:y1, x0:x1], (ZOOM_SIZE, ZOOM_SIZE), interpolation=cv2.INTER_LINEAR)
    return np.concatenate([hog(crop, CELL_WIDE), hog(near, CELL_NEAR)])


def hip_crop(img, points, side, jitter=None):
    """uint8 crop (CROP x CROP). jitter: optional (dx, dy, dang, dscale) augmentation."""
    im, p = normalise(img, points, side)
    m = crop_matrix(p, im.shape)
    if jitter is not None:
        dx, dy, dang, dscale = jitter
        a = cv2.getRotationMatrix2D((CROP / 2, CROP / 2), dang, dscale)
        a[:, 2] += (dx, dy)
        m = (np.vstack([a, [0, 0, 1]]) @ np.vstack([m, [0, 0, 1]]))[:2].astype(np.float32)
    out = cv2.warpAffine(np.ascontiguousarray(im), m, (CROP, CROP), flags=cv2.INTER_LINEAR, borderValue=0)
    return out, m
