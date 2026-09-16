"""Synthetic quality violations generated from normal scans with known keypoints.

Geometric transforms move the keypoints together with the pixels; labels are then
recomputed by dxaqc.geometry, so they follow the production rules. Artifacts are
drawn procedurally (bra underwires, hooks, wires) until a bank of real cut-outs exists.
Synthetic samples are for training only: metrics are reported on real scans.
"""
import cv2
import numpy as np

from . import geometry as G


def _map_kp(kp, fn, shape):
    out = {}
    for k, v in kp.items():
        p = None if v is None else fn(v)
        out[k] = [float(p[0]), float(p[1])] if G.inside(p, shape) else None
    return out


def rotate(img, kp, deg, center=None):
    """Rotate the patient inside a fixed, axis-aligned scan field (positive = counter-clockwise)."""
    h, w = img.shape
    if center is None:
        pts = [v for k, v in kp.items() if v is not None and k in ("col_top", "col_bottom")]
        center = tuple(np.mean(pts, axis=0)) if pts else (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(tuple(map(float, center)), deg, 1.0)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return out, _map_kp(kp, lambda p: M @ np.array([p[0], p[1], 1.0]), out.shape)


def crop(img, kp, top=0, bottom=0):
    """Scan started lower / stopped higher. Height varies across real exports, so cropping keeps them realistic."""
    h = img.shape[0]
    out = img[top:h - bottom]
    return out, _map_kp(kp, lambda p: (p[0], p[1] - top), out.shape)


def zoom_crop(img, kp, lateral_px, side):
    """Lateral field too narrow: cut the lateral strip, trim top/bottom to keep the aspect ratio, resize back.

    The scan width is fixed by the device, so the result keeps the original size; the
    anatomy gets slightly larger, which stays within normal patient-size variation.
    Shifting with a mirrored border instead would duplicate pelvic bones.
    """
    h, w = img.shape
    k = int(round(lateral_px))
    kv = int(round(k * h / w))
    x0 = k if side == "hip_right" else 0   # right hip: lateral side faces the image left
    y0 = kv // 2
    box = img[y0:h - (kv - kv // 2), x0:x0 + w - k]
    sx, sy = w / box.shape[1], h / box.shape[0]
    out = cv2.resize(box, (w, h), interpolation=cv2.INTER_LINEAR)
    return out, _map_kp(kp, lambda p: ((p[0] - x0) * sx, (p[1] - y0) * sy), out.shape)


def _stamp(img, mask, rng, return_mask=False):
    """Blend a metal mask: metal is near the top of the 8-bit range regardless of the tissue under it."""
    level = rng.uniform(0.85, 1.0) * max(int(img.max()), 200)
    mask = cv2.GaussianBlur(mask, (0, 0), rng.uniform(0.4, 0.9))
    out = np.maximum(img.astype(np.float32), mask * level).clip(0, 255).astype(img.dtype)
    return (out, mask > 0.3) if return_mask else out


def draw_underwire(img, rng, spine_x=None, return_mask=False):
    """Thin bright arcs near the top edge, often one per side (typical bra underwire); sometimes a double
    contour (metal wire inside a casing)."""
    h, w = img.shape
    ss = 4  # supersampling for thin anti-aliased lines
    mask = np.zeros((h * ss, w * ss), np.float32)
    cx = spine_x if spine_x is not None else w / 2
    sides = [-1, 1] if rng.random() < 0.6 else [rng.choice([-1, 1])]
    for s in sides:
        for _ in range(20):  # resample arcs that fall outside the frame: an invisible artifact is label noise
            arc = np.zeros_like(mask)
            center = (cx + s * rng.uniform(0.15, 0.35) * w, rng.uniform(-0.15, 0.12) * h)
            axes = (rng.uniform(0.18, 0.4) * w, rng.uniform(0.06, 0.2) * h)
            start = rng.uniform(0, 60)
            cv2.ellipse(arc, (int(center[0] * ss), int(center[1] * ss)), (int(axes[0] * ss), int(axes[1] * ss)),
                        rng.uniform(-15, 15), start, start + rng.uniform(90, 170), 1.0, int(rng.uniform(1.2, 2.5) * ss),
                        cv2.LINE_AA)
            if arc.sum() / ss ** 2 > 40:
                if rng.random() < 0.5:  # double contour: second arc 2-3 px inside
                    gap = rng.uniform(2, 3.5) * ss
                    cv2.ellipse(arc, (int(center[0] * ss), int(center[1] * ss)), (int(axes[0] * ss - gap), int(axes[1] * ss - gap)),
                                0, start, start + 150, 1.0, int(1.2 * ss), cv2.LINE_AA)
                mask = np.maximum(mask, arc)
                break
    return _stamp(img, cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA), rng, return_mask)


def draw_hardware(img, rng, return_mask=False):
    """Small metal pieces: hooks, clips, staples, wires, or a thick vertical clip. Kept visible at 8 bits."""
    h, w = img.shape
    mask = np.zeros((h, w), np.float32)
    if rng.random() < 0.25:  # thick clip, can sit right next to / over the spine
        x, y, ln, wd = rng.uniform(0.3, 0.7) * w, rng.uniform(0.02, 0.6) * h, rng.uniform(20, 45), rng.uniform(4, 8)
        ang = rng.uniform(-20, 20)
        box = cv2.boxPoints(((float(x), float(y)), (float(wd), float(ln)), float(ang))).astype(np.int32)
        cv2.fillConvexPoly(mask, box, 1.0, cv2.LINE_AA)
        return _stamp(img, mask, rng, return_mask)
    cx, cy = rng.uniform(0.15, 0.85) * w, rng.uniform(0.05, 0.8) * h
    for _ in range(rng.integers(2, 7)):
        x, y = cx + rng.normal(0, 10), cy + rng.normal(0, 10)
        if rng.random() < 0.5:  # straight piece
            ang, ln = rng.uniform(0, np.pi), rng.uniform(8, 30)
            p1 = (int(x - ln / 2 * np.cos(ang)), int(y - ln / 2 * np.sin(ang)))
            p2 = (int(x + ln / 2 * np.cos(ang)), int(y + ln / 2 * np.sin(ang)))
            cv2.line(mask, p1, p2, 1.0, int(rng.integers(2, 4)), cv2.LINE_AA)
        else:  # hook / ring
            r = int(rng.uniform(3, 7))
            cv2.ellipse(mask, (int(x), int(y)), (r, r), 0, rng.uniform(0, 180), rng.uniform(220, 360), 1.0, 2, cv2.LINE_AA)
    return _stamp(img, mask, rng, return_mask)


def _axis_at(kp, y):
    """Point of the spine centre line at row y (annotation rule: col_top/col_bottom sit on the frame edge)."""
    (x1, y1), (x2, y2) = kp["col_top"], kp["col_bottom"]
    return [x1 + (x2 - x1) * (y - y1) / (y2 - y1), float(y)]


def spine_sample(img, kp, kind, rng):
    """Return (image, keypoints, labels) for one synthetic spine variant.

    kind: 'tilt' | 'tilt_ok' | 'cut_top' | 'cut_top_ok' | 'cut_bottom' | 'artifact'.
    cut_top shortens the crest-to-top span below MIN_SPINE_SPAN_MM (Th12 leaves the frame).
    '*_ok' variants sit just inside the limits: hard negatives near the decision boundary.
    """
    artifact = False
    if kind in ("tilt", "tilt_ok"):
        cur = G.spine_tilt_deg(kp) or 0.0
        target = rng.uniform(5.8, 14.0) if kind == "tilt" else rng.uniform(0.0, 4.2)
        target *= rng.choice([-1, 1])
        img, kp = rotate(img, kp, cur - target)  # counter-clockwise rotation lowers the tilt
    elif kind in ("cut_top", "cut_top_ok"):
        span = G.spine_span_mm(kp)
        if span is None:
            raise ValueError("iliac crests are not in the frame")
        target = G.MIN_SPINE_SPAN_MM * (rng.uniform(0.7, 0.97) if kind == "cut_top" else rng.uniform(1.02, 1.1))
        rows = int(round((span - target) / G.MM_PER_PX))
        if rows <= 0:
            raise ValueError("span is already below the target")
        if kp["col_top"][1] < rows:
            kp = {**kp, "col_top": _axis_at(kp, rows)}
        img, kp = crop(img, kp, top=rows)
    elif kind == "cut_bottom":
        crests = [kp[k][1] for k in ("crest_a", "crest_b") if kp.get(k)]
        if not crests:
            raise ValueError("iliac crests are already outside the frame")
        bottom = int(img.shape[0] - min(crests) + rng.uniform(2, 20))
        new_h = img.shape[0] - bottom
        if kp["col_bottom"][1] >= new_h:
            kp = {**kp, "col_bottom": _axis_at(kp, new_h - 1)}
        img, kp = crop(img, kp, bottom=bottom)
    elif kind == "artifact":
        xs = [kp[k][0] for k in ("col_top", "col_bottom") if kp.get(k) is not None]
        spine_x = float(np.mean(xs)) if xs else None
        img = draw_underwire(img, rng, spine_x) if rng.random() < 0.7 else draw_hardware(img, rng)
        artifact = True
    else:
        raise ValueError(kind)
    labels = G.spine_rules(kp, img.shape)
    labels["v_artifact"] = artifact
    return img, kp, labels


def hip_sample(img, kp, side, kind, rng, mm_per_px=G.MM_PER_PX):
    """Return (image, keypoints, labels) for a hip whose field margin is pushed below the limit.

    kind: 'margin_top' | 'margin_bottom' | 'margin_lateral'.
    """
    m = G.hip_margins_mm(kp, img.shape, side, mm_per_px)
    edge = kind.split("_")[1]
    limit = {"top": G.HIP_MARGIN_TOP_MM, "bottom": G.HIP_MARGIN_BOTTOM_MM, "lateral": G.HIP_MARGIN_LATERAL_MM}[edge]
    if m[edge] is None:
        raise ValueError(f"{edge} reference point is missing")
    target_px = rng.uniform(0.15, 0.85) * limit / mm_per_px
    cut_px = m[edge] / mm_per_px - target_px
    if cut_px <= 0:
        raise ValueError("margin is already below the limit")
    if edge == "top":
        img, kp = crop(img, kp, top=int(cut_px))
    elif edge == "bottom":
        img, kp = crop(img, kp, bottom=int(cut_px))
    else:
        # After resizing by w / (w - k) the lateral margin becomes (lat - k) * w / (w - k); solve for k.
        w, lat = img.shape[1], m[edge] / mm_per_px
        img, kp = zoom_crop(img, kp, w * (lat - target_px) / (w - target_px), side)
    return img, kp, G.hip_rules(kp, img.shape, side, mm_per_px)
