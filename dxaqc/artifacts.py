"""Hand-crafted cues for foreign bodies on spine scans (bra underwires, hooks, clips, staples).

Metal shows up as thin, very bright structures. A white top-hat keeps structures narrower than
the kernel (wires, hooks) and suppresses broad bone and soft tissue; connected components are
then described by size, elongation and position relative to the spine column.
"""
import cv2
import numpy as np


def thin_bright_components(img, kernel=11, th_rel=0.18):
    g = img.astype(np.float32)
    tophat = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel)))
    mask = (tophat > th_rel * 255) & (g > np.percentile(g, 85))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    comps = []
    for k in range(1, n):
        area = stats[k, cv2.CC_STAT_AREA]
        if area < 6:
            continue
        ys, xs = np.where(lab == k)
        cov = np.cov(np.stack([xs, ys])) if area > 2 else np.eye(2)
        ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
        comps.append({"area": float(area), "elong": float(np.sqrt(ev[0] / max(ev[1], 1e-3))),
                      "length": float(4 * np.sqrt(max(ev[0], 0))), "cx": float(cent[k, 0]), "cy": float(cent[k, 1]),
                      "tophat": float(tophat[ys, xs].mean())})
    return comps, tophat


def _in_column(c, points, w, column_half_width):
    if not points or not points.get("col_top") or not points.get("col_bottom"):
        return abs(c["cx"] - w / 2) < column_half_width
    (x1, y1), (x2, y2) = points["col_top"], points["col_bottom"]
    return abs(c["cx"] - (x1 + (x2 - x1) * (c["cy"] - y1) / max(y2 - y1, 1))) < column_half_width


def artifact_mask(img, points=None, column_half_width=40, kernel=11, th_rel=0.18):
    """Pixels of the thin bright components that drive the artifact score (outside the column, top 40 %)."""
    g = img.astype(np.float32)
    tophat = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel)))
    mask = (tophat > th_rel * 255) & (g > np.percentile(g, 85))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = np.zeros(img.shape, bool)
    for k in range(1, n):
        c = {"cx": cent[k, 0], "cy": cent[k, 1]}
        if stats[k, cv2.CC_STAT_AREA] >= 6 and c["cy"] < 0.4 * img.shape[0] and not _in_column(c, points, img.shape[1], column_half_width):
            out |= lab == k
    return out


def artifact_features(img, points=None, column_half_width=40):
    """Image-level features. points: spine keypoints (col_top / col_bottom) to separate the column."""
    h, w = img.shape
    comps, tophat = thin_bright_components(img)

    def in_column(c):
        if not points or not points.get("col_top") or not points.get("col_bottom"):
            return abs(c["cx"] - w / 2) < column_half_width
        (x1, y1), (x2, y2) = points["col_top"], points["col_bottom"]
        ax = x1 + (x2 - x1) * (c["cy"] - y1) / max(y2 - y1, 1)
        return abs(c["cx"] - ax) < column_half_width

    out_col = [c for c in comps if not in_column(c)]
    lines = [c for c in out_col if c["elong"] > 3 and c["length"] > 15]
    top = [c for c in out_col if c["cy"] < 0.4 * h]
    f = {
        "n_out": len(out_col), "area_out": sum(c["area"] for c in out_col),
        "n_lines": len(lines), "len_lines": sum(c["length"] for c in lines), "max_len": max([c["length"] for c in out_col], default=0),
        "area_top": sum(c["area"] for c in top), "max_elong": max([c["elong"] for c in out_col], default=0),
        "tophat_p999": float(np.percentile(tophat, 99.9)), "sat_frac": float((img >= img.max() - 3).mean()),
        "n_in_col_bright": sum(1 for c in comps if in_column(c) and c["tophat"] > 0.35 * 255),
    }
    return f
