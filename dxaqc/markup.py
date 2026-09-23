"""Reading the ROI markup a densitometer prints onto the exported image, and judging it.

Some scanners burn their own analysis boxes into the picture: an outer rectangle around the lumbar
column with horizontal separators between L1, L2, L3 and L4 (the operator can move them, and in
practice sometimes should). ТЗ 2.6 lists correcting that markup as optional extra functionality, and
the organisers confirmed it in the chat on 23.09 — "разметка" there means exactly these printed boxes,
not the label file.

The scans in the organisers' export carry no such markup (0 of 249 images), so everything here is
developed and measured on a public DXA set from another centre (Мултан, Пакистан, CC BY 4.0), where
every image has it. On the organisers' data the module is simply inert.

How the markup is judged without knowing the true vertebral levels: a separator is supposed to sit in
an intervertebral disc, and a disc is darker than the bodies around it. So the profile of the column
along its axis has a local minimum at every correct separator, and the distance from a separator to
the nearest minimum is an objective error measure — no reference annotation needed.
"""
import cv2
import numpy as np

from .artifacts import OVERLAY_MAX_ANGLE, OVERLAY_SIDE_PX, _isolated
from .geometry import axes_mm

MIN_FRAC = 0.30       # a printed box side spans a good part of the frame
MERGE_PX = 6          # lines closer than this are the same drawn line
MIN_LEVEL_PX = 12     # a level thinner than this is a detection artefact, not a vertebra
PROMINENCE = 0.15     # how deep a dip of the column profile has to be to count as a disc
CORRECT_MM = 4.0      # a separator further than this from a disc is worth moving (see docs/night_report.md)


def _segments(img):
    """Long, straight, bright, isolated lines: (horizontal ys, vertical xs) in pixels."""
    g = img.astype(np.float32)
    tophat = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    mask = ((tophat > 0.18 * 255) & (g > np.percentile(g, 85))).astype(np.uint8) * 255
    hor, ver = [], []
    for axis, min_len in ((0, int(MIN_FRAC * img.shape[1])), (1, int(MIN_FRAC * img.shape[0]))):
        segs = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=int(min_len * 0.6),
                               minLineLength=min_len, maxLineGap=4)
        if segs is None:
            continue
        for x1, y1, x2, y2 in np.asarray(segs).reshape(-1, 4):
            a = abs(np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1)))) % 180
            if not _isolated(g, x1, y1, x2, y2):
                continue
            if axis == 0 and min(a, abs(a - 180)) < OVERLAY_MAX_ANGLE:
                hor.append((y1 + y2) / 2)
            elif axis == 1 and abs(a - 90) < OVERLAY_MAX_ANGLE:
                ver.append((x1 + x2) / 2)
    return _merge(hor), _merge(ver)


def _merge(vals):
    out = []
    for v in sorted(vals):
        if out and v - out[-1][-1] < MERGE_PX:
            out[-1].append(v)
        else:
            out.append([v])
    return [float(np.mean(m)) for m in out]


def _row_extent(img, y, thr_pct=80):
    """Longest run of bright ridge pixels in row y: where the drawn separator starts and ends."""
    g = img.astype(np.float32)
    row = g[y]
    up, down = g[max(y - 3, 0)], g[min(y + 3, len(g) - 1)]
    on = (row - np.maximum(up, down) > 0.10 * 255) & (row > np.percentile(g, thr_pct))
    best, run = (0, 0), None
    for x, v in enumerate(np.append(on, False)):
        if v and run is None:
            run = x
        elif not v and run is not None:
            if x - run > best[1] - best[0]:
                best = (run, x)
            run = None
    return best


def _ridge_rows(img, x0, x1, min_frac=0.55):
    """Rows that are a thin bright line across the box: how the separators are actually found.

    Hough needs a crisp line and misses the faint ones on some exports; a printed separator is simply a
    row where most columns inside the box are a local bright ridge (brighter than the rows above and
    below by a clear margin).
    """
    g = img.astype(np.float32)
    band = g[:, int(max(x0, 0)):int(min(x1, img.shape[1]))]
    if band.shape[1] < 10:
        return []
    up, down = np.roll(band, 3, axis=0), np.roll(band, -3, axis=0)
    ridge = (band - np.maximum(up, down) > 0.10 * 255) & (band > np.percentile(g, 80))
    frac = ridge.mean(1)
    rows = [y for y in range(3, len(frac) - 3) if frac[y] > min_frac and frac[y] >= frac[max(y - 1, 0)]
            and frac[y] >= frac[min(y + 1, len(frac) - 1)]]
    return _merge(rows)


def read(img):
    """The printed markup, or None when the image carries none.

    {box: [x0, y0, x1, y1], separators: [y...], levels: [[x0, y0, x1, y1], ...]} — levels top to bottom,
    as many as the separators define (four for a standard L1-L4 box).
    """
    h, w = img.shape
    hor, ver = _segments(img)
    ver = [x for x in ver if 2 < x < w - 3]          # the frame edge is not a drawn line
    if len(ver) >= 2:
        x0, x1 = ver[0], ver[-1]
    else:
        # only one side is crisp (the other often runs along the frame edge): take the sides from the
        # separators themselves — each is a drawn line spanning exactly the width of the box
        cand = [y for y in _ridge_rows(img, 0, w, min_frac=0.35) if OVERLAY_SIDE_PX < y < h - OVERLAY_SIDE_PX]
        spans = [_row_extent(img, int(y)) for y in cand]
        spans = [s for s in spans if s[1] - s[0] > 0.3 * w]
        if len(spans) < 3:
            return None
        # the vertebra contour interrupts some separators, so take the widest quartile, not the median
        x0 = float(np.quantile([s[0] for s in spans], 0.25))
        x1 = float(np.quantile([s[1] for s in spans], 0.75))
    if x1 - x0 < 0.3 * w:
        return None
    rows = [y for y in _ridge_rows(img, x0, x1) if OVERLAY_SIDE_PX < y < h - OVERLAY_SIDE_PX]
    rows = [y for k, y in enumerate(rows) if k == 0 or y - rows[k - 1] >= MIN_LEVEL_PX]
    if len(rows) < 3:
        return None
    levels = [[x0, rows[k], x1, rows[k + 1]] for k in range(len(rows) - 1)]
    return {"box": [x0, rows[0], x1, rows[-1]], "separators": rows[1:-1], "levels": levels}


def column_profile(img, box, smooth=5):
    """Mean brightness of the column inside the markup box, row by row (bodies bright, discs dark)."""
    x0, y0, x1, y1 = box
    xs = slice(int(max(x0, 0)), int(min(x1, img.shape[1])))
    band = img[:, xs].astype(np.float32)
    prof = band.mean(1)
    k = np.ones(smooth) / smooth
    return np.convolve(np.pad(prof, smooth // 2, mode="edge"), k, "same")[smooth // 2:smooth // 2 + len(prof)]


def _discs(img, markup):
    """Rows inside the box where the column is prominently dark: the intervertebral discs."""
    prof = column_profile(img, markup["box"])
    y0, y1 = int(markup["box"][1]), int(markup["box"][3])
    inner = prof[y0:y1]
    if len(inner) < 10:
        return np.empty(0)
    # only prominent minima count: the profile is noisy, and a metric that accepted any dip would call
    # every separator correct. A disc drops at least PROMINENCE of the profile range below the bodies
    # on both sides of it.
    rng = float(inner.max() - inner.min()) or 1.0
    out = []
    for k in range(1, len(inner) - 1):
        if not (inner[k] <= inner[k - 1] and inner[k] < inner[k + 1]):
            continue
        left, right = inner[:k].max(initial=inner[k]), inner[k + 1:].max(initial=inner[k])
        if min(left, right) - inner[k] >= PROMINENCE * rng:
            out.append(y0 + k)
    return np.asarray(out, float)


def separator_errors(img, markup, mm_per_px=None):
    """Distance in mm from every separator to the nearest disc.

    Returns [] when the profile has no usable minima. The same number can be computed for boxes the
    service proposes itself, which makes the two directly comparable on one objective criterion.
    """
    if not markup:
        return []
    sy = axes_mm(mm_per_px)[1]
    d = _discs(img, markup)
    if not len(d):
        return []
    return [float(np.min(np.abs(d - s)) * sy) for s in markup["separators"]]


def corrections(img, markup, mm_per_px=None):
    """Separators that sit off a disc, with the position we propose instead.

    This is the actual optional function of ТЗ 2.6: not re-drawing the whole box, but telling the
    operator which separator to move and where. The proposal is the nearest prominent dark row, i.e.
    the disc itself — no reference annotation is involved.
    """
    if not markup:
        return []
    sy = axes_mm(mm_per_px)[1]
    d = _discs(img, markup)
    if not len(d):
        return []
    out = []
    for i, s in enumerate(markup["separators"], start=1):
        to = float(d[int(np.argmin(np.abs(d - s)))])
        if abs(to - s) * sy >= CORRECT_MM:
            out.append({"index": i, "from": float(s), "to": to, "shift_mm": round((to - s) * sy, 1)})
    return out


def review(img, markup, mm_per_px=None):
    """Assessment of the printed markup (optional feature, ТЗ 2.6)."""
    if not markup:
        return None
    sy = axes_mm(mm_per_px)[1]
    h, w = img.shape
    err = separator_errors(img, markup, mm_per_px)
    box = markup["box"]
    return {"levels": len(markup["levels"]),
            "in_frame": bool(box[0] > 1 and box[1] > 1 and box[2] < w - 2 and box[3] < h - 2),
            "separator_error_mm": None if not err else round(float(np.median(err)), 1),
            "max_error_mm": None if not err else round(float(np.max(err)), 1),
            "height_mm": round((box[3] - box[1]) * sy, 1),
            "corrections": corrections(img, markup, mm_per_px)}


def describe(review_dict):
    """One Russian line for the report; None when there is no markup on the image."""
    if not review_dict:
        return None
    bits = [f"разметка аппарата: {review_dict['levels']} уровня"]
    if not review_dict["in_frame"]:
        bits.append("обрезана краем кадра")
    fix = review_dict.get("corrections") or []
    if fix:
        bits.append("предлагаем сдвинуть " + ", ".join(
            f"разделитель №{c['index']} на {abs(c['shift_mm']):.0f} мм {'вниз' if c['shift_mm'] > 0 else 'вверх'}"
            for c in fix))
    elif review_dict.get("max_error_mm") is not None:
        bits.append(f"разделители лежат в межпозвонковых промежутках (максимум {review_dict['max_error_mm']:.0f} мм)")
    return "; ".join(bits)
