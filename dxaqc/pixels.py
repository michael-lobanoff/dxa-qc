"""Pixel size of a DXA export.

These exports carry no PixelSpacing, but GE Lunar Prodigy writes Exposed Area (0040,0303) — the
size of the exposed field in mm. Where it describes the image itself it divides out to
0.60-0.64 mm/px (spine: 180 mm over 300 px; hip: 170-180 mm over 280 px); in part of the studies the
tag instead holds the whole scan-table area (520 mm) repeated for every image of the study, which is
rejected here by a plausibility check. Two independent anatomical rulers agree with the tag:
the vertebral pitch (50 px -> 30 mm) and the femoral head diameter (66 px -> 40 mm), both normal for
an elderly densitometry population.
"""
MIN_MM_PER_PX = 0.55
MAX_MM_PER_PX = 0.70
DEFAULT_MM_PER_PX = 0.607   # median of the images whose tag passes the check


def pixel_spacing_mm(ds, shape):
    """(mm per pixel, source) for a DICOM dataset and its (rows, cols) image shape."""
    rows, cols = shape[0], shape[1]
    for tag, (x, y) in (("PixelSpacing", ("y", "x")), ("ImagerPixelSpacing", ("y", "x"))):
        v = getattr(ds, tag, None)                      # honour a real spacing tag when present
        if v is not None and len(v) == 2 and float(v[0]) > 0:
            return float((float(v[0]) + float(v[1])) / 2), tag
    ea = getattr(ds, "ExposedArea", None)
    if ea is not None and len(ea) == 2 and float(ea[0]) > 0 and float(ea[1]) > 0 and rows and cols:
        sx, sy = float(ea[0]) / cols, float(ea[1]) / rows
        if (MIN_MM_PER_PX <= sx <= MAX_MM_PER_PX and MIN_MM_PER_PX <= sy <= MAX_MM_PER_PX
                and abs(sx - sy) / sy < 0.05):
            return (sx + sy) / 2, "ExposedArea"
    return DEFAULT_MM_PER_PX, "default"
