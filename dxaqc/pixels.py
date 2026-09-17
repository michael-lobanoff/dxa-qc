"""Pixel size of a DXA export, per axis.

These exports carry no PixelSpacing, but GE Lunar Prodigy writes Exposed Area (0040,0303) — the size
of the exposed field in mm. Where it describes the image itself it divides out to 0.60-0.64 mm/px on
BOTH axes (spine: 180 mm over 300 px and 175 mm over 289 px; hip: 170 mm over 280 px); in part of the
studies the tag instead holds the whole scan-table area (520 mm) repeated for every image of the
study, which is rejected here by a plausibility check.

The organisers state (разъяснения V2, вопрос 1) that the scanner's pixel is 1.05 mm along Y and
0.6 mm along X. That matches our X, but not our Y, and three independent measurements on the exported
images support a square pixel:

  * the tag itself gives 0.60 along X and 0.606 along Y on every image where it describes the image;
  * vertebral body width / vertebral pitch measures 64 px / 50 px = 1.28, and a lumbar vertebra really
    is about 1.3x wider than one body+disc is tall — at 0.6 x 1.05 it would come out taller than wide;
  * absolute sizes at 0.6 mm are anatomically normal (vertebral pitch 30 mm, femoral head 40 mm),
    while at 1.05 mm along Y they are not (52 mm and 69 mm).

So 1.05 mm is most likely the scanner's native scan-line spacing before the export is resampled for
display. Rather than argue, the spacing is a parameter: DXAQC_PIXEL_MM="0.6,1.05" (x,y) overrides it
for the whole service. Note that the verdict barely depends on this — every threshold is learned from
the experts' own labels, so it adapts to whatever scale those experts were looking at — and that
angles (spine tilt) are measured on the displayed image, where the ≤5° criterion of ТЗ was judged.
"""
import os

DXA_MIN_MM_PER_PX = 0.30   # below this the image is a plain radiograph, not densitometry
MIN_MM_PER_PX = 0.55
MAX_MM_PER_PX = 0.70
DEFAULT_MM_PER_PX = 0.607   # median over the images whose tag passes the check


def _override():
    v = os.environ.get("DXAQC_PIXEL_MM")
    if not v:
        return None
    try:
        parts = [float(x) for x in v.replace(";", ",").split(",")]
    except ValueError:
        return None
    if len(parts) == 1 and parts[0] > 0:
        return parts[0], parts[0]
    if len(parts) == 2 and min(parts) > 0:
        return parts[0], parts[1]
    return None


def pixel_spacing_mm(ds, shape):
    """((mm per pixel along x, along y), source) for a DICOM dataset and its (rows, cols) shape."""
    forced = _override()
    if forced:
        return forced, "DXAQC_PIXEL_MM"
    rows, cols = shape[0], shape[1]
    for tag in ("PixelSpacing", "ImagerPixelSpacing"):
        v = getattr(ds, tag, None)                      # DICOM order is (row spacing, column spacing)
        if v is not None and len(v) == 2 and float(v[0]) > 0:
            return (float(v[1]), float(v[0])), tag
    ea = getattr(ds, "ExposedArea", None)
    if ea is not None and len(ea) == 2 and float(ea[0]) > 0 and float(ea[1]) > 0 and rows and cols:
        sx, sy = float(ea[0]) / cols, float(ea[1]) / rows
        if (MIN_MM_PER_PX <= sx <= MAX_MM_PER_PX and MIN_MM_PER_PX <= sy <= MAX_MM_PER_PX
                and abs(sx - sy) / sy < 0.05):
            return (sx, sy), "ExposedArea"
    return (DEFAULT_MM_PER_PX, DEFAULT_MM_PER_PX), "default"
