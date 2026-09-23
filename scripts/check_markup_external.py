"""Read the ROI markup a densitometer printed on the image, and judge it (optional feature, ТЗ 2.6).

The organisers' export carries no printed markup (0 of 249 scans), so this runs on the public Mendeley
DXA set from Multan (CC BY 4.0), where every full-resolution image has it.

Two numbers per image, both without any reference annotation:
  - how far each printed separator sits from the nearest intervertebral disc (the darkest point of the
    column profile) — this judges the scanner's own markup;
  - the same distance for the boxes our service proposes, so the two can be compared directly.

Usage: python scripts/check_markup_external.py data/external/mendeley_dexa_spine
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from dxaqc import markup as M
from dxaqc.service import QCService


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--models", default="models")
    args = ap.parse_args()
    svc = QCService(args.models)

    files = [p for p in sorted(args.folder.iterdir()) if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    read, levels, theirs, ours, cut, fixes = 0, [], [], [], 0, []
    for p in files:
        img = np.asarray(Image.open(p).convert("L"))
        if img.shape == (224, 224):          # downscaled copies: the printed lines are no longer crisp
            continue
        m = M.read(img)
        if m is None:
            continue
        read += 1
        levels.append(len(m["levels"]))
        err = M.separator_errors(img, m)
        theirs += err
        rev = M.review(img, m)
        cut += int(not rev["in_frame"])
        if rev["corrections"]:
            fixes.append((p.name, M.describe(rev)))
        res = svc.analyse(img)
        rois = (res.get("measurements") or {}).get("rois") or []
        if len(rois) >= 2:                   # our own separators: the boundaries between our boxes
            sep = sorted(r["box"][3] for r in sorted(rois, key=lambda r: r["box"][1])[:-1])
            ours += M.separator_errors(img, {"box": m["box"], "separators": sep})

    total = sum(1 for p in files if Image.open(p).size != (224, 224))
    print(f"снимков в полном разрешении: {total}")
    print(f"разметка аппарата прочитана: {read} ({read / max(total, 1):.0%}), уровней "
          f"{ {k: levels.count(k) for k in sorted(set(levels))} }, обрезана краем кадра: {cut}")
    for name, v in (("разделители аппарата", theirs), ("наши границы", ours)):
        if v:
            v = np.asarray(v)
            print(f"{name}: отклонение от межпозвонкового промежутка — медиана {np.median(v):.1f} мм, "
                  f"p90 {np.quantile(v, 0.9):.1f} мм, n = {len(v)}")
    print(f"предложена коррекция (разделитель дальше {M.CORRECT_MM:.0f} мм от промежутка): "
          f"{len(fixes)} снимков из {read}")
    for name, line in fixes:
        print(f"  {name}: {line}")


if __name__ == "__main__":
    main()
