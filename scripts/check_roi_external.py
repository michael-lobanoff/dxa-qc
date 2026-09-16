"""Do our proposed L1-L4 boxes agree with the ones the densitometer printed itself?

The public Mendeley DXA set carries the scanner's own ROI boxes burned into the image, so the
horizontal separators between L1, L2, L3 and L4 are visible. This compares them with the boundaries
the service proposes (ТЗ 2.6, автоматическая коррекция разметки) — an external check of the vertebra
finder that needs no manual annotation.

Usage: python scripts/check_roi_external.py data/external/mendeley_dexa_spine [--vis out.png]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dxaqc.artifacts import OVERLAY_MAX_ANGLE
from dxaqc.service import QCService


def printed_separators(img, min_frac=0.30):
    """y of long horizontal printed lines (the ROI box edges), merged when close together."""
    g = img.astype(np.float32)
    tophat = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    mask = ((tophat > 0.18 * 255) & (g > np.percentile(g, 85))).astype(np.uint8) * 255
    min_len = int(min_frac * img.shape[1])
    segs = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=int(min_len * 0.6),
                           minLineLength=min_len, maxLineGap=4)
    ys = []
    if segs is not None:
        for x1, y1, x2, y2 in np.asarray(segs).reshape(-1, 4):
            a = abs(np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1)))) % 180
            if min(a, abs(a - 180)) < OVERLAY_MAX_ANGLE:          # horizontal only
                ys.append((y1 + y2) / 2)
    ys.sort()
    merged = []
    for y in ys:
        if merged and y - merged[-1][-1] < 6:
            merged[-1].append(y)
        else:
            merged.append([y])
    return [float(np.mean(m)) for m in merged]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--vis", type=Path)
    ap.add_argument("--models", default="models")
    args = ap.parse_args()

    svc = QCService(args.models)
    files = [p for p in sorted(args.folder.iterdir()) if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    per_image, all_err, with_roi, examples = [], [], 0, []
    for p in files:
        img = np.asarray(Image.open(p).convert("L"))
        if img.shape == (224, 224):      # the downscaled copies in that set: printed lines are no longer crisp
            continue
        res = svc.analyse(img)
        rois = (res.get("measurements") or {}).get("rois") or []
        seps = printed_separators(img)
        if not rois or len(seps) < 4:
            continue
        with_roi += 1
        ours = sorted({round(b, 1) for r in rois for b in (r["box"][1], r["box"][3])})
        pitch = res["measurements"].get("vert_pitch_mm") or 1
        err = [min(abs(s - o) for o in ours) for s in seps]
        per_image.append(np.median(err))
        all_err += err
        if len(examples) < 6:
            examples.append((p, img, ours, seps))

    if not per_image:
        print("нет снимков, где найдены и наши области, и напечатанные линии")
        return
    e = np.array(all_err)
    print(f"{with_roi} снимков с напечатанной разметкой и нашим предложением области")
    print(f"расхождение границ: медиана {np.median(e):.1f} пикс, p90 {np.quantile(e, 0.9):.1f}, "
          f"доля в пределах 5 пикс {np.mean(e < 5):.0%}")
    print(f"медиана по снимку: {np.median(per_image):.1f} пикс")

    if args.vis and examples:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, len(examples), figsize=(3.2 * len(examples), 8))
        for ax, (p, img, ours, seps) in zip(np.atleast_1d(axs), examples):
            ax.imshow(img, cmap="gray")
            for y in seps:
                ax.axhline(y, color="#ffd24a", lw=1.0)
            for y in ours:
                ax.axhline(y, color="#4cc9f0", lw=1.0, ls="--")
            ax.set_title(p.name[:16], fontsize=8)
            ax.axis("off")
        fig.suptitle("жёлтое — разметка аппарата, голубое пунктиром — предложение сервиса", fontsize=11)
        plt.tight_layout()
        plt.savefig(args.vis, dpi=90)
        print("->", args.vis)


if __name__ == "__main__":
    main()
