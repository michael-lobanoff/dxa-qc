"""Run the service over an external image folder (PNG/JPG) to see how it behaves on foreign DXA scans.

Used with the public Mendeley set of spine DEXA images from Pakistan (CC BY 4.0): those come from
another centre and another scanner, so they test two things the organisers' data cannot — whether the
"is this our kind of image" guard refuses legitimate DXA, and whether the region classifier still
recognises a lumbar spine.

Usage: python scripts/check_external.py data/external/mendeley_dexa_spine [--vis out.png]
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from dxaqc.service import QCService


def load(path):
    """PNG/JPEG through PIL, DICOM through the same reader the service uses."""
    if path.suffix.lower() == ".dcm":
        from dxaqc.service import read_dicom_image
        return read_dicom_image(path)[0]
    return np.asarray(Image.open(path).convert("L"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--vis", type=Path, help="save a contact sheet of examples")
    ap.add_argument("--models", default="models")
    args = ap.parse_args()

    svc = QCService(args.models)
    # rglob: external sets come as nested folders; "._name" are macOS resource forks, not images
    files = sorted(p for p in args.folder.rglob("*")
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".dcm") and not p.name.startswith("._"))
    if not files:
        raise SystemExit(f"в {args.folder} нет изображений (.png/.jpg/.dcm) — проверьте путь")
    rows, shapes = [], Counter()
    for p in files:
        img = load(p)
        shapes[img.shape] += 1
        ok, why, stats = svc.region.check(img)
        region = svc.region.predict(img)[0] if ok else None
        rows.append({"file": p.name, "ok": ok, "why": why, "region": region, **stats})

    n = len(rows)
    acc = sum(r["ok"] for r in rows)
    print(f"{n} изображений, размеры: {dict(list(shapes.most_common(4)))}")
    print(f"принято сервисом: {acc} ({acc / max(n, 1):.0%}), отклонено: {n - acc}")
    print("причины отказа:", dict(Counter(r["why"] for r in rows if not r["ok"])))
    print("определённая область:", dict(Counter(r["region"] for r in rows if r["ok"])))
    nov = np.array([r["novelty"] for r in rows])
    print(f"новизна: медиана {np.median(nov):.2f}, p10 {np.quantile(nov, 0.1):.2f}, max {nov.max():.2f}"
          f"  (порог отказа 1.0, на обучающих снимках max 0.80)")

    if args.vis:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pick = rows[:12]
        fig, axs = plt.subplots(2, 6, figsize=(18, 8))
        for ax, r in zip(axs.ravel(), pick):
            ax.imshow(load(args.folder / r["file"]), cmap="gray")
            ax.set_title(f"{'принят' if r['ok'] else 'отказ'}\n{r['region'] or r['why']}\nновизна {r['novelty']:.2f}",
                         fontsize=8)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(args.vis, dpi=80)
        print("->", args.vis)


if __name__ == "__main__":
    main()
