"""Калибровка отказа при размытии (dxaqc/quality.py).

На всех снимках выгрузки считаем резкость как есть и после размытия шести видов: гауссово σ = 1, 1.5,
3 пикселя и смаз движения на 9 и 15 пикселей по вертикали и на 15 по горизонтали. Порог — 70 % от самого
нерезкого реального снимка своей области. Печатаем, сколько реальных снимков отказ задел бы и сколько
размытых ловит.

    python scripts/blur_check.py
"""
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dxaqc.quality import SHARPNESS_MIN, sharpness  # noqa: E402
from dxaqc.service import read_dicom_image  # noqa: E402

warnings.filterwarnings("ignore")


def kernel(n, horizontal=False):
    k = np.zeros((n, n), np.float32)
    if horizontal:
        k[n // 2, :] = 1 / n
    else:
        k[:, n // 2] = 1 / n
    return k


BLURS = {
    "гаусс σ=1": lambda x: cv2.GaussianBlur(x, (0, 0), 1.0),
    "гаусс σ=1.5": lambda x: cv2.GaussianBlur(x, (0, 0), 1.5),
    "гаусс σ=3": lambda x: cv2.GaussianBlur(x, (0, 0), 3.0),
    "смаз 9 по вертикали": lambda x: cv2.filter2D(x, -1, kernel(9)),
    "смаз 15 по вертикали": lambda x: cv2.filter2D(x, -1, kernel(15)),
    "смаз 15 по горизонтали": lambda x: cv2.filter2D(x, -1, kernel(15, True)),
}


def main():
    idx = pd.read_csv("data/train/image_index.csv")
    rows = []
    for _, r in idx.iterrows():
        img = read_dicom_image(Path("data/train") / r.path)[0].astype(np.float32)
        reg = "spine" if r.region == "spine" else "hip"
        rows.append({"reg": reg, "kind": "реальный", "s": sharpness(img)})
        for name, f in BLURS.items():
            rows.append({"reg": reg, "kind": name, "s": sharpness(f(img))})
    d = pd.DataFrame(rows)
    for reg in ("spine", "hip"):
        sub, thr = d[d.reg == reg], SHARPNESS_MIN[reg]
        real = sub[sub.kind == "реальный"].s
        print(f"{reg}: порог {thr}, самый нерезкий реальный {real.min():.3f}, "
              f"отказов на реальных {int((real < thr).sum())} из {len(real)}")
        for name in BLURS:
            v = sub[sub.kind == name].s
            print(f"    {name:<24} ловится {100 * (v < thr).mean():5.1f} %")


if __name__ == "__main__":
    main()
