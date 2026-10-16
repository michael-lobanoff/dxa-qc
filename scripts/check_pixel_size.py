"""Проверка размера пикселя по форме головки бедра.

Головка бедренной кости почти шар, на снимке это круг. Если пиксель квадратный, контур головки
в пикселях — тоже круг; если по вертикали пиксель крупнее, круг сплющивается. Подгоняем к краю
головки эллипс с осями вдоль X и Y и берём отношение полуосей: оно равно размеру пикселя по Y,
делённому на размер по X. Проверка не зависит ни от роста пациента, ни от калибровки аппарата.

Центр и верх головки берём из ручной разметки, край ищем лучами из центра по верхней полуокружности
(нижняя переходит в шейку): резкое падение яркости на суставной щели.

    python scripts/check_pixel_size.py

Результат на 150 бёдрах выгрузки организаторов: отношение 0.964, 95 % ДИ 0.941–0.996. Квадратный
пиксель из тега Exposed Area (0.600 × 0.606) даёт 1.01, вариант 0.6 × 1.05 из разъяснений V2 — 1.75.
Диаметр головки 46 мм по X и 49 мм по Y — норма; при 1.05 мм по вертикали вышло бы 84 мм.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from scipy.ndimage import gaussian_filter1d, map_coordinates

warnings.filterwarnings("ignore")
ROOT = Path("data/train")


def edge_points(img, c, r0):
    """Точки края головки на лучах из центра по верхней полуокружности; слабые края отбрасываем."""
    pts, strength = [], []
    for th in np.deg2rad(np.arange(-168, -11, 3)):           # ось Y направлена вниз: это верх
        r = np.arange(0.45 * r0, 1.55 * r0, 0.25)
        xs, ys = c[0] + r * np.cos(th), c[1] + r * np.sin(th)
        prof = gaussian_filter1d(map_coordinates(img, [ys, xs], order=1, mode="nearest"), 2.0)
        g = np.gradient(prof)
        lo, hi = np.searchsorted(r, 0.6 * r0), np.searchsorted(r, 1.4 * r0)
        j = lo + int(np.argmin(g[lo:hi]))
        pts.append((xs[j], ys[j]))
        strength.append(-g[j])
    pts, strength = np.array(pts), np.array(strength)
    return pts[strength >= np.percentile(strength, 30)]


def ellipse_axes(pts, c):
    """Полуоси эллипса с центром в c и осями вдоль X и Y: (x-cx)^2 u + (y-cy)^2 v = 1."""
    dx, dy = pts[:, 0] - c[0], pts[:, 1] - c[1]
    (u, v), *_ = np.linalg.lstsq(np.c_[dx ** 2, dy ** 2], np.ones_like(dx), rcond=None)
    if u <= 0 or v <= 0:
        return None
    return 1 / np.sqrt(u), 1 / np.sqrt(v)


def main():
    idx = pd.read_csv(ROOT / "image_index.csv")
    idx["id"] = idx.n.map("{:03d}".format) + "_" + idx.region
    idx = idx.set_index("id")
    kp = json.loads((ROOT / "keypoints.json").read_text())

    rows = []
    for i, d in kp.items():
        p = d["points"]
        if "hip" not in i or not p.get("fh_c") or not p.get("fh_top") or i not in idx.index:
            continue
        img = pydicom.dcmread(str(ROOT / idx.loc[i, "path"])).pixel_array.astype(np.float32)
        if img.ndim == 3:
            img = img[..., 0]
        c = np.array(p["fh_c"], float)
        r0 = float(np.hypot(*(np.array(p["fh_top"]) - c)))
        if r0 < 8:
            continue
        pts = edge_points(img, c, r0)
        axes = ellipse_axes(pts, c) if len(pts) >= 15 else None
        if axes:
            rows.append({"id": i, "a": axes[0], "b": axes[1], "ratio": axes[0] / axes[1]})

    d = pd.DataFrame(rows)
    v = d.ratio[(d.ratio > 0.3) & (d.ratio < 3)]
    boot = [v.sample(len(v), replace=True, random_state=s).median() for s in range(2000)]
    print(f"бёдер: {len(v)}")
    print(f"размер пикселя Y / X по форме головки: {v.median():.3f} "
          f"[95 % ДИ {np.percentile(boot, 2.5):.3f}–{np.percentile(boot, 97.5):.3f}], "
          f"межквартильный размах {v.quantile(.25):.2f}–{v.quantile(.75):.2f}")
    print("ожидание: 0.600 × 0.606 → 1.010;  0.6 × 1.05 → 1.750;  0.6 × 1.2 → 2.000")
    print(f"ближе к квадратному пикселю, чем к 0.6 × 1.05: {100 * (abs(v - 1.01) < abs(v - 1.75)).mean():.0f} % бёдер")
    a, b = d.a.median(), d.b.median()
    print(f"диаметр головки: по X {2 * a * 0.6:.1f} мм, по Y {2 * b * 0.606:.1f} мм (при 1.05 мм было бы {2 * b * 1.05:.1f})")


if __name__ == "__main__":
    main()
