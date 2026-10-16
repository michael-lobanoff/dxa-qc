"""Сколиоз как наблюдение: центральная линия позвоночного столба и угол её искривления.

Эксперт отмечал сколиоз комментарием и никогда — нарушением «наклон оси»: 14 и 10 снимков, без
пересечений. Наклон — ошибка укладки (столб прямой, но повёрнут), сколиоз — анатомия (столб изогнут).
Здесь сравниваем две меры изгиба на точках детектора вне обучающих фолдов:
  * отклонение центра столба от прямой «верх — уровень таза», мм (признак curvature сервиса);
  * угол искривления по центральной линии, как у Фергюсона по центрам позвонков: размах наклона
    касательной к линии, в градусах.

    python scripts/scoliosis_probe.py
"""
import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dxaqc.service import read_dicom_image  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path("data/train")


def centerline(img, t, b, window=60):
    """x центра столба на каждой строке между верхом и уровнем таза (сглаженный), и сами строки."""
    g = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 3)
    h, w = g.shape
    ys = np.arange(int(max(t[1], 0)) + 5, int(min(b[1], h - 1)) - 5)
    xs, yy = [], []
    for y in ys:
        xl = t[0] + (b[0] - t[0]) * (y - t[1]) / (b[1] - t[1])
        lo, hi = int(max(xl - window, 0)), int(min(xl + window, w - 1))
        row = g[y, lo:hi + 1]
        if row.size < 10:
            continue
        thr = np.percentile(row, 20) + 0.5 * (row.max() - np.percentile(row, 20))
        on = np.where(row > thr)[0]
        xs.append(lo + (on.min() + on.max()) / 2)
        yy.append(y)
    if len(xs) < 20:
        return None, None
    return np.asarray(yy, float), median_filter(np.asarray(xs, float), size=15, mode="nearest")


def measures(img, t, b, mm=0.607, seg=24):
    yy, xs = centerline(img, t, b)
    if yy is None:
        return None
    xl = t[0] + (b[0] - t[0]) * (yy - t[1]) / (b[1] - t[1])
    dev_mm = float(np.abs(xs - xl).max() * mm)
    # наклон касательной на отрезках по ~seg пикселей (полпозвонка); угол искривления — размах наклонов
    angs = []
    for k in range(0, len(yy) - seg, seg // 2):
        dx, dy = xs[k + seg] - xs[k], yy[k + seg] - yy[k]
        angs.append(np.degrees(np.arctan2(dx, dy)))
    angle = float(np.ptp(angs)) if len(angs) >= 2 else 0.0
    return dev_mm, angle


def main():
    idx = pd.read_csv(ROOT / "image_index.csv")
    idx["id"] = idx.n.map("{:03d}".format) + "_" + idx.region
    sp = idx[idx.region == "spine"].copy()
    sp["scol"] = sp.comment.fillna("").str.lower().str.contains("сколио").astype(int)
    oof = json.loads((ROOT / "kp_oof_spine_ens3.json").read_text())
    rows = []
    for _, r in sp.iterrows():
        p = oof.get(r.id, {}).get("points", {})
        if not p.get("col_top") or not p.get("col_bottom"):
            continue
        img, _ = read_dicom_image(ROOT / r.path)
        m = measures(img, p["col_top"], p["col_bottom"])
        if m:
            rows.append({"id": r.id, "study": r.study, "scol": r.scol, "axis": int(r.v_axis == 1), "dev_mm": m[0], "angle": m[1]})
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "scoliosis_measures.csv", index=False)
    print(f"позвоночников: {len(d)}, сколиоз: {d.scol.sum()}, наклон оси: {d.axis.sum()}")
    for c, name in (("dev_mm", "отклонение от прямой, мм"), ("angle", "угол искривления, °")):
        print(f"  {name:<28} AUC сколиоз {roc_auc_score(d.scol, d[c]):.3f}   AUC наклон {roc_auc_score(d.axis, d[c]):.3f}"
              f"   медиана: сколиоз {d[d.scol == 1][c].median():.1f}, остальные {d[d.scol == 0][c].median():.1f}")
    oof_threshold(d)


def oof_threshold(d, reps=20):
    """Порог угла по F1 внутри обучающих фолдов (по исследованиям), отметки считаем на отложенных."""
    y, a, g = d.scol.to_numpy(), d.angle.to_numpy(), d.study.to_numpy()
    tp = fp = 0
    thr = []
    for rep in range(reps):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(a, y, g):
            t = max(np.unique(a[tr]), key=lambda t: f1_score(y[tr], a[tr] >= t))
            thr.append(t)
            tp += ((a[te] >= t) & (y[te] == 1)).sum()
            fp += ((a[te] >= t) & (y[te] == 0)).sum()
    print(f"  порог по F1 внутри фолдов: {np.median(thr):.1f}° (5–95 %: {np.percentile(thr, 5):.1f}–{np.percentile(thr, 95):.1f}); "
          f"на отложенных отмечено сколиозов {tp / reps:.1f} из {y.sum()}, остальных {fp / reps:.1f} из {(y == 0).sum()}")


if __name__ == "__main__":
    main()
