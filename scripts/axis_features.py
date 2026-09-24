"""Better estimate of the spine axis tilt (ТЗ 2.3: «правильно выровненная ось, допустимый наклон до 5°»).

The shipped feature is the angle of the line through two landmarks — the top of the column and its level
at the iliac crests. Two points mean the whole feature rests on two heatmap peaks, and the type it feeds
has the weakest F1 of all five (0.42 at AUC 0.89).

Candidates measured here, all on the same out-of-fold detector points and the same repeated
study-grouped CV as the pipeline:
  tilt2      the shipped two-point angle;
  tiltv      a robust (Theil-Sen) line through the centres of the vertebral bodies that vertebrae.find
             already locates — many points instead of two;
  tiltw      the same line weighted by how bright each vertebra is (dim rows are less reliable);
  combinations of them with the curvature feature that separates scoliosis from a tilted lay.

    python scripts/axis_features.py --tag ens3
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from dxaqc.decision import Monotone, best_threshold, spine_measurements
from dxaqc.geometry import axes_mm

SEEDS = range(300, 310)


def theil_sen_angle(pts, sx, sy):
    """Angle to the vertical of the robust line through the points, degrees (positive = leaning right)."""
    p = np.asarray([(x * sx, y * sy) for x, y in pts], float)
    if len(p) < 3:
        return None
    slopes = []
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            dy = p[j, 1] - p[i, 1]
            if abs(dy) > 1e-6:
                slopes.append((p[j, 0] - p[i, 0]) / dy)       # dx per dy: 0 = vertical
    if not slopes:
        return None
    return float(np.degrees(np.arctan(np.median(slopes))))


def column_centres(img, points, window=60):
    """(x, y) of the bright column in every row between the two axis landmarks, measured on the image.

    Unlike the vertebra centres of vertebrae.find, these do not lie on the col_top -> col_bottom line by
    construction, so a line fitted through them is a genuinely independent estimate of the axis.
    """
    import cv2
    t, b = points.get("col_top"), points.get("col_bottom")
    if not t or not b or b[1] - t[1] < 30:
        return []
    g = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 3)
    h, w = g.shape
    out = []
    for y in range(int(max(t[1], 0)) + 5, int(min(b[1], h - 1)) - 5):
        xl = t[0] + (b[0] - t[0]) * (y - t[1]) / (b[1] - t[1])
        lo, hi = int(max(xl - window, 0)), int(min(xl + window, w - 1))
        row = g[y, lo:hi + 1]
        if row.size < 10:
            continue
        thr = np.percentile(row, 20) + 0.5 * (row.max() - np.percentile(row, 20))
        on = np.where(row > thr)[0]
        out.append((lo + (on.min() + on.max()) / 2, y))
    return out


def rows(tag):
    idx = pd.read_csv("data/train/image_index.csv")
    idx["id"] = idx.n.map("{:03d}".format) + "_" + idx.region
    idx = idx.set_index("id")
    oof = json.loads(Path(f"data/train/kp_oof_spine_{tag}.json").read_text())
    seg = json.loads(Path("data/train/artifact_seg_oof.json").read_text())
    out = []
    for i in sorted(oof):
        if pd.isna(idx.loc[i, "y"]):
            continue
        img = np.asarray(Image.open(f"data/annotation/images/{i}.png"))
        mm = (float(idx.loc[i, "mm_per_px"]), float(idx.loc[i, "mm_per_px_y"]))
        m = spine_measurements(img, oof[i]["points"], oof[i]["conf"], mm, seg.get(i, 0.0))
        sx, sy = axes_mm(mm)
        vp = m.get("vert_points") or []
        ang = theil_sen_angle(vp, sx, sy)
        # brightness-weighted variant: repeat a vertebra centre proportionally to its mean brightness,
        # so faint rows (where the body is hard to see) count less
        wpts = []
        for x, y in vp:
            band = img[max(int(y) - 2, 0):int(y) + 3, max(int(x) - 25, 0):int(x) + 26]
            w = int(np.clip(band.mean() / 40, 1, 4)) if band.size else 1
            wpts += [(x, y)] * w
        # угол оси столба относительно таза: перпендикуляр к линии между гребнями вместо вертикали кадра
        ca, cb = oof[i]["points"].get("crest_a"), oof[i]["points"].get("crest_b")
        tilt_pelvis = np.nan
        if ca and cb and abs(cb[0] - ca[0]) > 1:
            pel = np.degrees(np.arctan2((cb[1] - ca[1]) * sy, (cb[0] - ca[0]) * sx))
            t, b = oof[i]["points"].get("col_top"), oof[i]["points"].get("col_bottom")
            if t and b and abs(b[1] - t[1]) > 1:
                col = np.degrees(np.arctan2((b[0] - t[0]) * sx, (b[1] - t[1]) * sy))
                tilt_pelvis = abs(col + pel)      # ось относительно перпендикуляра к линии гребней
        cc = column_centres(img, oof[i]["points"])
        angc = theil_sen_angle(cc, sx, sy) if len(cc) >= 20 else None
        # scoliosis bends the column; a tilted but straight lay does not. The shipped feature measures
        # the deviation from the line between two landmarks, so it inherits their error; this one fits a
        # parabola to the measured centres and takes its curvature — endpoints play no part.
        quad = np.nan
        if len(cc) >= 20:
            xs = np.array([c[0] for c in cc]) * sx
            ys = np.array([c[1] for c in cc]) * sy
            a = np.polyfit(ys, xs, 2)[0]
            quad = abs(a) * (ys.max() - ys.min()) ** 2 / 4       # sagitta in mm: comparable to curvature
        out.append({"id": i, "y": int(idx.loc[i, "y"]), "v_axis": int(idx.loc[i, "v_axis"]),
                    "tiltc": abs(angc) if angc is not None else np.nan, "quad": quad,
                    "tilt_pelvis": tilt_pelvis,
                    "tilt2": m["abs_tilt"], "curvature": m["curvature"], "n_vert": len(vp),
                    "tiltv": abs(ang) if ang is not None else np.nan,
                    "tiltw": abs(theil_sen_angle(wpts, sx, sy) or np.nan)})
    return pd.DataFrame(out)


def evaluate(df, feats, signs):
    X = df[feats].to_numpy(float)
    X = np.nan_to_num(X, nan=np.nanmedian(X, axis=0))
    y = df.v_axis.to_numpy(int)
    g = df.id.str[:3].astype(int).to_numpy()
    aucs, f1s = [], []
    for seed in SEEDS:
        prob, dec = np.zeros(len(y)), np.zeros(len(y), bool)
        for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, g):
            m = Monotone(signs).fit(X[tr], y[tr])
            p_tr = m.predict_proba(X[tr])[:, 1]
            prob[va] = m.predict_proba(X[va])[:, 1]
            dec[va] = prob[va] >= best_threshold(y[tr], p_tr)
        aucs.append(roc_auc_score(y, prob))
        f1s.append(f1_score(y, dec, zero_division=0))
    return float(np.mean(aucs)), float(np.mean(f1s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ens3")
    args = ap.parse_args()
    df = rows(args.tag)
    df.to_csv("data/train/axis_features.csv", index=False)
    print(f"снимков {len(df)}, нарушений {int(df.v_axis.sum())}, позвонков найдено в среднем {df.n_vert.mean():.1f}")
    print(f"корреляция углов: двухточечный vs по позвонкам {df[['tilt2', 'tiltv']].corr().iloc[0, 1]:.3f}, "
          f"vs по центрам столба {df[['tilt2', 'tiltc']].corr().iloc[0, 1]:.3f}")
    for feats, signs in (
        (["tilt2", "curvature"], [1, -1]),        # как в сервисе
        (["tiltv", "curvature"], [1, -1]),
        (["tiltw", "curvature"], [1, -1]),
        (["tilt2", "tiltv", "curvature"], [1, 1, -1]),
        (["tilt_pelvis"], [-1]),
        (["tilt2", "tilt_pelvis"], [1, -1]),
        (["tilt2", "tilt_pelvis", "curvature"], [1, -1, -1]),
        (["tilt2", "quad"], [1, -1]),
        (["tilt2", "curvature", "quad"], [1, -1, -1]),
        (["tiltc", "curvature"], [1, -1]),
        (["tilt2", "tiltc", "curvature"], [1, 1, -1]),
        (["tilt2"], [1]),
        (["tiltc"], [1]),
    ):
        auc, f1 = evaluate(df, feats, signs)
        print(f"  {'+'.join(feats):32s} AUC {auc:.3f}  F1 {f1:.3f}")


if __name__ == "__main__":
    main()
