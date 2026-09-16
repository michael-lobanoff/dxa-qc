"""Hip positioning / rotation (expert label v_posrot): compare feature sets with study-grouped CV.

Crops come from manual keypoints for training images and, with --oof, from the detector's
out-of-fold predictions for validation images (what the service will see on new scans).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dxaqc.hipcrop import CROP, hip_crop
from dxaqc.hog import hog
from dxaqc.features import hip_features

def contour_profile(crop, n=24):
    """Medial (right-hand) edge of the femur below the neck, detrended: the lesser trochanter bump."""
    g = cv2.GaussianBlur(crop.astype(np.float32), (0, 0), 1.5)
    rows = np.linspace(CROP * 0.45, CROP * 0.85, n).astype(int)
    xs = []
    for y in rows:
        row = g[y]; c = CROP // 2 - 10                       # shaft sits left of the crop centre
        thr = row[max(c - 15, 0):c + 15].mean() * 0.55
        x = c
        while x < CROP - 1 and row[x] > thr:
            x += 1
        xs.append(x)
    xs = np.array(xs, np.float32)
    t = np.arange(n)
    return xs - np.polyval(np.polyfit(t, xs, 1), t)


def features(crop):
    small = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255
    return {"hog": hog(crop), "pixels": small, "contour": contour_profile(crop)}


def load(use_oof):
    kp = json.loads(Path("data/train/keypoints.json").read_text())
    oof = json.loads(Path("data/train/kp_oof_hip.json").read_text()) if use_oof else {}
    idx = pd.read_csv("data/train/image_index.csv")
    idx["id"] = idx.n.map("{:03d}".format) + "_" + idx.region
    idx = idx.set_index("id")
    rows = []
    for i in sorted(kp):
        if "hip" not in i or kp[i]["flags"].get("skip") or pd.isna(idx.loc[i, "v_posrot"]):
            continue
        img = np.asarray(Image.open(f"data/annotation/images/{i}.png"))
        side = i[4:]
        man = kp[i]["points"]; pts = oof[i]["points"] if use_oof else man
        crop, _ = hip_crop(img, pts, side)
        f = features(crop)
        g = hip_features(pts, img.shape, side)
        f["geom"] = np.array([g.get(k, np.nan) for k in ("margin_top", "margin_bottom", "margin_lateral", "lt_absent",
                                                           "missing", "neck_offset", "neck_shaft_angle", "head_gt_width")], np.float32)
        rows.append({"id": i, "n": int(i[:3]), "y": int(idx.loc[i, "v_posrot"]), **f})
    return rows


def cv_auc(X, y, groups, make_model, repeats=5):
    aucs, oof_all = [], np.zeros((repeats, len(y)))
    for r in range(repeats):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=r)
        oof = np.zeros(len(y))
        for tr, va in cv.split(X, y, groups):
            m = make_model(); m.fit(X[tr], y[tr]); oof[va] = m.predict_proba(X[va])[:, 1]
        aucs.append(roc_auc_score(y, oof)); oof_all[r] = oof
    return np.mean(aucs), np.std(aucs), oof_all.mean(0)


def bootstrap_ci(y, s, n=2000, seed=0):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        b = rng.integers(0, len(y), len(y))
        if len(set(y[b])) == 2:
            vals.append(roc_auc_score(y[b], s[b]))
    return np.percentile(vals, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", action="store_true", help="crops from detector out-of-fold keypoints")
    args = ap.parse_args()
    rows = load(args.oof)
    y = np.array([r["y"] for r in rows]); groups = np.array([r["n"] for r in rows])
    print(f"{len(rows)} hips, {y.sum()} violations; crops from {'predicted' if args.oof else 'manual'} keypoints")

    def lr(c, pca=None):
        steps = [SimpleImputer(strategy="median"), StandardScaler()] + ([PCA(pca, random_state=0)] if pca else []) + \
                [LogisticRegression(C=c, class_weight="balanced", max_iter=2000)]
        return lambda: make_pipeline(*steps)

    def stack(*keys):
        return np.hstack([np.stack([r[k] for r in rows]).astype(np.float32) for k in keys])

    n_hog = len(rows[0]["hog"])
    combo = lambda: make_pipeline(
        ColumnTransformer([("hog", make_pipeline(StandardScaler(), PCA(20, random_state=0)), slice(0, n_hog)),
                           ("rest", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), slice(n_hog, None))]),
        LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000))
    experiments = {
        "geom (keypoint geometry)": (stack("geom"), lr(0.3)),
        "contour profile": (stack("contour"), lr(0.3)),
        "pixels 32x32 + PCA": (stack("pixels"), lr(0.1, 20)),
        "HOG + PCA": (stack("hog"), lr(0.1, 20)),
        "HOG + contour + geom": (stack("hog", "contour", "geom"), combo),
    }
    results = {}
    for name, (X, mk) in experiments.items():
        m, s, oof = cv_auc(X, y, groups, mk)
        lo, hi = bootstrap_ci(y, oof)
        results[name] = {"auc": round(m, 3), "sd": round(s, 3), "ci95": [round(lo, 3), round(hi, 3)]}
        print(f"  {name:28s} AUC {m:.3f} ± {s:.3f}  (95% CI {lo:.2f}–{hi:.2f})")
    out = Path(f"data/train/rotation_classic_{'oof' if args.oof else 'manual'}.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
