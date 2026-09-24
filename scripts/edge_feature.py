"""Is the hip field of view cut too tight — measured from pixels, not from landmark positions?

The margins of ТЗ 2.3 (3 cm above the greater trochanter, 3 cm below the lesser, 2 cm to the side) are
computed from the detector's landmarks. The same trap as with the iliac crests applies: when a structure
is cut off, the net can still place it confidently just inside the frame, and the margin comes out
plausible. A pixel measure has no such failure mode — when the field is too tight, bone touches the edge
of the frame.

    edge_<side> = share of that border strip whose brightness is at bone level

Measured here against the expert's v_roi labels with the pipeline's own CV, alone and next to the
shipped margins.

    python scripts/edge_feature.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from dxaqc.decision import Monotone, best_threshold
from dxaqc.features import hip_features

SEEDS = range(300, 310)


def edge_bone(img, side, strip=3, q=0.80):
    """Share of each border strip that is as bright as bone (above the q-quantile of the image)."""
    g = img.astype(np.float32)
    thr = np.quantile(g[g > 0], q) if (g > 0).any() else 255
    top = (g[:strip] > thr).mean()
    bottom = (g[-strip:] > thr).mean()
    lat = (g[:, :strip] > thr).mean() if side == "hip_right" else (g[:, -strip:] > thr).mean()
    return {"edge_top": float(top), "edge_bottom": float(bottom), "edge_lat": float(lat)}


def rows(tag="ens3"):
    idx = pd.read_csv("data/train/image_index.csv")
    idx["id"] = idx.n.map("{:03d}".format) + "_" + idx.region
    idx = idx.set_index("id")
    kp = json.loads(Path("data/train/keypoints.json").read_text())
    oof = json.loads(Path(f"data/train/kp_oof_hip_{tag}.json").read_text())
    out = []
    for i in sorted(oof):
        if pd.isna(idx.loc[i, "y"]) or kp[i]["flags"].get("skip"):
            continue
        img = np.asarray(Image.open(f"data/annotation/images/{i}.png"))
        mm = (float(idx.loc[i, "mm_per_px"]), float(idx.loc[i, "mm_per_px_y"]))
        f = hip_features(oof[i]["points"], img.shape, i[4:], mm)
        out.append({"id": i, "v_roi": int(idx.loc[i, "v_roi"]),
                    "margin_bottom": f["margin_bottom"], "margin_top": f["margin_top"],
                    **edge_bone(img, i[4:])})
    return pd.DataFrame(out)


def cv(df, feats, signs):
    X = np.nan_to_num(df[feats].to_numpy(float), nan=0.0)
    y = df.v_roi.to_numpy(int)
    g = df.id.str[:3].astype(int).to_numpy()
    aucs, f1s = [], []
    for seed in SEEDS:
        prob, dec = np.zeros(len(y)), np.zeros(len(y), bool)
        for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, g):
            m = Monotone(signs).fit(X[tr], y[tr])
            prob[va] = m.predict_proba(X[va])[:, 1]
            dec[va] = prob[va] >= best_threshold(y[tr], m.predict_proba(X[tr])[:, 1])
        aucs.append(roc_auc_score(y, prob))
        f1s.append(f1_score(y, dec, zero_division=0))
    return float(np.mean(aucs)), float(np.mean(f1s))


def main():
    df = rows()
    df.to_csv("data/train/edge_feature.csv", index=False)
    print(f"бёдер {len(df)}, нарушений области интереса {int(df.v_roi.sum())}")
    print(df.groupby("v_roi")[["margin_top", "margin_bottom", "edge_top", "edge_bottom", "edge_lat"]].median().round(3).to_string())
    for feats, signs in ((["margin_bottom", "margin_top"], [-1, -1]),
                         (["edge_top", "edge_bottom"], [1, 1]),
                         (["margin_bottom", "margin_top", "edge_top"], [-1, -1, 1]),
                         (["margin_bottom", "margin_top", "edge_top", "edge_bottom"], [-1, -1, 1, 1])):
        auc, f1 = cv(df, feats, signs)
        print(f"  {'+'.join(feats):48s} AUC {auc:.3f}  F1 {f1:.3f}")


if __name__ == "__main__":
    main()
