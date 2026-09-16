"""Fit the final decision layer (models/decision.pkl) on all labelled scans.

Measurements come from out-of-fold detector keypoints and out-of-fold rotation scores, i.e. from the
same distribution the service produces on unseen scans, so calibration and thresholds transfer.
The rotation forest itself is refitted on all manual-keypoint crops.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

from dxaqc.decision import TYPES, DecisionModel, hip_measurements, spine_measurements
from dxaqc.hipcrop import hip_crop
from dxaqc.hog import hog


def rotation_forest():
    return RandomForestClassifier(500, min_samples_leaf=3, max_features=0.1, class_weight="balanced_subsample", random_state=0, n_jobs=4)


def main():
    idx = pd.read_csv("data/train/image_index.csv")
    idx["id"] = idx.n.map("{:03d}".format) + "_" + idx.region
    idx = idx.set_index("id")
    kp = json.loads(Path("data/train/keypoints.json").read_text())
    oof = {**json.loads(Path("data/train/kp_oof_spine.json").read_text()), **json.loads(Path("data/train/kp_oof_hip.json").read_text())}
    img = {i: np.asarray(Image.open(f"data/annotation/images/{i}.png")) for i in oof}

    # rotation: out-of-fold scores (averaged over 5 CV repeats) for calibration, final forest on everything
    hp = [i for i in sorted(oof) if "hip" in i and not kp[i]["flags"].get("skip") and not pd.isna(idx.loc[i, "y"])]
    y_rot = idx.loc[hp, "v_posrot"].astype(int).to_numpy(); groups = np.array([int(i[:3]) for i in hp])
    hog_man = np.stack([hog(hip_crop(img[i], kp[i]["points"], i[4:])[0], 16) for i in hp])
    hog_det = np.stack([hog(hip_crop(img[i], oof[i]["points"], i[4:])[0], 16) for i in hp])
    rot_oof = np.zeros(len(hp))
    for seed in range(5):
        for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(hog_man, y_rot, groups):
            rot_oof[va] += rotation_forest().fit(hog_man[tr], y_rot[tr]).predict_proba(hog_det[va])[:, 1] / 5
    rotation = rotation_forest().fit(hog_man, y_rot)
    rot_of = dict(zip(hp, rot_oof))

    out = {"rotation_rf": rotation, "hog_cell": 16}
    for region in ("spine", "hip"):
        ids = [i for i in sorted(oof) if (i.endswith("spine") if region == "spine" else "hip" in i)
               and not kp[i]["flags"].get("skip") and not pd.isna(idx.loc[i, "y"])]
        meas = {}
        for i in ids:
            p, c = oof[i]["points"], oof[i]["conf"]
            if region == "spine":
                meas[i] = spine_measurements(img[i], p, c, idx.loc[i, "mm_per_px"])
            else:
                meas[i] = hip_measurements(img[i], p, c, i[4:], idx.loc[i, "mm_per_px"])
                meas[i]["rotation"] = rot_of[i]
        feats = {vt: np.array([[0.0 if meas[i].get(c) is None else float(meas[i][c]) for c in cols] for i in ids])
                 for vt, (cols, _) in TYPES[region].items()}
        labels = {vt: idx.loc[ids, vt].astype(int).to_numpy() for vt in TYPES[region]}
        y_img = idx.loc[ids, "y"].astype(int).to_numpy()
        out[region] = DecisionModel().fit(region, feats, labels, y_img)
        print(f"{region}: {len(ids)} images; per-type thresholds:",
              {vt: round(t, 3) for vt, t in out[region].thresholds.items()},
              "| image thresholds:", {k: round(v, 3) for k, v in out[region].image_thresholds.items()})
    Path("models").mkdir(exist_ok=True)
    with open("models/decision.pkl", "wb") as f:
        pickle.dump(out, f)
    print("saved models/decision.pkl")


if __name__ == "__main__":
    main()
