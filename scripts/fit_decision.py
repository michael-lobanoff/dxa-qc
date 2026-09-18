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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold

from dxaqc.decision import ROTATION_BLEND, SYNTH_TYPES, TYPES, DecisionModel, hip_measurements, spine_measurements
from dxaqc import hipcrop
from dxaqc.hipcrop import hip_crop, rotation_features


def spacing(idx, i):
    """(mm per pixel along x, along y) for one image, as stored by build_index."""
    return float(idx.loc[i, "mm_per_px"]), float(idx.loc[i, "mm_per_px_y"])


def rotation_forest():
    """Extremely randomised trees beat a random forest here (honest AUC 0.825 against 0.807): with 150
    hips and a 1152-dimensional descriptor the extra randomisation is the cheapest regulariser."""
    return ExtraTreesClassifier(800, min_samples_leaf=3, max_features=0.1,
                                class_weight="balanced_subsample", random_state=0, n_jobs=4)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="detector: kp_oof_<region>_<tag>.json (the one shipped in models/)")
    ap.add_argument("--members", nargs="*", default=[],
                    help="detector runs of the shipped ensemble: rotation trains on all their crops and averages over them")
    ap.add_argument("--seg", default="", help="metal segmentation variant: data/train/artifact_seg_oof_<seg>.json")
    ap.add_argument("--synth", default=None, help="synthetic scans for the rare types: data/train/synth_meas_<synth>.csv")
    args = ap.parse_args()
    sfx = f"_{args.tag}" if args.tag else ""
    idx = pd.read_csv("data/train/image_index.csv")
    idx["id"] = idx.n.map("{:03d}".format) + "_" + idx.region
    idx = idx.set_index("id")
    kp = json.loads(Path("data/train/keypoints.json").read_text())
    oof = {**json.loads(Path(f"data/train/kp_oof_spine{sfx}.json").read_text()),
           **json.loads(Path(f"data/train/kp_oof_hip{sfx}.json").read_text())}
    syn_all = pd.read_csv(f"data/train/synth_meas_{args.synth}.csv") if args.synth else None
    seg_oof = json.loads(Path(f"data/train/artifact_seg_oof{'_' + args.seg if args.seg else ''}.json").read_text())
    img = {i: np.asarray(Image.open(f"data/annotation/images/{i}.png")) for i in oof}

    # rotation: out-of-fold scores (averaged over 5 CV repeats) for calibration, final forest on everything
    hp = [i for i in sorted(oof) if "hip" in i and not kp[i]["flags"].get("skip") and not pd.isna(idx.loc[i, "y"])]
    y_rot = idx.loc[hp, "v_posrot"].astype(int).to_numpy(); groups = np.array([int(i[:3]) for i in hp])
    hog_man = np.stack([rotation_features(hip_crop(img[i], kp[i]["points"], i[4:])[0]) for i in hp])
    hog_det = np.stack([rotation_features(hip_crop(img[i], oof[i]["points"], i[4:])[0]) for i in hp])
    hog_mem = [np.stack([rotation_features(hip_crop(img[i], m[i]["points"], i[4:])[0]) for i in hp])
               for m in (json.loads(Path(f"data/train/kp_oof_hip_{t}.json").read_text()) for t in args.members)] or [hog_det]
    rot_oof = np.zeros(len(hp))
    for seed in range(5):
        for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(hog_man, y_rot, groups):
            # train on both versions of every crop: hand-placed and detector-placed. The model then
            # sees the landmark error it will meet in production (honest AUC 0.789 -> 0.807).
            Xtr = np.vstack([hog_man[tr]] + [h[tr] for h in hog_mem]); ytr = np.concatenate([y_rot[tr]] * (1 + len(hog_mem)))
            rf = rotation_forest().fit(Xtr, ytr)
            rot_oof[va] += np.mean([rf.predict_proba(h[va])[:, 1] for h in hog_mem], 0) / 5
    rotation = rotation_forest().fit(np.vstack([hog_man] + hog_mem), np.concatenate([y_rot] * (1 + len(hog_mem))))

    # weight of the other hip of the same study: a fixed constant (decision.ROTATION_BLEND), the value
    # the evaluation has always used. Picking it by AUC on these same scores gave 0.40 and +0.007 F1 —
    # noise-level, and it would be tuned on the data the estimate is reported on.
    side = [i[4:] for i in hp]
    pos = {(int(i[:3]), sd): k for k, (i, sd) in enumerate(zip(hp, side))}
    partner = [pos.get((int(i[:3]), "hip_right" if sd == "hip_left" else "hip_left")) for i, sd in zip(hp, side)]
    other = np.array([rot_oof[j] if j is not None else np.nan for j in partner])
    alpha = ROTATION_BLEND
    rot_oof = np.where(np.isfinite(other), (1 - alpha) * rot_oof + alpha * np.nan_to_num(other), rot_oof)
    print(f"вес второго бедра: {alpha:.2f}")
    rot_of = dict(zip(hp, rot_oof))

    out = {"rotation_rf": rotation, "hog_cell": 16, "rotation_blend": alpha, "rotation_members": bool(args.members),
           "image_axis": hipcrop.IMAGE_AXIS}
    for region in ("spine", "hip"):
        ids = [i for i in sorted(oof) if (i.endswith("spine") if region == "spine" else "hip" in i)
               and not kp[i]["flags"].get("skip") and not pd.isna(idx.loc[i, "y"])]
        meas = {}
        for i in ids:
            p, c = oof[i]["points"], oof[i]["conf"]
            if region == "spine":
                meas[i] = spine_measurements(img[i], p, c, spacing(idx, i), seg_oof.get(i, 0.0))
            else:
                meas[i] = hip_measurements(img[i], p, c, i[4:], spacing(idx, i))
                meas[i]["rotation"] = rot_of[i]
        feats = {vt: np.array([[0.0 if meas[i].get(c) is None else float(meas[i][c]) for c in cols] for i in ids])
                 for vt, (cols, _) in TYPES[region].items()}
        labels = {vt: idx.loc[ids, vt].astype(int).to_numpy() for vt in TYPES[region]}
        y_img = idx.loc[ids, "y"].astype(int).to_numpy()
        synth = {}
        if syn_all is not None:
            sy = syn_all[syn_all.src.isin(ids)]
            for vt, (cols, _) in TYPES[region].items():
                if vt in SYNTH_TYPES and set(cols) <= set(sy.columns):
                    synth[vt] = (np.nan_to_num(sy[cols].to_numpy(float), nan=0.0), sy.label.to_numpy(int))
            print(f"{region}: synthetic scans {len(sy)}")
        out[region] = DecisionModel().fit(region, feats, labels, y_img, synth)
        print(f"{region}: {len(ids)} images; per-type thresholds:",
              {vt: round(t, 3) for vt, t in out[region].thresholds.items()},
              "| image thresholds:", {k: round(v, 3) for k, v in out[region].image_thresholds.items()})
    # one operating point for both regions, fitted on every labelled image at once
    y_all = np.concatenate([out[r].y_img_ for r in ("spine", "hip")])
    s_all = np.concatenate([out[r].scores_ for r in ("spine", "hip")])
    for r in ("spine", "hip"):
        out[r].set_image_thresholds(y_all, s_all)
    print("общий порог по изображению:", {k: round(v, 3) for k, v in out["spine"].image_thresholds.items()})
    Path("models").mkdir(exist_ok=True)
    with open("models/decision.pkl", "wb") as f:
        pickle.dump(out, f)
    print("saved models/decision.pkl")


if __name__ == "__main__":
    main()
