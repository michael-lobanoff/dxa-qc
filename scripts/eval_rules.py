"""Compare QC measurements from manual keypoints and from out-of-fold detector predictions
against the expert labels. A measurement is useful if it separates the expert's violations
from normal scans (AUC); the detector is good enough if its AUCs stay close to the manual ones.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from dxaqc import geometry as G
from dxaqc.features import hip_features, spine_features


def table(points_by_id, idx):
    rows = []
    for image_id, p in points_by_id.items():
        r = idx.loc[image_id]
        shape = (r.rows, r.cols)
        f = (spine_features(p, shape, r.mm_per_px) if r.region == "spine"
             else hip_features(p, shape, r.region, r.mm_per_px))
        rows.append({"id": image_id, **f})
    return pd.DataFrame(rows).set_index("id")


def auc(y, x, sign):
    ok = ~pd.isna(x)
    if ok.sum() < 10 or len(set(y[ok])) < 2:
        return np.nan
    return roc_auc_score(y[ok], sign * x[ok].astype(float))


def main():
    idx = pd.read_csv("data/train/image_index.csv")
    idx["id"] = idx.n.map("{:03d}".format) + "_" + idx.region
    idx = idx.set_index("id")
    kp = json.loads(Path("data/train/keypoints.json").read_text())
    checks = {  # (label, feature, sign: +1 larger = worse)
        "spine": [("v_axis", "abs_tilt", 1), ("v_pos", "crests_missing", 1), ("v_pos", "span_mm", -1)],
        "hip": [("v_roi", "margin_bottom", -1), ("v_roi", "margin_top", -1), ("v_roi", "margin_lateral", -1),
                ("v_posrot", "lt_absent", 1), ("v_posrot", "lt_protrusion", -1), ("v_posrot", "neck_offset", -1),
                ("v_posrot", "neck_shaft_angle", 1), ("v_posrot", "head_gt_width", -1), ("v_posrot", "missing", 1)],
    }
    for region, rules in checks.items():
        oof_path = Path(f"data/train/kp_oof_{region}.json")
        if not oof_path.exists():
            print(f"{region}: no out-of-fold predictions yet"); continue
        oof = {k: v["points"] for k, v in json.loads(oof_path.read_text()).items()}
        ids = [i for i in oof if not pd.isna(idx.loc[i, "y"])]
        man = table({i: kp[i]["points"] for i in ids}, idx)
        prd = table({i: oof[i] for i in ids}, idx)
        print(f"\n{region.upper()} (n={len(ids)})   AUC vs expert: manual points | predicted points")
        for label, feat, sign in rules:
            y = idx.loc[ids, label].to_numpy()
            a_m = auc(y, man[feat].to_numpy() if feat in man else np.full(len(ids), np.nan), sign)
            a_p = auc(y, prd[feat].to_numpy() if feat in prd else np.full(len(ids), np.nan), sign)
            print(f"  {label:9s} {feat:18s} {a_m:6.3f} | {a_p:6.3f}   (violations: {int(np.nansum(y))})")


if __name__ == "__main__":
    main()
