"""Does the rotation model gain from what the landmarks already know?

ТЗ 2.3 judges hip rotation by the lesser trochanter: too big means under-rotated, absent means
over-rotated. The shipped model reads only a HOG descriptor of the aligned crop. Meanwhile the
detector already reports how sure it is that it sees a lesser trochanter (and how far it sticks out of
the shaft line), and none of that reaches the model.

Everything here mirrors the pipeline exactly: the same ExtraTrees, training on manual plus every
ensemble member's crop, the same study-grouped repeated CV, the same blending with the other hip of the
study. Only the feature stack changes.

    python scripts/rotation_features.py --members fix_s100 strong_s100 base_s100
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from dxaqc.decision import ROTATION_BLEND, ROTATION_MAX_FEATURES, best_threshold
from dxaqc.features import hip_features
from dxaqc.hipcrop import hip_crop, rotation_features

SEEDS = range(300, 310)
GEOM = ["lt_conf", "lt_absent", "lt_protrusion", "neck_offset", "neck_shaft_angle", "shaft_angle", "head_gt_width"]


def blend(prob, ids, alpha):
    pos = {(i[:3], i[4:]): k for k, i in enumerate(ids)}
    out = prob.copy()
    for k, i in enumerate(ids):
        j = pos.get((i[:3], "hip_right" if i[4:] == "hip_left" else "hip_left"))
        if j is not None:
            out[k] = (1 - alpha) * prob[k] + alpha * prob[j]
    return out


def load(members, tag="ens3"):
    idx = pd.read_csv("data/train/image_index.csv")
    idx["id"] = idx.n.map("{:03d}".format) + "_" + idx.region
    idx = idx.set_index("id")
    kp = json.loads(Path("data/train/keypoints.json").read_text())
    oof = json.loads(Path(f"data/train/kp_oof_hip_{tag}.json").read_text())
    ids = [i for i in sorted(oof) if not kp[i]["flags"].get("skip") and not pd.isna(idx.loc[i, "y"])]
    img = {i: np.asarray(Image.open(f"data/annotation/images/{i}.png")) for i in ids}
    mem = [json.loads(Path(f"data/train/kp_oof_hip_{t}.json").read_text()) for t in members] or [oof]

    def geom(points_of, i):
        p = points_of(i)
        f = hip_features(p, img[i].shape, i[4:], (float(idx.loc[i, "mm_per_px"]), float(idx.loc[i, "mm_per_px_y"])))
        conf = points_of.__self__[i]["conf"] if hasattr(points_of, "__self__") else None
        return f, conf

    def stack_for(source):
        """(HOG, geometry) for crops placed by one source of keypoints."""
        hogs, geoms = [], []
        for i in ids:
            p = source[i]["points"] if isinstance(source, dict) else kp[i]["points"]
            conf = source[i]["conf"] if isinstance(source, dict) else {k: float(v is not None) for k, v in kp[i]["points"].items()}
            hogs.append(rotation_features(hip_crop(img[i], p, i[4:])[0]))
            f = hip_features(p, img[i].shape, i[4:], (float(idx.loc[i, "mm_per_px"]), float(idx.loc[i, "mm_per_px_y"])))
            f["lt_conf"] = float(conf.get("lt", 0.0))
            geoms.append([f.get(k, np.nan) for k in GEOM])
        return np.stack(hogs), np.asarray(geoms, float)

    sets = [stack_for(None)] + [stack_for(m) for m in mem]        # [manual, member1, member2, ...]
    y = idx.loc[ids, "v_posrot"].astype(int).to_numpy()
    groups = np.array([int(i[:3]) for i in ids])
    return ids, sets, y, groups


def evaluate(ids, sets, y, groups, use_geom, use_hog=True):
    """Out-of-fold AUC and F1 with the shipped procedure; sets[0] is the manual-crop training set."""
    def X_of(k):
        h, g = sets[k]
        g = np.nan_to_num(g, nan=np.nanmedian(g, axis=0))
        parts = ([h] if use_hog else []) + ([g] if use_geom else [])
        return np.hstack(parts)

    Xs = [X_of(k) for k in range(len(sets))]
    members = Xs[1:] or [Xs[0]]
    aucs, f1s = [], []
    for seed in SEEDS:
        prob = np.zeros(len(y))
        for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(Xs[0], y, groups):
            rf = ExtraTreesClassifier(800, min_samples_leaf=3, max_features=ROTATION_MAX_FEATURES,
                                      class_weight="balanced_subsample", random_state=0, n_jobs=4)
            rf.fit(np.vstack([Xs[0][tr]] + [m[tr] for m in members]), np.concatenate([y[tr]] * (1 + len(members))))
            prob[va] = np.mean([rf.predict_proba(m[va])[:, 1] for m in members], 0)
        pb = blend(prob, ids, ROTATION_BLEND)
        dec = np.zeros(len(y), bool)
        for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(Xs[0], y, groups):
            dec[va] = pb[va] >= best_threshold(y[tr], pb[tr])
        aucs.append(roc_auc_score(y, pb))
        f1s.append(f1_score(y, dec, zero_division=0))
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(f1s))


def stacked(ids, sets, y, groups):
    """Two storeys: the forest reads the crop, a small logistic then mixes its answer with what the
    landmarks say about the lesser trochanter — the very structure ТЗ 2.3 names for rotation."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    hogs = [h for h, _ in sets]
    geom = np.nan_to_num(sets[1][1] if len(sets) > 1 else sets[0][1], nan=0.0)
    cols = [GEOM.index(k) for k in ("lt_conf", "lt_absent", "lt_protrusion")]
    members = hogs[1:] or [hogs[0]]
    aucs, f1s = [], []
    for seed in SEEDS:
        prob = np.zeros(len(y))
        for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(hogs[0], y, groups):
            rf = ExtraTreesClassifier(800, min_samples_leaf=3, max_features=ROTATION_MAX_FEATURES,
                                      class_weight="balanced_subsample", random_state=0, n_jobs=4)
            rf.fit(np.vstack([hogs[0][tr]] + [m[tr] for m in members]), np.concatenate([y[tr]] * (1 + len(members))))
            p_tr = np.mean([rf.predict_proba(m[tr])[:, 1] for m in members], 0)
            p_va = np.mean([rf.predict_proba(m[va])[:, 1] for m in members], 0)
            Z_tr = np.column_stack([np.log(np.clip(p_tr, 1e-6, 1 - 1e-6) / (1 - np.clip(p_tr, 1e-6, 1 - 1e-6))), geom[tr][:, cols]])
            Z_va = np.column_stack([np.log(np.clip(p_va, 1e-6, 1 - 1e-6) / (1 - np.clip(p_va, 1e-6, 1 - 1e-6))), geom[va][:, cols]])
            top = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000))
            top.fit(Z_tr, y[tr])
            prob[va] = top.predict_proba(Z_va)[:, 1]
        pb = blend(prob, ids, ROTATION_BLEND)
        dec = np.zeros(len(y), bool)
        for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(hogs[0], y, groups):
            dec[va] = pb[va] >= best_threshold(y[tr], pb[tr])
        aucs.append(roc_auc_score(y, pb))
        f1s.append(f1_score(y, dec, zero_division=0))
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(f1s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="*", default=["fix_s100", "strong_s100", "base_s100"])
    args = ap.parse_args()
    ids, sets, y, groups = load(args.members)
    print(f"бёдер {len(ids)}, нарушений ротации {int(y.sum())}, признаков геометрии {len(GEOM)}")
    for name, kw in (("HOG (как в сервисе)", dict(use_geom=False)),
                     ("HOG + геометрия точек", dict(use_geom=True)),
                     ("только геометрия точек", dict(use_geom=True, use_hog=False))):
        auc, sd, f1 = evaluate(ids, sets, y, groups, **kw)
        print(f"  {name:26s} AUC {auc:.3f} ± {sd:.3f}   F1 {f1:.3f}")
    auc, sd, f1 = stacked(ids, sets, y, groups)
    print(f"  {'стекинг: лес + малый вертел':26s} AUC {auc:.3f} ± {sd:.3f}   F1 {f1:.3f}")


if __name__ == "__main__":
    main()
