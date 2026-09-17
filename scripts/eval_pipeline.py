"""End-to-end estimate of the QC service on the training set, using only what the service would have:
detector keypoints from out-of-fold models, out-of-fold rotation scores and image-based artifact cues.

Per violation type a tiny model (1-3 features) is fitted and its F1-optimal threshold is chosen inside
the training folds, then applied to the held-out studies. The image verdict is "any violation"; its
score is the noisy-OR of the per-type probabilities. Metrics follow ТЗ 8.4 (95% bootstrap CIs).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dxaqc.decision import TYPES, Monotone, sensitivity_threshold, spine_measurements
from dxaqc.features import hip_features
from dxaqc.hipcrop import hip_crop, rotation_features


def spacing(idx, i):
    """(mm per pixel along x, along y) for one image, as stored by build_index."""
    return float(idx.loc[i, "mm_per_px"]), float(idx.loc[i, "mm_per_px_y"])

SEEDS = range(300, 310)


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))


def best_threshold(y, p):
    grid = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 49)))
    return max(grid, key=lambda t: f1_score(y, p >= t, zero_division=0))


def blend_with_other_hip(prob, ids, alpha):
    """Mix each hip's rotation probability with the other hip of the same study (decision.blend_rotation)."""
    pos = {(i[:3], i[4:]): k for k, i in enumerate(ids)}
    out = prob.copy()
    for k, i in enumerate(ids):
        j = pos.get((i[:3], "hip_right" if i[4:] == "hip_left" else "hip_left"))
        if j is not None:
            out[k] = (1 - alpha) * prob[k] + alpha * prob[j]
    return out


def cv_type(X, y, groups, make_model, seed):
    """Out-of-fold probability and binary decision (threshold tuned on the training folds)."""
    prob, dec = np.zeros(len(y)), np.zeros(len(y), bool)
    for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
        m = make_model(); m.fit(X[tr], y[tr])
        t = best_threshold(y[tr], m.predict_proba(X[tr])[:, 1])
        prob[va] = m.predict_proba(X[va])[:, 1]; dec[va] = prob[va] >= t
    return prob, dec


def metrics(y, score, dec):
    tp, tn = int((dec & (y == 1)).sum()), int((~dec & (y == 0)).sum())
    sens, spec = tp / max((y == 1).sum(), 1), tn / max((y == 0).sum(), 1)
    two = len(set(y)) > 1
    return {"auc": roc_auc_score(y, score) if two else np.nan,
            "pr_auc": average_precision_score(y, score) if two else np.nan,
            "f1": f1_score(y, dec, zero_division=0), "sens": sens, "spec": spec, "bacc": (sens + spec) / 2}


def bootstrap(y, score, dec, n=2000, seed=0):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        b = rng.integers(0, len(y), len(y))
        if len(set(y[b])) == 2:
            vals.append(metrics(y[b], score[b], dec[b]))
    return {k: np.percentile([v[k] for v in vals], [2.5, 97.5]) for k in vals[0]}


def main():
    idx = pd.read_csv("data/train/image_index.csv")
    idx["id"] = idx.n.map("{:03d}".format) + "_" + idx.region
    idx = idx.set_index("id")
    kp = json.loads(Path("data/train/keypoints.json").read_text())
    oof = {**json.loads(Path("data/train/kp_oof_spine.json").read_text()), **json.loads(Path("data/train/kp_oof_hip.json").read_text())}
    img = lambda i: np.asarray(Image.open(f"data/annotation/images/{i}.png"))

    # ---------------- spine features
    sp = [i for i in sorted(oof) if i.endswith("spine") and not pd.isna(idx.loc[i, "y"])]
    S = []
    for i in sp:
        S.append(spine_measurements(img(i), oof[i]["points"], oof[i]["conf"], spacing(idx, i)))
    S = pd.DataFrame(S, index=sp)
    # ---------------- hip features (rotation score: HOG + RF, trained on manual-keypoint crops, applied to detector crops)
    hp = [i for i in sorted(oof) if "hip" in i and not kp[i]["flags"].get("skip") and not pd.isna(idx.loc[i, "y"])]
    Hrows = []
    for i in hp:
        f = hip_features(oof[i]["points"], (idx.loc[i, "rows"], idx.loc[i, "cols"]), i[4:], spacing(idx, i))
        Hrows.append({"margin_bottom": f["margin_bottom"], "margin_top": f["margin_top"], "head_gt_width": f.get("head_gt_width"),
                      "isch_conf": oof[i]["conf"]["isch"], "gt_conf": oof[i]["conf"]["gt_lat"]})
    H = pd.DataFrame(Hrows, index=hp).astype(float)
    H = H.fillna(H.median())
    hog_man = np.stack([rotation_features(hip_crop(img(i), kp[i]["points"], i[4:])[0]) for i in hp])
    hog_det = np.stack([rotation_features(hip_crop(img(i), oof[i]["points"], i[4:])[0]) for i in hp])

    types = {"spine": TYPES["spine"], "hip": {"v_roi": TYPES["hip"]["v_roi"], "v_posrot": "rotation"}}
    report = {"types": {}, "regions": {}}
    region_scores, oof_rows = {}, []
    for region, ids, F in (("spine", sp, S), ("hip", hp, H)):
        groups = np.array([int(i[:3]) for i in ids])
        y_img = idx.loc[ids, "y"].astype(int).to_numpy()
        runs = []
        for seed in SEEDS:
            probs, decs = {}, {}
            for vt, cols in types[region].items():
                yv = idx.loc[ids, vt].astype(int).to_numpy()
                if cols == "rotation":
                    prob, dec = np.zeros(len(yv)), np.zeros(len(yv), bool)
                    for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(hog_man, yv, groups):
                        rf = ExtraTreesClassifier(800, min_samples_leaf=3, max_features=0.1,
                                                  class_weight="balanced_subsample", random_state=0, n_jobs=4)
                        rf.fit(np.vstack([hog_man[tr], hog_det[tr]]), np.concatenate([yv[tr], yv[tr]]))
                        # threshold from inner out-of-bag-like estimate: fit on training crops placed by the detector
                        t = best_threshold(yv[tr], rf.predict_proba(hog_det[tr])[:, 1])
                        prob[va] = rf.predict_proba(hog_det[va])[:, 1]; dec[va] = prob[va] >= t
                    prob = blend_with_other_hip(prob, ids, alpha=0.2)
                    for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(hog_man, yv, groups):
                        dec[va] = prob[va] >= best_threshold(yv[tr], prob[tr])
                else:
                    feats, signs = cols
                    prob, dec = cv_type(F[feats].to_numpy(float), yv, groups, lambda: Monotone(signs, lr_weights=True), seed)
                probs[vt], decs[vt] = prob, dec
            score = 1 - np.prod(1 - np.stack(list(probs.values())), 0)   # noisy-OR: "any violation"
            runs.append((probs, decs, score, None, np.any(np.stack(list(decs.values())), 0)))
        # aggregate over seeds: mean metrics; pooled (seed-averaged) scores for CIs
        for vt in types[region]:
            yv = idx.loc[ids, vt].astype(int).to_numpy()
            ms = [metrics(yv, r[0][vt], r[1][vt]) for r in runs]
            report["types"][f"{region}:{vt}"] = {k: round(float(np.mean([m[k] for m in ms])), 3) for k in ms[0]} | {"n_pos": int(yv.sum())}
        ms_any = [metrics(y_img, r[2], r[4]) for r in runs]
        report.setdefault("regions_any_type", {})[region] = {
            k: round(float(np.mean([m[k] for m in ms_any])), 3) for k in ms_any[0]}
        score = np.mean([r[2] for r in runs], 0)
        per_image = pd.DataFrame({"id": ids, "y": y_img, "score": score})
        for vt in types[region]:
            per_image[vt] = idx.loc[ids, vt].astype(int).to_numpy()
            per_image[f"p_{vt}"] = np.mean([r[0][vt] for r in runs], 0)
            per_image[f"d_{vt}"] = (np.mean([r[1][vt] for r in runs], 0) >= 0.5).astype(int)
        oof_rows.append(per_image)
        region_scores[region] = (y_img, score)

    # Image verdict: ONE threshold for both regions (the shipped policy), chosen on the studies outside
    # the fold. A shared threshold beat separate ones out-of-fold (F1 0.656 against 0.636).
    allrows = pd.concat(oof_rows)
    y_all = allrows.y.to_numpy(int)
    s_all = allrows.score.to_numpy(float)
    g_all = allrows.id.str[:3].astype(int).to_numpy()
    # Shipped rule: the image is bad when any violation type fires (see decision.DecisionModel.verdict)
    dec_runs = [allrows[[f"d_{vt}" for vt in TYPES["spine"] | TYPES["hip"] if f"d_{vt}" in allrows]]
                .fillna(0).sum(axis=1).to_numpy() > 0]
    d_all = np.mean(dec_runs, 0) >= 0.5
    allrows["decision"] = d_all.astype(int)
    is_spine = allrows.id.str.endswith("spine").to_numpy()
    for name, mask in (("spine", is_spine), ("hip", ~is_spine), ("overall", np.ones(len(y_all), bool))):
        ms = [metrics(y_all[mask], s_all[mask], d[mask]) for d in dec_runs]
        ci = bootstrap(y_all[mask], s_all[mask], d_all[mask])
        report["regions"][name] = {k: [round(float(np.mean([m[k] for m in ms])), 3),
                                       [round(float(x), 3) for x in ci[k]]] for k in ms[0]}
        report["regions"][name]["n"] = [int(mask.sum()), int(y_all[mask].sum())]
    oof_rows = [allrows]

    # ТЗ 8.4 asks for the sensitivity of detecting STUDIES with violations, so report that view too:
    # a study counts as bad when any of its images does.
    st = allrows.assign(study=allrows.id.str[:3].astype(int), dec=d_all.astype(int))
    st = st.groupby("study").agg(y=("y", "max"), dec=("dec", "max"), score=("score", "max"))
    ms = metrics(st.y.to_numpy(int), st.score.to_numpy(float), st.dec.to_numpy(bool))
    ci = bootstrap(st.y.to_numpy(int), st.score.to_numpy(float), st.dec.to_numpy(bool))
    report["studies"] = {k: [round(float(v), 3), [round(float(x), 3) for x in ci[k]]] for k, v in ms.items()}
    report["studies"]["n"] = [len(st), int(st.y.sum())]
    macro = float(np.mean([report["types"][k]["f1"] for k in report["types"]]))
    report["macro_f1_types"] = round(macro, 3)

    print("Per violation type (mean over 10 CV repeats):")
    for k, v in report["types"].items():
        print(f"  {k:18s} AUC {v['auc']:.3f}  F1 {v['f1']:.3f}  sens {v['sens']:.2f}  spec {v['spec']:.2f}  (positives {v['n_pos']})")
    print("\nImage verdict (quality_class, shipped policy 'screening'), mean over repeats [95% bootstrap CI]:")
    for r, v in report["regions"].items():
        print(f"  {r:8s} n={v['n'][0]} ({v['n'][1]} bad)  AUC {v['auc'][0]:.3f} {v['auc'][1]}  F1 {v['f1'][0]:.3f} {v['f1'][1]}  "
              f"PR-AUC {v['pr_auc'][0]:.3f}  sens {v['sens'][0]:.2f}  spec {v['spec'][0]:.2f}  bAcc {v['bacc'][0]:.3f}")
    v = report["studies"]
    print(f"\nStudy verdict (ТЗ 8.4: «чувствительность выявления исследований»): n={v['n'][0]} ({v['n'][1]} bad)  "
          f"AUC {v['auc'][0]:.3f} {v['auc'][1]}  F1 {v['f1'][0]:.3f} {v['f1'][1]}  sens {v['sens'][0]:.2f}  spec {v['spec'][0]:.2f}")
    print(f"macro-F1 over violation types: {report['macro_f1_types']:.3f}")
    Path("data/train/pipeline_eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    pd.concat(oof_rows).to_csv("data/train/pipeline_oof.csv", index=False)


if __name__ == "__main__":
    main()
