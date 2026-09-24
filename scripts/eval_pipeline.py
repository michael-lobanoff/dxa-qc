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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# best_threshold comes from the shipped module on purpose: the evaluation must choose thresholds
# exactly the way fit_decision does, including DXAQC_THRESH_RULE
from dxaqc.decision import (FUSION, ROTATION_BLEND, ROTATION_MAX_FEATURES, SENS_TARGET, SYNTH_TYPES, SYNTH_WEIGHT,
                            TYPES, Monotone, best_threshold, sensitivity_threshold, spine_measurements)
from dxaqc.features import hip_features
from dxaqc.hipcrop import hip_crop, rotation_features


def spacing(idx, i):
    """(mm per pixel along x, along y) for one image, as stored by build_index."""
    return float(idx.loc[i, "mm_per_px"]), float(idx.loc[i, "mm_per_px_y"])

SEEDS = range(300, 310)
_rb = __import__("os").environ.get("DXAQC_ROT_BLEND", "")
ROT_BLEND = None if _rb == "nested" else (float(_rb) if _rb else ROTATION_BLEND)   # weight of the other hip; "nested" = chosen inside each fold
BLEND_GRID = np.arange(0, 0.55, 0.05)


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))


def blend_with_other_hip(prob, ids, alpha):
    """Mix each hip's rotation probability with the other hip of the same study (decision.blend_rotation)."""
    pos = {(i[:3], i[4:]): k for k, i in enumerate(ids)}
    out = prob.copy()
    for k, i in enumerate(ids):
        j = pos.get((i[:3], "hip_right" if i[4:] == "hip_left" else "hip_left"))
        if j is not None:
            out[k] = (1 - alpha) * prob[k] + alpha * prob[j]
    return out


def cv_type(X, y, groups, make_model, seed, syn=None):
    """Out-of-fold probability, decision (threshold from the training folds) and fusion score.
    syn: (X_syn, y_syn, study of the source scan) — only variants of training-fold scans are used."""
    prob, dec, rank = np.zeros(len(y)), np.zeros(len(y), bool), np.zeros(len(y))
    dec_s = np.zeros(len(y), bool)   # the "screening" operating point (decision.SENS_TARGET per type)
    for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
        m = make_model()
        if syn is None:
            m.fit(X[tr], y[tr])
        else:
            keep = np.isin(syn[2], groups[tr])
            m.fit(X[tr], y[tr], syn[0][keep], syn[1][keep], SYNTH_WEIGHT)
        p_tr = m.predict_proba(X[tr])[:, 1]
        prob[va] = m.predict_proba(X[va])[:, 1]
        t = best_threshold(y[tr], p_tr)
        dec[va] = prob[va] >= t
        dec_s[va] = prob[va] >= min(t, sensitivity_threshold(y[tr], p_tr, SENS_TARGET))
        rank[va] = m.rank_score(X[va])
    return prob, dec, rank, dec_s


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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="detector variant: reads kp_oof_<region>_<tag>.json, writes pipeline_eval_<tag>.json")
    ap.add_argument("--seg", default="", help="metal segmentation variant: data/train/artifact_seg_oof_<seg>.json")
    ap.add_argument("--synth", default=None, help="synthetic scans for the rare types: data/train/synth_meas_<synth>.csv")
    ap.add_argument("--members", nargs="*", default=[],
                    help="detector runs of the ensemble: rotation trains on all their crops and averages over them")
    ap.add_argument("--out", default=None, help="suffix of the output files (default: the tag)")
    args = ap.parse_args()
    sfx = f"_{args.tag}" if args.tag else ""
    out_sfx = sfx if args.out is None else (f"_{args.out}" if args.out else "")
    idx = pd.read_csv("data/train/image_index.csv")
    idx["id"] = idx.n.map("{:03d}".format) + "_" + idx.region
    idx = idx.set_index("id")
    kp = json.loads(Path("data/train/keypoints.json").read_text())
    oof = {**json.loads(Path(f"data/train/kp_oof_spine{sfx}.json").read_text()),
           **json.loads(Path(f"data/train/kp_oof_hip{sfx}.json").read_text())}
    seg_oof = json.loads(Path(f"data/train/artifact_seg_oof{'_' + args.seg if args.seg else ''}.json").read_text())
    syn_all = pd.read_csv(f"data/train/synth_meas_{args.synth}.csv") if args.synth else None
    img = lambda i: np.asarray(Image.open(f"data/annotation/images/{i}.png"))

    # ---------------- spine features
    sp = [i for i in sorted(oof) if i.endswith("spine") and not pd.isna(idx.loc[i, "y"])]
    S = []
    for i in sp:
        S.append(spine_measurements(img(i), oof[i]["points"], oof[i]["conf"], spacing(idx, i), seg_oof.get(i, 0.0)))
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
    # crops placed by every member of the detector ensemble (service.analyse does the same)
    hog_mem = [np.stack([rotation_features(hip_crop(img(i), m[i]["points"], i[4:])[0]) for i in hp])
               for m in (json.loads(Path(f"data/train/kp_oof_hip_{t}.json").read_text()) for t in args.members)] or [hog_det]

    types = {"spine": TYPES["spine"], "hip": {"v_roi": TYPES["hip"]["v_roi"], "v_posrot": "rotation"}}
    report = {"types": {}, "regions": {}}
    region_scores, oof_rows = {}, []
    for region, ids, F in (("spine", sp, S), ("hip", hp, H)):
        groups = np.array([int(i[:3]) for i in ids])
        y_img = idx.loc[ids, "y"].astype(int).to_numpy()
        runs = []
        for seed in SEEDS:
            probs, decs, decs_s, fuse = {}, {}, {}, {}
            for vt, cols in types[region].items():
                yv = idx.loc[ids, vt].astype(int).to_numpy()
                if cols == "rotation":
                    prob, dec, dec_s = np.zeros(len(yv)), np.zeros(len(yv), bool), np.zeros(len(yv), bool)
                    for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(hog_man, yv, groups):
                        rf = ExtraTreesClassifier(800, min_samples_leaf=3, max_features=ROTATION_MAX_FEATURES,
                                                  class_weight="balanced_subsample", random_state=0, n_jobs=4)
                        rf.fit(np.vstack([hog_man[tr]] + [h[tr] for h in hog_mem]), np.concatenate([yv[tr]] * (1 + len(hog_mem))))
                        # threshold from inner out-of-bag-like estimate: fit on training crops placed by the detector
                        t = best_threshold(yv[tr], np.mean([rf.predict_proba(h[tr])[:, 1] for h in hog_mem], 0))
                        prob[va] = np.mean([rf.predict_proba(h[va])[:, 1] for h in hog_mem], 0); dec[va] = prob[va] >= t
                    # weight of the other hip: chosen on the training folds by AUC, the procedure
                    # fit_decision uses on all data (a fixed weight can be forced with DXAQC_ROT_BLEND)
                    raw_rot, prob = prob.copy(), np.zeros(len(yv))
                    for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(hog_man, yv, groups):
                        a = ROT_BLEND if ROT_BLEND is not None else max(
                            BLEND_GRID, key=lambda a: roc_auc_score(yv[tr], blend_with_other_hip(raw_rot, ids, a)[tr]))
                        prob[va] = blend_with_other_hip(raw_rot, ids, a)[va]
                    fuse[vt] = prob
                    for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(hog_man, yv, groups):
                        t = best_threshold(yv[tr], prob[tr])
                        dec[va] = prob[va] >= t
                        dec_s[va] = prob[va] >= min(t, sensitivity_threshold(yv[tr], prob[tr], SENS_TARGET))
                else:
                    feats, signs = cols
                    syn = None
                    if syn_all is not None and vt in SYNTH_TYPES and set(feats) <= set(syn_all.columns):
                        sy = syn_all[syn_all.src.isin(ids)]
                        syn = (np.nan_to_num(sy[feats].to_numpy(float), nan=0.0), sy.label.to_numpy(int),
                               sy.src.str[:3].astype(int).to_numpy())
                    prob, dec, rank, dec_s = cv_type(F[feats].to_numpy(float), yv, groups,
                                                     lambda: Monotone(signs, lr_weights=True), seed, syn)
                probs[vt], decs[vt], decs_s[vt] = prob, dec, dec_s
                fuse[vt] = rank if FUSION[region] == "percentile" else prob
            # combined score for quality_prob: percentiles for the spine types, probabilities for hips,
            # then ONE calibration per region (fitted on the training folds) so that the two regions end
            # up on the same probability scale and can be pooled — without it the overall AUC collapses.
            raw = 1 - np.prod(1 - np.stack([fuse.get(vt, probs[vt]) for vt in probs]), 0)
            score = np.zeros(len(raw))
            for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(raw[:, None], y_img, groups):
                cal = LogisticRegression(C=10.0, max_iter=2000).fit(raw[tr, None], y_img[tr])
                if cal.coef_[0, 0] <= 0:
                    cal.coef_[0, 0] = 1e-3
                score[va] = cal.predict_proba(raw[va, None])[:, 1]
            runs.append((probs, decs, score, raw, np.any(np.stack(list(decs.values())), 0), decs_s))
        # aggregate over seeds: mean metrics; pooled (seed-averaged) scores for CIs
        for vt in types[region]:
            yv = idx.loc[ids, vt].astype(int).to_numpy()
            ms = [metrics(yv, r[0][vt], r[1][vt]) for r in runs]
            report["types"][f"{region}:{vt}"] = {k: round(float(np.mean([m[k] for m in ms])), 3) for k in ms[0]} | {"n_pos": int(yv.sum())}
        ms_any = [metrics(y_img, r[2], r[4]) for r in runs]
        report.setdefault("regions_any_type", {})[region] = {
            k: round(float(np.mean([m[k] for m in ms_any])), 3) for k in ms_any[0]}
        score = np.mean([r[2] for r in runs], 0)          # calibrated: comparable between regions
        raw_mean = np.mean([r[3] for r in runs], 0)         # uncalibrated: the ranking inside a region
        per_image = pd.DataFrame({"id": ids, "y": y_img, "score": score, "score_raw": raw_mean})
        for vt in types[region]:
            per_image[vt] = idx.loc[ids, vt].astype(int).to_numpy()
            per_image[f"p_{vt}"] = np.mean([r[0][vt] for r in runs], 0)
            per_image[f"d_{vt}"] = (np.mean([r[1][vt] for r in runs], 0) >= 0.5).astype(int)
            per_image[f"s_{vt}"] = (np.mean([r[5][vt] for r in runs], 0) >= 0.5).astype(int)
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
    raw_all = allrows.score_raw.to_numpy(float)
    for name, mask in (("spine", is_spine), ("hip", ~is_spine), ("overall", np.ones(len(y_all), bool))):
        # inside one region the shipped calibration is a single monotone map, so the ranking there is
        # the uncalibrated one; across regions only the calibrated score is comparable.
        rank = raw_all if name != "overall" else s_all
        ms = [metrics(y_all[mask], rank[mask], d[mask]) for d in dec_runs]
        ci = bootstrap(y_all[mask], rank[mask], d_all[mask])
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

    # The second operating point the service offers (DXAQC_POLICY=screening): per-type thresholds that
    # catch SENS_TARGET of each violation instead of maximising its F1. Same models, same folds.
    scr = allrows[[f"s_{vt}" for vt in TYPES["spine"] | TYPES["hip"] if f"s_{vt}" in allrows]].fillna(0).sum(axis=1).to_numpy() > 0
    report["screening"] = {"image": {k: round(float(v), 3) for k, v in metrics(y_all, s_all, scr).items()}}
    st_s = allrows.assign(study=allrows.id.str[:3].astype(int), dec=scr.astype(int)).groupby("study").agg(
        y=("y", "max"), dec=("dec", "max"), score=("score", "max"))
    report["screening"]["studies"] = {k: round(float(v), 3)
                                      for k, v in metrics(st_s.y.to_numpy(int), st_s.score.to_numpy(float),
                                                          st_s.dec.to_numpy(bool)).items()}
    report["screening"]["macro_f1_types"] = round(float(np.mean([
        f1_score(allrows.loc[allrows[vt].notna(), vt].astype(int), allrows.loc[allrows[vt].notna(), f"s_{vt}"])
        for vt in TYPES["spine"] | TYPES["hip"] if f"s_{vt}" in allrows])), 3)

    print("Per violation type (mean over 10 CV repeats):")
    for k, v in report["types"].items():
        print(f"  {k:18s} AUC {v['auc']:.3f}  F1 {v['f1']:.3f}  sens {v['sens']:.2f}  spec {v['spec']:.2f}  (positives {v['n_pos']})")
    print("\nImage verdict (quality_class, shipped policy 'balanced'), mean over repeats [95% bootstrap CI]:")
    for r, v in report["regions"].items():
        print(f"  {r:8s} n={v['n'][0]} ({v['n'][1]} bad)  AUC {v['auc'][0]:.3f} {v['auc'][1]}  F1 {v['f1'][0]:.3f} {v['f1'][1]}  "
              f"PR-AUC {v['pr_auc'][0]:.3f}  sens {v['sens'][0]:.2f}  spec {v['spec'][0]:.2f}  bAcc {v['bacc'][0]:.3f}")
    v = report["studies"]
    print(f"\nStudy verdict (ТЗ 8.4: «чувствительность выявления исследований»): n={v['n'][0]} ({v['n'][1]} bad)  "
          f"AUC {v['auc'][0]:.3f} {v['auc'][1]}  F1 {v['f1'][0]:.3f} {v['f1'][1]}  sens {v['sens'][0]:.2f}  spec {v['spec'][0]:.2f}")
    print(f"macro-F1 over violation types: {report['macro_f1_types']:.3f}")
    sc = report["screening"]
    print(f"\nAlternative operating point DXAQC_POLICY=screening (per-type sensitivity target {SENS_TARGET:.2f}):")
    print(f"  images  F1 {sc['image']['f1']:.3f}  sens {sc['image']['sens']:.2f}  spec {sc['image']['spec']:.2f}  bAcc {sc['image']['bacc']:.3f}")
    print(f"  studies F1 {sc['studies']['f1']:.3f}  sens {sc['studies']['sens']:.2f}  spec {sc['studies']['spec']:.2f}  "
          f"| macro-F1 {sc['macro_f1_types']:.3f}")
    Path(f"data/train/pipeline_eval{out_sfx}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    pd.concat(oof_rows).to_csv(f"data/train/pipeline_oof{out_sfx}.csv", index=False)


if __name__ == "__main__":
    main()
