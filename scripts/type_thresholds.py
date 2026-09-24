"""Which rule should pick the threshold of each violation type?

The verdict of the service is "any type fired", so the per-type thresholds drive both macro-F1 and the
image verdict. With 6-36 positives per type, the F1-optimal threshold found inside a training fold is
noisy: for the axis tilt the ranking is good (AUC 0.893) while F1 is 0.42, because the threshold ends up
flagging about a fifth of the scans at a true rate of one in ten.

This compares threshold rules on the stored out-of-fold probabilities (data/train/pipeline_oof_<tag>.csv):
the rule sees only the training studies of each fold, the metrics come from the held-out ones, and the
same repeated study-grouped splits are used for every rule.

    python scripts/type_thresholds.py --tag _base24
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, fbeta_score
from sklearn.model_selection import StratifiedGroupKFold

SEEDS = range(300, 310)
TYPES = ["v_axis", "v_pos", "v_artifact", "v_roi", "v_posrot"]


def grid(p):
    return np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97)))


def pick(y, p, rule, rng=None):
    g = grid(p)
    if rule == "f1":                      # what the service ships
        return max(g, key=lambda t: f1_score(y, p >= t, zero_division=0))
    if rule == "f05":                     # precision matters twice as much as recall
        return max(g, key=lambda t: fbeta_score(y, p >= t, beta=0.5, zero_division=0))
    if rule == "prev":
        # flag as many scans as the training prevalence: with a good ranking this is the F1 optimum
        # in expectation, and it does not depend on where a handful of positives happen to sit
        return float(np.quantile(p, 1 - y.mean()))
    if rule == "f1_smooth":               # add-one smoothing damps the jump on a single scan
        def sm(t):
            d = p >= t
            tp, fp, fn = ((d & (y == 1)).sum(), (d & (y == 0)).sum(), ((~d) & (y == 1)).sum())
            return (2 * tp + 1) / (2 * tp + fp + fn + 2)
        return max(g, key=sm)
    if rule == "f1_bag":                  # average of the F1 optima over bootstrap resamples
        rng = rng or np.random.default_rng(0)
        ts = []
        for _ in range(25):
            b = rng.integers(0, len(y), len(y))
            if len(set(y[b])) < 2:
                continue
            ts.append(max(g, key=lambda t: f1_score(y[b], p[b] >= t, zero_division=0)))
        return float(np.median(ts)) if ts else pick(y, p, "f1")
    if rule == "theory":
        # for a calibrated probability the F1 optimum sits at half the achievable F1 (Zhao et al.):
        # a fixed point that depends on the whole distribution, not on where two positives landed
        t = 0.5
        for _ in range(50):
            f = f1_score(y, p >= t, zero_division=0)
            t_new = max(f / 2, 1e-6)
            if abs(t_new - t) < 1e-4:
                break
            t = t_new
        return float(t)
    if rule.startswith("prev") and len(rule) > 4:      # flag k times the prevalence
        k = float(rule[4:]) / 10
        return float(np.quantile(p, max(1 - k * y.mean(), 0.0)))
    if rule == "mix":                     # halfway between the F1 optimum and the prevalence quantile
        return float(np.mean([pick(y, p, "f1"), pick(y, p, "prev")]))
    raise ValueError(rule)


def run(df, rules):
    rows, per_type = [], {}
    ids = df.id.to_numpy()
    groups_all = df.id.str[:3].astype(int).to_numpy()
    for rule in rules:
        f1s, seeds_img = {t: [] for t in TYPES}, []
        for seed in SEEDS:
            fired = np.zeros(len(df), bool)
            for t in TYPES:
                m = df[t].notna().to_numpy()
                y, p = df[t].to_numpy()[m].astype(int), df[f"p_{t}"].to_numpy()[m].astype(float)
                g, dec = groups_all[m], np.zeros(m.sum(), bool)
                for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(p[:, None], y, g):
                    dec[va] = p[va] >= pick(y[tr], p[tr], rule, np.random.default_rng(seed))
                f1s[t].append(f1_score(y, dec, zero_division=0))
                fired[np.where(m)[0][dec]] = True
            yi = df.y.to_numpy(int)
            seeds_img.append((f1_score(yi, fired, zero_division=0),
                              (fired & (yi == 1)).sum() / max((yi == 1).sum(), 1),
                              ((~fired) & (yi == 0)).sum() / max((yi == 0).sum(), 1)))
        macro = float(np.mean([np.mean(f1s[t]) for t in TYPES]))
        img = np.mean(seeds_img, axis=0)
        per_type[rule] = {t: float(np.mean(f1s[t])) for t in TYPES}
        rows.append({"rule": rule, "macro_F1": macro, "img_F1": img[0], "img_sens": img[1], "img_spec": img[2]})
    return pd.DataFrame(rows), per_type


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_base24")
    ap.add_argument("--rules", nargs="+", default=["f1", "f05", "prev", "f1_smooth", "f1_bag", "mix"])
    args = ap.parse_args()
    df = pd.read_csv(f"data/train/pipeline_oof{args.tag}.csv")
    table, per_type = run(df, args.rules)
    print(table.round(3).to_string(index=False))
    print()
    print(pd.DataFrame(per_type).round(3).to_string())


if __name__ == "__main__":
    main()
