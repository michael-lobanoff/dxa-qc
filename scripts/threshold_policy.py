"""Which operating point should the service ship with?

ROC-AUC does not depend on the threshold, but F1, sensitivity and specificity do, and ТЗ 8.4 scores
all of them. This compares policies on the honest out-of-fold scores (data/train/pipeline_oof.csv),
choosing each policy's threshold inside training folds and measuring it on held-out studies, so the
numbers are not the optimistic "best threshold on the same data".

Usage: python scripts/threshold_policy.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, fbeta_score
from sklearn.model_selection import StratifiedGroupKFold

SEEDS = range(300, 310)


def pick(y, p, policy):
    """Threshold chosen on training folds only."""
    grid = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97)))
    if policy == "f1":
        return max(grid, key=lambda t: f1_score(y, p >= t, zero_division=0))
    if policy == "f2":
        return max(grid, key=lambda t: fbeta_score(y, p >= t, beta=2, zero_division=0))
    if policy == "bacc":
        return max(grid, key=lambda t: (((p >= t) & (y == 1)).sum() / max((y == 1).sum(), 1)
                                        + ((p < t) & (y == 0)).sum() / max((y == 0).sum(), 1)) / 2)
    if policy.startswith("sens"):
        target = float(policy[4:]) / 100
        ok = [t for t in grid if ((p >= t) & (y == 1)).sum() / max((y == 1).sum(), 1) >= target]
        return max(ok) if ok else grid.min()     # the most specific threshold that still reaches the target
    raise ValueError(policy)


def evaluate(df, policies):
    y, p, groups = df.y.to_numpy(int), df.score.to_numpy(float), df.id.str[:3].astype(int).to_numpy()
    out = {}
    for policy in policies:
        rows = []
        for seed in SEEDS:
            dec = np.zeros(len(y), bool)
            for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(p[:, None], y, groups):
                dec[va] = p[va] >= pick(y[tr], p[tr], policy)
            tp, fn = int((dec & (y == 1)).sum()), int((~dec & (y == 1)).sum())
            fp, tn = int((dec & (y == 0)).sum()), int((~dec & (y == 0)).sum())
            sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
            rows.append((f1_score(y, dec, zero_division=0), sens, spec, (sens + spec) / 2, fp))
        m = np.mean(rows, 0)
        out[policy] = dict(f1=m[0], sens=m[1], spec=m[2], bacc=m[3], fp=m[4])
    return out


def main():
    df = pd.read_csv("data/train/pipeline_oof.csv")
    policies = ["f1", "bacc", "f2", "sens80", "sens90"]
    print("Current service (per-type F1 thresholds, verdict = any type fires):")
    for name, d in (("overall", df), ("spine", df[df.id.str.endswith("spine")]), ("hip", df[df.id.str.contains("hip")])):
        y, dec = d.y.to_numpy(int), d.decision.to_numpy(int)
        tp, fn = int((dec & (y == 1)).sum()), int(((1 - dec) & (y == 1)).sum())
        fp, tn = int((dec & (y == 0)).sum()), int(((1 - dec) & (y == 0)).sum())
        print(f"  {name:8s} F1 {f1_score(y, dec, zero_division=0):.3f}  sens {tp / max(tp + fn, 1):.2f}  "
              f"spec {tn / max(tn + fp, 1):.2f}  bAcc {((tp / max(tp + fn, 1)) + (tn / max(tn + fp, 1))) / 2:.3f}  FP {fp}")

    print("\nImage-level threshold on the noisy-OR score (threshold picked on training folds):")
    print(f"  {'policy':8s} {'F1':>6s} {'sens':>6s} {'spec':>6s} {'bAcc':>6s} {'FP':>5s}")
    for name, d in (("overall", df), ("spine", df[df.id.str.endswith("spine")]), ("hip", df[df.id.str.contains("hip")])):
        res = evaluate(d, policies)
        print(f"  --- {name} (n={len(d)}, bad={int(d.y.sum())})")
        for policy, m in res.items():
            print(f"  {policy:8s} {m['f1']:6.3f} {m['sens']:6.2f} {m['spec']:6.2f} {m['bacc']:6.3f} {m['fp']:5.1f}")


if __name__ == "__main__":
    main()
