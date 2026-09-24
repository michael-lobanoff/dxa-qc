"""A scanner-independent check that the iliac crests are in the frame (ТЗ 2.3, нижний уровень укладки).

The shipped feature for «некорректная укладка» is the detector's own confidence in the two crest
landmarks. On the organisers' scans it works (AUC 0.88), but the injected-violation test on a foreign
densitometer showed it does not travel: after cropping the crests away the confidence falls only from
0.97 to 0.90, against 0.97 -> 0.1 on our own scans, and the service catches 22 % of such crops instead
of nearly all.

A correctly framed lumbar scan shows two large bright wings of the pelvis in the lower corners. This
measures exactly that, from pixels only:

    pelvis = (brightness of the lower lateral corners) / (brightness of the column in the same rows)

    python scripts/crest_feature.py            # на данных организаторов
    python scripts/crest_feature.py --external data/external/mendeley_dexa_spine
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from dxaqc.decision import Monotone, best_threshold
from sklearn.metrics import f1_score

SEEDS = range(300, 310)


def pelvis_ratio(img, points, band=0.22, gap=1.6):
    """How much bright bone sits in the lower corners, relative to the column itself.

    band: share of the image height, counted from the bottom edge.
    gap:  how far from the column axis (in half-widths of the column) the corners start.
    """
    h, w = img.shape
    t, b = points.get("col_top"), points.get("col_bottom")
    if not t or not b:
        return np.nan
    y0 = int(h * (1 - band))
    rows = np.arange(y0, h)
    if len(rows) < 5:
        return np.nan
    g = img.astype(np.float32)
    # the axis position in those rows, extrapolated from the two landmarks
    denom = (b[1] - t[1]) or 1
    xs = t[0] + (b[0] - t[0]) * (rows - t[1]) / denom
    half = max(w * 0.08, 12)                       # half-width of the lumbar column in pixels
    col, side = [], []
    for y, x in zip(rows, xs):
        lo, hi = int(max(x - half, 0)), int(min(x + half, w - 1))
        col.append(g[y, lo:hi + 1].mean() if hi > lo else np.nan)
        left = g[y, :max(int(x - gap * half), 0)]
        right = g[y, min(int(x + gap * half), w - 1):]
        vals = np.concatenate([left, right])
        side.append(vals.mean() if vals.size else np.nan)
    c, s = np.nanmean(col), np.nanmean(side)
    return float(s / c) if c and np.isfinite(c) and np.isfinite(s) else np.nan


def organiser_rows(tag="ens3"):
    idx = pd.read_csv("data/train/image_index.csv")
    idx["id"] = idx.n.map("{:03d}".format) + "_" + idx.region
    idx = idx.set_index("id")
    oof = json.loads(Path(f"data/train/kp_oof_spine_{tag}.json").read_text())
    out = []
    for i in sorted(oof):
        if pd.isna(idx.loc[i, "y"]):
            continue
        img = np.asarray(Image.open(f"data/annotation/images/{i}.png"))
        out.append({"id": i, "v_pos": int(idx.loc[i, "v_pos"]),
                    "crest_conf": min(oof[i]["conf"]["crest_a"], oof[i]["conf"]["crest_b"]),
                    "pelvis": pelvis_ratio(img, oof[i]["points"])})
    return pd.DataFrame(out)


def cv(df, feats, signs):
    X = np.nan_to_num(df[feats].to_numpy(float), nan=0.0)
    y = df.v_pos.to_numpy(int)
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


def external(folder, limit=50):
    """Does the feature react to the crests being cropped away on a foreign densitometer?"""
    from dxaqc import synth
    from dxaqc.service import QCService
    svc = QCService("models")
    rng = np.random.default_rng(0)
    before, after, conf_b, conf_a = [], [], [], []
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.lower() != ".png":
            continue
        img = np.asarray(Image.open(p).convert("L"))
        if img.shape == (224, 224):
            continue
        r = svc.analyse(img)
        if r.get("unsupported") or r["region"] != "spine":
            continue
        kp = {k: (list(v) if v else None) for k, v in r["points"].items()}
        try:
            simg, _, _ = synth.spine_sample(img, kp, "cut_bottom", rng)
        except (ValueError, KeyError, TypeError):
            continue
        r2 = svc.analyse(simg)
        before.append(pelvis_ratio(img, r["points"]))
        after.append(pelvis_ratio(simg, r2["points"]))
        conf_b.append(r["measurements"]["crest_conf"])
        conf_a.append(r2["measurements"]["crest_conf"])
        if len(before) >= limit:
            break
    b, a = np.array(before), np.array(after)
    print(f"внешних снимков: {len(b)}")
    print(f"  уверенность сети в гребнях: до {np.median(conf_b):.2f} -> после обрезки {np.median(conf_a):.2f}")
    print(f"  доля яркой кости в нижних углах: до {np.nanmedian(b):.2f} -> после {np.nanmedian(a):.2f}")
    print(f"  разделяет обрезанные и целые: {np.mean(a < b):.0%} снимков стали меньше")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", default=None)
    args = ap.parse_args()
    if args.external:
        external(args.external)
        return
    df = organiser_rows()
    df.to_csv("data/train/crest_feature.csv", index=False)
    print(f"снимков {len(df)}, нарушений укладки {int(df.v_pos.sum())}")
    print(df.groupby("v_pos")[["crest_conf", "pelvis"]].median().round(3).to_string())
    for feats, signs in ((["crest_conf"], [-1]), (["pelvis"], [-1]), (["crest_conf", "pelvis"], [-1, -1])):
        auc, f1 = cv(df, feats, signs)
        print(f"  {'+'.join(feats):24s} AUC {auc:.3f}  F1 {f1:.3f}")


if __name__ == "__main__":
    main()
