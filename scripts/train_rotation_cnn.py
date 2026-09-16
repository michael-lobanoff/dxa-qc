"""Cross-validate the rotation CNN (study-grouped folds), optionally validating on crops placed by the
detector's out-of-fold keypoints, and compare / ensemble it with the HOG + boosting baseline."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from dxaqc.hipcrop import hip_crop
from dxaqc.rotnet import predict_rotnet, train_rotnet
from dxaqc.hog import hog
from rotation_experiments import bootstrap_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", action="store_true", help="validation crops from detector out-of-fold keypoints")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    kp = json.loads(Path("data/train/keypoints.json").read_text())
    oof_kp = json.loads(Path("data/train/kp_oof_hip.json").read_text()) if args.oof else {}
    idx = pd.read_csv("data/train/image_index.csv")
    idx["id"] = idx.n.map("{:03d}".format) + "_" + idx.region
    idx = idx.set_index("id")
    ids = [i for i in sorted(kp) if "hip" in i and not kp[i]["flags"].get("skip") and not pd.isna(idx.loc[i, "v_posrot"])]
    imgs = {i: np.asarray(Image.open(f"data/annotation/images/{i}.png")) for i in ids}
    y = idx.loc[ids, "v_posrot"].astype(int).to_numpy(); g = np.array([int(i[:3]) for i in ids])
    train_items = [(imgs[i], kp[i]["points"], i[4:], int(t)) for i, t in zip(ids, y)]
    val_items = [(imgs[i], (oof_kp[i]["points"] if args.oof else kp[i]["points"]), i[4:], int(t)) for i, t in zip(ids, y)]
    hog_tr = np.stack([hog(hip_crop(*it[:3])[0], 16) for it in train_items])
    hog_va = np.stack([hog(hip_crop(*it[:3])[0], 16) for it in val_items])

    res = {"cnn": [], "hgb": [], "ens": []}
    pooled = {k: np.zeros(len(y)) for k in res}
    for r in range(args.repeats):
        oof = {k: np.zeros(len(y)) for k in res}
        for fold, (tr, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=100 + r).split(hog_tr, y, g)):
            t0 = time.time()
            net = train_rotnet([train_items[i] for i in tr], device, epochs=args.epochs, seed=r * 10 + fold)
            oof["cnn"][va] = predict_rotnet(net, [val_items[i] for i in va], device)
            hgb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, class_weight="balanced", random_state=0)
            hgb.fit(hog_tr[tr], y[tr]); oof["hgb"][va] = hgb.predict_proba(hog_va[va])[:, 1]
            print(f"  repeat {r} fold {fold}: {time.time() - t0:.0f}s", flush=True)
        # rank-average ensemble: the two models' scores live on different scales
        oof["ens"] = (pd.Series(oof["cnn"]).rank().to_numpy() + pd.Series(oof["hgb"]).rank().to_numpy()) / 2
        for k in res:
            res[k].append(roc_auc_score(y, oof[k])); pooled[k] += oof[k] / args.repeats
    print(f"\nrotation / positioning, {len(y)} hips ({y.sum()} violations), validation crops from "
          f"{'detector OOF' if args.oof else 'manual'} keypoints")
    summary = {}
    for k in res:
        lo, hi = bootstrap_ci(y, pooled[k])
        summary[k] = {"auc": round(float(np.mean(res[k])), 3), "sd": round(float(np.std(res[k])), 3), "ci95": [round(lo, 3), round(hi, 3)]}
        print(f"  {k:4s} AUC {np.mean(res[k]):.3f} ± {np.std(res[k]):.3f}  (95% CI {lo:.2f}–{hi:.2f})")
    Path(f"data/train/rotation_cnn_{'oof' if args.oof else 'manual'}.json").write_text(json.dumps(summary, indent=1))
    np.save(f"data/train/rotation_scores_{'oof' if args.oof else 'manual'}.npy", np.stack([pooled[k] for k in res]))


if __name__ == "__main__":
    main()
