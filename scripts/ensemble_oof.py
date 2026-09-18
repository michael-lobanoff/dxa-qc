"""Out-of-fold keypoints of a seed ensemble: the fold models of several detector runs, averaged.

All runs of train_keypoints.py use the same study-grouped split (random_state=0), so fold k of every run
left out the same studies, and averaging their heatmaps is still an honest out-of-fold prediction.

    python scripts/ensemble_oof.py --tags strong_s100 strong_s200 strong_s300 --out strong_ens3
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dxaqc import kpmodel as K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--regions", nargs="+", default=["spine", "hip"])
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    for region in args.regions:
        names, oof = K.POINTS[region], {}
        samples = {s.id: s for s in K.load_samples(region)}
        for fold in range(5):
            nets, val = [], None
            for tag in args.tags:
                ck = torch.load(Path("data/train/kp_folds", tag, f"{region}_f{fold}.pt"), map_location="cpu", weights_only=False)
                assert val is None or ck["val_ids"] == val, "runs were trained on different splits"
                val = ck["val_ids"]
                net = K.KPNet(len(names)); net.load_state_dict(ck["state_dict"])
                nets.append(net.to(device))
            preds = K.predict(nets, [samples[i].img for i in val], region, device)
            for i, (p, conf) in zip(val, preds):
                s = samples[i]
                if s.flipped:
                    p = p.copy(); p[:, 0] = s.img.shape[1] - 1 - p[:, 0]
                oof[i] = {"points": {k: (None if np.isnan(q).any() else [round(float(q[0]), 1), round(float(q[1]), 1)])
                                     for k, q in zip(names, p)},
                          "conf": {k: round(float(c), 3) for k, c in zip(names, conf)}}
        out = Path(f"data/train/kp_oof_{region}_{args.out}.json")
        out.write_text(json.dumps(oof, ensure_ascii=False))
        print(f"{region}: {len(oof)} images -> {out}")


if __name__ == "__main__":
    main()
