"""Train the landmark detector with study-grouped cross-validation, then a final model on all data.

Out-of-fold predictions are saved so the QC rules can be evaluated on predicted points
exactly as they would run on unseen scans.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

from dxaqc import kpmodel as K
from dxaqc.geometry import MM_PER_PX


def train_one(samples, region, epochs, device, seed=0, log_every=0):
    torch.manual_seed(seed)
    model = K.KPNet(len(K.POINTS[region])).to(device)
    ds = K.KPDataset(samples, region, train=True, seed=seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, drop_last=True, num_workers=4,
                                     persistent_workers=True, worker_init_fn=K.worker_init)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * len(dl), pct_start=0.1)
    for ep in range(epochs):
        model.train()
        tot = []
        for x, hm, vis in dl:
            x, hm, vis = x.to(device), hm.to(device), vis.to(device)
            hm_p, vis_p = model(x)
            loss, l_hm, l_vis = K.loss_fn(hm_p, vis_p, hm, vis)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot.append((l_hm, l_vis))
        if log_every and (ep + 1) % log_every == 0:
            print(f"    epoch {ep + 1}: heatmap {np.mean([t[0] for t in tot]):.3f}  in-frame {np.mean([t[1] for t in tot]):.3f}", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", choices=["spine", "hip"], required=True)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only-fold", type=int, default=None, help="debug: run a single fold")
    ap.add_argument("--no-final", action="store_true")
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    samples = K.load_samples(args.region)
    idx = pd.read_csv("data/train/image_index.csv")
    y_of = {f"{r.n:03d}_{r.region}": int(r.y) if not pd.isna(r.y) else 0 for r in idx.itertuples()}
    y = np.array([y_of[s.id] for s in samples]); groups = np.array([s.n for s in samples])
    cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=0)
    names = K.POINTS[args.region]
    oof = {}
    for fold, (tr, va) in enumerate(cv.split(np.zeros(len(samples)), y, groups)):
        if args.only_fold is not None and fold != args.only_fold:
            continue
        t0 = time.time()
        model = train_one([samples[i] for i in tr], args.region, args.epochs, device, seed=fold, log_every=40)
        preds = K.predict(model, [samples[i].img for i in va], args.region, device)
        for i, (p, conf) in zip(va, preds):
            s = samples[i]
            if s.flipped:  # back to original orientation for the rules
                p = p.copy(); p[:, 0] = s.img.shape[1] - 1 - p[:, 0]
            oof[s.id] = {"points": {k: (None if np.isnan(q).any() else [round(float(q[0]), 1), round(float(q[1]), 1)]) for k, q in zip(names, p)},
                         "conf": {k: round(float(c), 3) for k, c in zip(names, conf)}}
        print(f"fold {fold}: {len(tr)} train / {len(va)} val, {time.time() - t0:.0f}s", flush=True)

    # ---- point errors against the manual annotation (original orientation)
    kp = json.loads(Path("data/train/keypoints.json").read_text())
    rows = []
    for image_id, pr in oof.items():
        for k in names:
            gt, pd_ = kp[image_id]["points"].get(k), pr["points"][k]
            rows.append({"k": k, "has_gt": gt is not None, "has_pred": pd_ is not None,
                         "err": float(np.hypot(gt[0] - pd_[0], gt[1] - pd_[1])) if gt and pd_ else np.nan})
    df = pd.DataFrame(rows)
    print(f"\n{args.region}: {len(oof)} out-of-fold images")
    for k in names:
        d = df[df.k == k]; e = d.err.dropna()
        acc = (d.has_gt == d.has_pred).mean()
        print(f"  {k:10s} median {e.median():5.1f}px ({e.median() * MM_PER_PX:4.1f} mm)  p90 {e.quantile(.9):5.1f}px  "
              f"in-frame acc {acc:.0%}  (absent in GT: {int((~d.has_gt).sum())}, missed/extra: {int((d.has_gt & ~d.has_pred).sum())}/{int((~d.has_gt & d.has_pred).sum())})")
    out = Path(f"data/train/kp_oof_{args.region}.json")
    out.write_text(json.dumps(oof, ensure_ascii=False))
    print("saved", out)

    if not args.no_final and args.only_fold is None:
        model = train_one(samples, args.region, args.epochs, device, seed=42)
        Path("models").mkdir(exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "region": args.region, "points": names,
                    "size": K.SIZE, "margin": K.MARGIN}, f"models/kp_{args.region}.pt")
        print(f"saved models/kp_{args.region}.pt")


if __name__ == "__main__":
    main()
