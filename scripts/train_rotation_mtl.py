"""Multi-task hip network: landmark heatmaps (dense anatomical supervision) + positioning/rotation logit.

Within every CV fold the whole network is trained from scratch on the training studies only, so the
rotation AUC on the held-out studies is honest. Compared and rank-ensembled with HOG + RandomForest.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from dxaqc import kpmodel as K
from dxaqc.hipcrop import hip_crop
from dxaqc.hog import hog
from rotation_experiments import bootstrap_ci


class MTLNet(K.KPNet):
    def __init__(self, k):
        super().__init__(k)
        self.cls = nn.Sequential(nn.Dropout(0.5), nn.Linear(256 + 128, 1))

    def forward(self, x):
        b, _, h, w = x.shape
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device), torch.linspace(-1, 1, w, device=x.device), indexing="ij")
        x = torch.cat([x, xx.expand(b, 1, h, w), yy.expand(b, 1, h, w)], 1)
        feats = []
        for i, blk in enumerate(self.enc):
            x = blk(x if i == 0 else F.max_pool2d(x, 2))
            feats.append(x)
        vis = self.vis(feats[-1].mean((2, 3)))
        logit = self.cls(torch.cat([feats[-1].mean((2, 3)), feats[2].mean((2, 3))], 1)).squeeze(1)
        x = feats[-1]
        for blk, skip in zip(self.up, (feats[3], feats[2], feats[1])):
            x = blk(torch.cat([F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip], 1))
        return self.hm(x), vis, logit


class MTLDataset(K.KPDataset):
    def __init__(self, samples, labels, train, seed=0):
        super().__init__(samples, "hip", train, seed)
        self.labels = labels

    def __getitem__(self, i):
        x, hm, vis = super().__getitem__(i)
        return x, hm, vis, torch.tensor(float(self.labels[i]))


def train_fold(samples, labels, device, epochs, seed, lam):
    torch.manual_seed(seed)
    net = MTLNet(len(K.POINTS["hip"])).to(device)
    dl = torch.utils.data.DataLoader(MTLDataset(samples, labels, True, seed), batch_size=8, shuffle=True, drop_last=True,
                                     num_workers=4, persistent_workers=True, worker_init_fn=K.worker_init)
    y = np.asarray(labels, float)
    pw = torch.tensor(float((1 - y.mean()) / y.mean()), dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * len(dl), pct_start=0.1)
    for _ in range(epochs):
        net.train()
        for x, hm, vis, t in dl:
            x, hm, vis, t = x.to(device), hm.to(device), vis.to(device), t.to(device)
            hm_p, vis_p, logit = net(x)
            loss, _, _ = K.loss_fn(hm_p, vis_p, hm, vis)
            loss = loss + lam * F.binary_cross_entropy_with_logits(logit, t, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return net


@torch.no_grad()
def predict_fold(net, samples, device):
    net.eval()
    x = torch.from_numpy(np.stack([K.prepare(s.img)[0] for s in samples]))[:, None].to(device)
    return torch.sigmoid(net(x)[2]).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--lam", type=float, default=1.0)
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    idx = pd.read_csv("data/train/image_index.csv")
    idx["id"] = idx.n.map("{:03d}".format) + "_" + idx.region
    idx = idx.set_index("id")
    samples = [s for s in K.load_samples("hip") if not pd.isna(idx.loc[s.id, "v_posrot"])]
    ids = [s.id for s in samples]
    y = idx.loc[ids, "v_posrot"].astype(int).to_numpy(); g = np.array([s.n for s in samples])
    kp = json.loads(Path("data/train/keypoints.json").read_text())
    oof_kp = json.loads(Path("data/train/kp_oof_hip.json").read_text())
    imgs = {i: np.asarray(Image.open(f"data/annotation/images/{i}.png")) for i in ids}
    hog_tr = np.stack([hog(hip_crop(imgs[i], kp[i]["points"], i[4:])[0], 16) for i in ids])
    hog_va = np.stack([hog(hip_crop(imgs[i], oof_kp[i]["points"], i[4:])[0], 16) for i in ids])

    names = ("mtl", "rf", "ens")
    aucs = {k: [] for k in names}; pooled = {k: np.zeros(len(y)) for k in names}
    for r in range(args.repeats):
        oof = {k: np.zeros(len(y)) for k in names}
        for fold, (tr, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=200 + r).split(hog_tr, y, g)):
            t0 = time.time()
            net = train_fold([samples[i] for i in tr], y[tr], device, args.epochs, seed=r * 10 + fold, lam=args.lam)
            oof["mtl"][va] = predict_fold(net, [samples[i] for i in va], device)
            rf = RandomForestClassifier(500, min_samples_leaf=3, max_features=0.1, class_weight="balanced_subsample", random_state=0, n_jobs=4)
            rf.fit(hog_tr[tr], y[tr]); oof["rf"][va] = rf.predict_proba(hog_va[va])[:, 1]
            print(f"  repeat {r} fold {fold}: mtl fold AUC {roc_auc_score(y[va], oof['mtl'][va]) if len(set(y[va])) > 1 else float('nan'):.2f}  {time.time() - t0:.0f}s", flush=True)
        oof["ens"] = (pd.Series(oof["mtl"]).rank().to_numpy() + pd.Series(oof["rf"]).rank().to_numpy()) / 2
        for k in names:
            aucs[k].append(roc_auc_score(y, oof[k])); pooled[k] += oof[k] / args.repeats
    print(f"\nmulti-task rotation, {len(y)} hips ({y.sum()} violations), lambda={args.lam}, epochs={args.epochs}")
    out = {}
    for k in names:
        lo, hi = bootstrap_ci(y, pooled[k])
        out[k] = {"auc": round(float(np.mean(aucs[k])), 3), "sd": round(float(np.std(aucs[k])), 3), "ci95": [round(lo, 3), round(hi, 3)]}
        print(f"  {k:4s} AUC {np.mean(aucs[k]):.3f} ± {np.std(aucs[k]):.3f}  (95% CI {lo:.2f}–{hi:.2f})")
    Path(f"data/train/rotation_mtl_lam{args.lam}.json").write_text(json.dumps(out, indent=1))
    np.save(f"data/train/rotation_mtl_scores_lam{args.lam}.npy", np.stack([pooled[k] for k in names]))


if __name__ == "__main__":
    main()
