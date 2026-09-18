"""Foreign-body segmentation on spine scans: real masks (bright thin pixels inside reviewed boxes) plus
synthetic metal drawn on clean scans. Image score = predicted metal area; evaluated against the expert
label with stratified CV and rank-ensembled with the hand-crafted top-hat cue.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from dxaqc import kpmodel as K
from dxaqc import synth
from dxaqc.artifacts import artifact_features
from rotation_experiments import bootstrap_ci


def real_mask(img, boxes):
    g = img.astype(np.float32)
    tophat = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    m = np.zeros(img.shape, bool)
    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = int(max(x1, 0)), int(max(y1, 0)), int(min(x2, img.shape[1] - 1)), int(min(y2, img.shape[0] - 1))
        sub = (tophat[y1:y2 + 1, x1:x2 + 1] > 0.10 * 255) & (g[y1:y2 + 1, x1:x2 + 1] > np.percentile(g, 80))
        m[y1:y2 + 1, x1:x2 + 1] |= sub
    return cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


class ArtDataset(torch.utils.data.Dataset):
    def __init__(self, items, train, seed=0):
        self.items, self.train, self.rng = items, train, np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it, r = self.items[i], self.rng
        img, mask = it["img"].copy(), (it["mask"].copy() if it["mask"] is not None else np.zeros(it["img"].shape, bool))
        h, w = img.shape
        if self.train:
            if r.random() < (0.6 if it["clean"] else 0.3):
                img, m = (synth.draw_underwire(img, r, it["spine_x"], return_mask=True) if r.random() < 0.6
                          else synth.draw_hardware(img, r, return_mask=True))
                mask |= m
            if r.random() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            aff = cv2.getRotationMatrix2D((K.SIZE / 2, K.SIZE / 2), r.uniform(-8, 8), r.uniform(0.92, 1.08))
            aff[:, 2] += r.uniform(-6, 6, 2)
            m_ = K.compose(aff.astype(np.float32), K.letterbox_matrix(h, w))
        else:
            m_ = K.letterbox_matrix(h, w)
        x = cv2.warpAffine(np.ascontiguousarray(img).astype(np.float32), m_, (K.SIZE, K.SIZE), flags=cv2.INTER_LINEAR) / 255.0
        t = cv2.warpAffine(np.ascontiguousarray(mask).astype(np.float32), m_, (K.SIZE, K.SIZE), flags=cv2.INTER_LINEAR) > 0.3
        t = t.reshape(K.SIZE // 2, 2, K.SIZE // 2, 2).any((1, 3))  # stride-2 target, max-pooled to keep thin wires
        if self.train:
            x = np.clip(np.clip(x, 0, 1) ** r.uniform(0.8, 1.25) * r.uniform(0.9, 1.1) + r.normal(0, 0.02, x.shape), 0, 1)
        return torch.from_numpy(x.astype(np.float32))[None], torch.from_numpy(t.astype(np.float32))[None]


def worker_init(_):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed % 2 ** 32)


def train(items, device, epochs, seed):
    torch.manual_seed(seed)
    net = K.KPNet(1).to(device)
    dl = torch.utils.data.DataLoader(ArtDataset(items, True, seed), batch_size=8, shuffle=True, drop_last=True,
                                     num_workers=4, persistent_workers=True, worker_init_fn=worker_init)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * len(dl), pct_start=0.1)
    pw = torch.tensor(10.0, device=device)
    for _ in range(epochs):
        net.train()
        for x, t in dl:
            x, t = x.to(device), t.to(device)
            logit = net(x)[0]
            p = torch.sigmoid(logit)
            dice = 1 - (2 * (p * t).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + t.sum((1, 2, 3)) + 1)
            loss = F.binary_cross_entropy_with_logits(logit, t, pos_weight=pw) + dice.mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return net


@torch.no_grad()
def scores(net, items, device):
    net.eval()
    ds = ArtDataset(items, False)
    x = torch.stack([ds[i][0] for i in range(len(items))]).to(device)
    p = torch.sigmoid(net(x)[0]).cpu().numpy()[:, 0]
    return (p > 0.5).sum((1, 2)).astype(float), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--save-final", action="store_true")
    ap.add_argument("--tag", default="", help="suffix of the output files, to compare variants without overwriting")
    args = ap.parse_args()
    sfx = f"_{args.tag}" if args.tag else ""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    kp = json.loads(Path("data/train/keypoints.json").read_text())
    oof = json.loads(Path("data/train/kp_oof_spine.json").read_text())
    idx = pd.read_csv("data/train/image_index.csv")
    idx["id"] = idx.n.map("{:03d}".format) + "_" + idx.region
    idx = idx.set_index("id")
    ids = sorted(i for i in kp if i.endswith("spine"))
    y = idx.loc[ids, "v_artifact"].astype(int).to_numpy()
    items = []
    for i, yi in zip(ids, y):
        img = np.asarray(Image.open(f"data/annotation/images/{i}.png"))
        boxes = kp[i]["boxes"]
        p = kp[i]["points"]
        sx = np.mean([p[k][0] for k in ("col_top", "col_bottom") if p.get(k)]) if p.get("col_top") else None
        items.append({"id": i, "img": img, "mask": real_mask(img, boxes) if boxes else None, "clean": not boxes and yi == 0,
                      "usable": bool(boxes) or yi == 0, "spine_x": sx})
    hand = np.array([artifact_features(it["img"], oof[it["id"]]["points"])["area_top"] for it in items])
    res = {"seg": [], "hand": [], "ens": []}; pooled = {k: np.zeros(len(y)) for k in res}
    for r in range(args.repeats):
        o = np.zeros(len(y))
        for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=400 + r).split(ids, y)):
            t0 = time.time()
            net = train([items[i] for i in tr if items[i]["usable"]], device, args.epochs, seed=r * 10 + fold)
            o[va] = scores(net, [items[i] for i in va], device)[0]
            print(f"  repeat {r} fold {fold}: {time.time() - t0:.0f}s", flush=True)
        pooled_oof = o if r == 0 else pooled_oof + o
        ens = (pd.Series(o).rank().to_numpy() + pd.Series(hand).rank().to_numpy()) / 2
        for k, s in (("seg", o), ("hand", hand), ("ens", ens)):
            res[k].append(roc_auc_score(y, s)); pooled[k] += s / args.repeats
    # per-image out-of-fold scores, so the decision layer can use the net as a feature
    Path(f"data/train/artifact_seg_oof{sfx}.json").write_text(json.dumps(
        {i: float(v) for i, v in zip(ids, pooled_oof / args.repeats)}, ensure_ascii=False, indent=1))
    print(f"saved data/train/artifact_seg_oof{sfx}.json")
    print(f"\nforeign bodies, {len(y)} spines ({y.sum()} with artifacts)")
    out = {}
    for k in res:
        lo, hi = bootstrap_ci(y, pooled[k])
        out[k] = {"auc": round(float(np.mean(res[k])), 3), "ci95": [round(lo, 3), round(hi, 3)]}
        print(f"  {k:4s} AUC {np.mean(res[k]):.3f}  (95% CI {lo:.2f}–{hi:.2f})")
    Path(f"data/train/artifact_seg_eval{sfx}.json").write_text(json.dumps(out, indent=1))
    if args.save_final:
        net = train([it for it in items if it["usable"]], device, args.epochs, seed=42)
        torch.save({"state_dict": net.state_dict(), "size": K.SIZE, "margin": K.MARGIN}, "models/artifact_seg.pt")
        print("saved models/artifact_seg.pt")


if __name__ == "__main__":
    main()
