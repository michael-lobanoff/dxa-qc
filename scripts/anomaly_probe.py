"""Посторонние предметы как аномалия: насколько фрагмент снимка не похож на нормальные фрагменты.

Признаки фрагментов берём из кодировщика нашего детектора ключевых точек — он обучен на DXA и получает
на вход координатные каналы, поэтому знает, где на снимке находится фрагмент. Банк нормальных
фрагментов собирается с чистых снимков обучающей части разбиения по исследованиям; оценка фрагмента —
расстояние до ближайшего нормального. Оценка снимка — максимум по фрагментам.

    python scripts/anomaly_probe.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dxaqc import kpmodel as K  # noqa: E402
from dxaqc.service import read_dicom_image  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path("data/train")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def load_net(region):
    ckpt = torch.load(f"models/kp_{region}.pt", map_location="cpu", weights_only=False)
    net = K.KPNet(len(ckpt["points"]))
    net.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt["model"])
    return net.eval().to(DEV)


@torch.no_grad()
def patch_features(net, img):
    """(64*64, D) признаки фрагментов и маска «внутри снимка» на той же сетке."""
    x, m = K.prepare(img)
    t = torch.from_numpy(x)[None, None].to(DEV)
    b, _, h, w = t.shape
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=DEV), torch.linspace(-1, 1, w, device=DEV), indexing="ij")
    z = torch.cat([t, xx.expand(b, 1, h, w), yy.expand(b, 1, h, w)], 1)
    feats = []
    for i, blk in enumerate(net.enc[:3]):
        z = blk(z if i == 0 else F.max_pool2d(z, 2))
        feats.append(z)
    f1 = F.avg_pool2d(feats[1], 2)                      # 128 -> 64
    f2 = feats[2]                                       # 64
    f = torch.cat([f1, f2], 1)
    f = F.avg_pool2d(f, 3, 1, 1)                        # соседство 3x3, как в PatchCore
    f = f[0].permute(1, 2, 0).reshape(-1, f.shape[1]).float().cpu().numpy()
    valid = F.avg_pool2d(torch.from_numpy((x > 0.02).astype(np.float32))[None, None], 4)[0, 0].numpy().reshape(-1) > 0.5
    return f, valid


def main(region="spine"):
    idx = pd.read_csv(ROOT / "image_index.csv")
    idx = idx[idx.region == "spine"] if region == "spine" else idx[idx.region.str.startswith("hip")]
    idx = idx[idx.v_artifact.notna()] if region == "spine" else idx
    net = load_net(region)
    feats, valids = [], []
    for _, r in idx.iterrows():
        img, _ = read_dicom_image(ROOT / r.path)       # то же чтение и нормализация, что в сервисе
        f, v = patch_features(net, img)
        feats.append(f)
        valids.append(v)
    y = idx.v_artifact.fillna(0).astype(int).values
    groups = idx.study.values
    scores = np.zeros(len(idx))
    rng = np.random.default_rng(0)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y, groups):
        clean = [i for i in tr if y[i] == 0]
        bank = np.concatenate([feats[i][valids[i]] for i in clean])
        bank = bank[rng.choice(len(bank), min(len(bank), 40000), replace=False)]
        bank_t = torch.from_numpy(bank).to(DEV)
        for i in te:
            q = torch.from_numpy(feats[i][valids[i]]).to(DEV)
            d = torch.cdist(q, bank_t).min(1).values.cpu().numpy()
            scores[i] = np.percentile(d, 99.9)          # «самый странный» фрагмент, но не одиночный выброс
    auc = roc_auc_score(y, scores)
    print(f"{region}: снимков {len(y)}, с посторонними предметами {y.sum()}, AUC аномальности {auc:.3f}")
    np.save(f"data/train/anomaly_scores_{region}.npy", scores)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "spine")
