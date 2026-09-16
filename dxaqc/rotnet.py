"""Small CNN for hip positioning / rotation on pose-normalised proximal-femur crops (see hipcrop)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hipcrop import CROP, hip_crop


class RotNet(nn.Module):
    def __init__(self, width=(16, 32, 64, 96), dropout=0.5):
        super().__init__()
        layers, cin = [], 1
        for c in width:
            layers += [nn.Conv2d(cin, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                       nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            cin = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(cin, 1))

    def forward(self, x):
        return self.head(self.features(x).mean((2, 3))).squeeze(1)


class CropDataset(torch.utils.data.Dataset):
    """items: list of (image, points, side, label). Training crops get small pose jitter and intensity changes;
    orientation is never flipped (medial/lateral matters)."""

    def __init__(self, items, train, seed=0):
        self.items, self.train = items, train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img, pts, side, label = self.items[i]
        if self.train:
            r = self.rng
            crop, _ = hip_crop(img, pts, side, jitter=(r.uniform(-4, 4), r.uniform(-4, 4), r.uniform(-5, 5), r.uniform(0.95, 1.05)))
            x = (crop / 255.0) ** r.uniform(0.8, 1.25) * r.uniform(0.9, 1.1)
            x = x + r.normal(0, 0.02, x.shape)
        else:
            crop, _ = hip_crop(img, pts, side)
            x = crop / 255.0
        return torch.from_numpy(np.clip(x, 0, 1).astype(np.float32))[None], torch.tensor(float(label))


def worker_init(worker_id):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed % 2 ** 32)


def train_rotnet(items, device, epochs=60, seed=0, pos_weight=None):
    torch.manual_seed(seed)
    model = RotNet().to(device)
    dl = torch.utils.data.DataLoader(CropDataset(items, True, seed), batch_size=16, shuffle=True, drop_last=True,
                                     num_workers=2, persistent_workers=True, worker_init_fn=worker_init)
    y = np.array([it[3] for it in items])
    pw = torch.tensor(float(pos_weight if pos_weight is not None else (1 - y.mean()) / max(y.mean(), 1e-3)), dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * len(dl), pct_start=0.15)
    for _ in range(epochs):
        model.train()
        for x, t in dl:
            x, t = x.to(device), t.to(device)
            loss = F.binary_cross_entropy_with_logits(model(x), t, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model


@torch.no_grad()
def predict_rotnet(model, items, device):
    model.eval()
    xs = torch.stack([CropDataset(items, False)[i][0] for i in range(len(items))]).to(device)
    return torch.sigmoid(model(xs)).cpu().numpy()
