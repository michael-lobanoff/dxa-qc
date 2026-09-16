"""Landmark detector: a small U-Net that predicts one heatmap per keypoint plus an in-frame logit.

Hips are normalised to one orientation (left hips mirrored so the femoral head is on the
image right); spine and hips get separate models. Images are letterboxed into SIZE x SIZE,
heatmaps come out at stride 2, and predictions are mapped back to original pixels.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

POINTS = {
    "spine": ["col_top", "col_bottom", "crest_a", "crest_b"],
    "hip": ["fh_c", "fh_top", "fn_c", "gt_top", "gt_lat", "lt", "isch", "shaft_p", "shaft_d"],
}
SIZE, STRIDE, SIGMA = 256, 2, 2.0  # SIGMA in heatmap pixels
MARGIN = 12  # black border around the letterboxed scan: edge landmarks (col_top sits on the top edge) survive jitter


@dataclass
class Sample:
    id: str
    n: int
    img: np.ndarray        # uint8, normalised orientation
    pts: np.ndarray        # (K, 2) float32, NaN where the point is absent / out of frame
    flipped: bool          # True for left hips (mirrored on load)


def region_of(image_id: str) -> str:
    return "spine" if image_id.endswith("spine") else "hip"


def load_samples(region, kp_path="data/train/keypoints.json", img_dir="data/annotation/images"):
    kp = json.loads(Path(kp_path).read_text())
    names, out = POINTS[region], []
    for image_id, rec in sorted(kp.items()):
        if region_of(image_id) != region or rec.get("flags", {}).get("skip"):
            continue
        img = np.asarray(Image.open(Path(img_dir) / f"{image_id}.png"))
        pts = np.array([rec["points"].get(k) or [np.nan, np.nan] for k in names], np.float32)
        flipped = image_id.endswith("hip_left")
        if flipped:
            img = img[:, ::-1].copy()
            pts[:, 0] = img.shape[1] - 1 - pts[:, 0]
        out.append(Sample(image_id, int(image_id[:3]), img, pts, flipped))
    return out


# ---------------------------------------------------------------- geometry helpers
def letterbox_matrix(h, w, size=SIZE, margin=MARGIN):
    s = (size - 2 * margin) / max(h, w)
    return np.array([[s, 0, (size - w * s) / 2], [0, s, (size - h * s) / 2]], np.float32)


def compose(a, b):
    """Affine a after b (both 2x3)."""
    return (np.vstack([a, [0, 0, 1]]) @ np.vstack([b, [0, 0, 1]]))[:2].astype(np.float32)


def apply(m, pts):
    return pts @ m[:, :2].T + m[:, 2]


def invert(m):
    return cv2.invertAffineTransform(m)


def _axis_at(top, bottom, y):
    t = (y - top[1]) / max(bottom[1] - top[1], 1e-6)
    return np.array([top[0] + (bottom[0] - top[0]) * t, y], np.float32)


# ---------------------------------------------------------------- augmentation
def augment(s: Sample, region: str, rng: np.random.Generator):
    """Return (image, points) with field crops, flips, affine jitter and intensity changes."""
    img, pts = s.img.astype(np.float32), s.pts.copy()
    h, w = img.shape
    # Field crops: the scan started lower / stopped higher. Real exports vary in height, so this is realistic.
    top = int(rng.uniform(0, 0.18) * h) if rng.random() < 0.3 else 0
    bot = int(rng.uniform(0, 0.18) * h) if rng.random() < 0.3 else 0
    if top or bot:
        new_h = h - top - bot
        if region == "spine" and not np.isnan(pts[0]).any() and not np.isnan(pts[1]).any():
            ct, cb = pts[0].copy(), pts[1].copy()
            crests_left = [k for k in (2, 3) if not np.isnan(pts[k, 1]) and top <= pts[k, 1] < h - bot]
            # Annotation rule: col_top sits on the top edge; col_bottom on the crest level, or the bottom edge without crests.
            if ct[1] < top:
                pts[0] = _axis_at(ct, cb, top)
            if cb[1] >= h - bot or not crests_left:
                pts[1] = _axis_at(ct, cb, h - bot - 1)
        if region == "hip" and bot and not np.isnan(pts[7:9]).any() and pts[8, 1] >= h - bot > pts[7, 1]:
            pts[8] = _axis_at(pts[7], pts[8], h - bot - 1)  # shaft_d is defined on the bottom edge
        img = img[top:h - bot]
        pts[:, 1] -= top
        h = new_h
    if region == "spine" and rng.random() < 0.5:  # mirror: left/right crests swap
        img = img[:, ::-1]
        pts[:, 0] = w - 1 - pts[:, 0]
        pts[[2, 3]] = pts[[3, 2]]
    lb = letterbox_matrix(h, w)
    ang = rng.uniform(-12, 12) if region == "spine" else rng.uniform(-10, 10)
    aff = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), ang, rng.uniform(0.92, 1.08))
    aff[:, 2] += rng.uniform(-6, 6, 2)
    m = compose(aff.astype(np.float32), lb)
    out = cv2.warpAffine(np.ascontiguousarray(img), m, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderValue=0)
    p = apply(m, pts)
    # intensity: gamma, contrast/brightness, noise (exports differ in windowing)
    out = np.clip(out / 255.0, 0, 1) ** rng.uniform(0.7, 1.4)
    out = out * rng.uniform(0.8, 1.2) + rng.uniform(-0.08, 0.08)
    out = out + rng.normal(0, rng.uniform(0, 0.03), out.shape)
    return np.clip(out, 0, 1).astype(np.float32), p


def prepare(img: np.ndarray):
    """Deterministic letterbox for inference. Returns (image in [0,1], matrix original->model)."""
    h, w = img.shape
    m = letterbox_matrix(h, w)
    out = cv2.warpAffine(np.ascontiguousarray(img.astype(np.float32)), m, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderValue=0)
    return (out / 255.0).astype(np.float32), m


def targets(p: np.ndarray):
    """Gaussian heatmaps at stride 2 and in-frame flags for points p (K, 2) in model space."""
    hs = SIZE // STRIDE
    g = np.arange(hs, dtype=np.float32) * STRIDE + (STRIDE - 1) / 2
    vis = ~np.isnan(p).any(1) & (p[:, 0] >= 0) & (p[:, 0] < SIZE) & (p[:, 1] >= 0) & (p[:, 1] < SIZE)
    hm = np.zeros((len(p), hs, hs), np.float32)
    s2 = 2 * (SIGMA * STRIDE) ** 2
    for k in np.where(vis)[0]:
        hm[k] = np.exp(-((g[None, :] - p[k, 0]) ** 2 + (g[:, None] - p[k, 1]) ** 2) / s2)
    return hm, vis.astype(np.float32)


def worker_init(worker_id):
    """DataLoader workers start from copies of one generator: give each its own augmentation stream."""
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed % 2 ** 32)


class KPDataset(torch.utils.data.Dataset):
    def __init__(self, samples, region, train, seed=0):
        self.s, self.region, self.train = samples, region, train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        s = self.s[i]
        if self.train:
            img, p = augment(s, self.region, self.rng)
        else:
            img, m = prepare(s.img)
            p = apply(m, s.pts)
        hm, vis = targets(p)
        return torch.from_numpy(img)[None], torch.from_numpy(hm), torch.from_numpy(vis)


# ---------------------------------------------------------------- model
def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class KPNet(nn.Module):
    """U-Net, input 1x256x256 (+2 coordinate channels), heatmaps at 128x128, in-frame logits from the bottleneck."""

    def __init__(self, k, ch=(32, 64, 128, 256, 256)):
        super().__init__()
        self.enc = nn.ModuleList([_block(3, ch[0])] + [_block(ch[i], ch[i + 1]) for i in range(4)])
        self.up = nn.ModuleList([_block(ch[i + 1] + ch[i], ch[i]) for i in (3, 2, 1)])
        self.hm = nn.Conv2d(ch[1], k, 1)
        self.vis = nn.Linear(ch[4], k)

    def forward(self, x):
        b, _, h, w = x.shape
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device), torch.linspace(-1, 1, w, device=x.device), indexing="ij")
        x = torch.cat([x, xx.expand(b, 1, h, w), yy.expand(b, 1, h, w)], 1)
        feats = []
        for i, blk in enumerate(self.enc):
            x = blk(x if i == 0 else F.max_pool2d(x, 2))
            feats.append(x)
        vis = self.vis(feats[-1].mean((2, 3)))
        x = feats[-1]
        for blk, skip in zip(self.up, (feats[3], feats[2], feats[1])):
            x = blk(torch.cat([F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip], 1))
        return self.hm(x), vis


def loss_fn(hm_pred, vis_pred, hm, vis):
    per_map = ((hm_pred - hm) ** 2).mean((2, 3))                       # (B, K)
    l_hm = (per_map * vis).sum() / vis.sum().clamp(min=1) * 100
    l_vis = F.binary_cross_entropy_with_logits(vis_pred, vis)
    return l_hm + 0.5 * l_vis, l_hm.item(), l_vis.item()


def decode(hm: torch.Tensor):
    """Sub-pixel peak per heatmap: argmax, then a 5x5 weighted centroid. Returns (B, K, 2) in model space."""
    b, k, hs, ws = hm.shape
    flat = hm.reshape(b, k, -1).argmax(-1)
    iy, ix = (flat // ws).float(), (flat % ws).float()
    pad = F.pad(hm.clamp(min=0), (2, 2, 2, 2))
    out = torch.zeros(b, k, 2, device=hm.device)
    offs = torch.arange(-2, 3, device=hm.device).float()
    for bi in range(b):
        for ki in range(k):
            y, x = int(iy[bi, ki]), int(ix[bi, ki])
            win = pad[bi, ki, y:y + 5, x:x + 5]
            s = win.sum().clamp(min=1e-6)
            out[bi, ki, 0] = x + (win.sum(0) * offs).sum() / s
            out[bi, ki, 1] = y + (win.sum(1) * offs).sum() / s
    return out * STRIDE + (STRIDE - 1) / 2


@torch.no_grad()
def predict(model, images, region, device, flips=None, vis_threshold=0.5):
    """Keypoints in original pixel coordinates for a list of uint8 images (already in normalised orientation
    unless flips[i] is True, in which case the image is a left hip that gets mirrored here and un-mirrored after)."""
    model.eval()
    flips = flips or [False] * len(images)
    batch, mats, shapes = [], [], []
    for img, fl in zip(images, flips):
        im = img[:, ::-1].copy() if fl else img
        x, m = prepare(im)
        batch.append(x); mats.append(m); shapes.append(im.shape)
    x = torch.from_numpy(np.stack(batch))[:, None].to(device)
    hm, vis = model(x)
    if region == "spine":  # test-time mirror: average with the flipped prediction (crests swapped back)
        hm_f, vis_f = model(torch.flip(x, dims=[3]))
        hm = (hm + torch.flip(hm_f, dims=[3])[:, [0, 1, 3, 2]]) / 2
        vis = (vis + vis_f[:, [0, 1, 3, 2]]) / 2
    pts = decode(hm).cpu().numpy()
    conf = torch.sigmoid(vis).cpu().numpy()
    out = []
    for i, (m, (h, w), fl) in enumerate(zip(mats, shapes, flips)):
        p = apply(invert(m), pts[i])
        if fl:
            p[:, 0] = w - 1 - p[:, 0]
        p[conf[i] < vis_threshold] = np.nan
        out.append((p, conf[i]))
    return out
