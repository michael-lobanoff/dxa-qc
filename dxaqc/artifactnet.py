"""Segmentation of foreign bodies (models/artifact_seg.pt).

The same U-Net as the landmark models, with one output channel, trained on the boxes we annotated plus
synthetic metal drawn by synth.py. It contributes two things: the contour a doctor sees on the overlay,
and a second opinion for the decision layer — the hand-crafted top-hat feature alone scores 0.898 for
foreign bodies, the net alone 0.822, the two together 0.911 (matched folds, 17 positives).
"""
from pathlib import Path

import numpy as np
import torch

from . import kpmodel as K


class ArtifactSegmenter:
    def __init__(self, model_dir="models", device="cpu"):
        ckpt = torch.load(Path(model_dir) / "artifact_seg.pt", map_location="cpu", weights_only=False)
        net = K.KPNet(1)
        net.load_state_dict(ckpt["state_dict"])
        self.net = net.to(device).eval()
        self.device = device

    @torch.no_grad()
    def area(self, img) -> float:
        """Number of pixels the net calls metal (in its own stride-2 grid)."""
        import cv2
        h, w = img.shape[:2]
        m = K.letterbox_matrix(h, w)
        x = cv2.warpAffine(np.ascontiguousarray(img).astype(np.float32), m, (K.SIZE, K.SIZE),
                           flags=cv2.INTER_LINEAR) / 255.0
        t = torch.from_numpy(x.astype(np.float32))[None, None].to(self.device)
        p = torch.sigmoid(self.net(t)[0])[0, 0].cpu().numpy()
        return float((p > 0.5).sum())
