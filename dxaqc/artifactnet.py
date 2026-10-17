"""Segmentation of foreign bodies (models/artifact_seg.pt).

The same U-Net as the landmark models, with one output channel, trained on the boxes we annotated plus
synthetic metal drawn by synth.py. It contributes two things: the contour a doctor sees on the overlay,
and a second opinion for the decision layer — on matched folds (17 positives) the net alone scores 0.894
for foreign bodies, the hand-crafted top-hat feature 0.903, the two together 0.911
(data/train/artifact_seg_eval.json).
"""
from pathlib import Path

import numpy as np
import torch

from . import kpmodel as K


class ArtifactSegmenter:
    """artifact_seg.pt, plus artifact_seg_<n>.pt when present: nets trained with different seeds, whose
    probability maps are averaged."""

    def __init__(self, model_dir="models", device="cpu"):
        self.nets = []
        for f in sorted(Path(model_dir).glob("artifact_seg*.pt")):
            ckpt = torch.load(f, map_location="cpu", weights_only=False)
            net = K.KPNet(1)
            net.load_state_dict(ckpt["state_dict"])
            self.nets.append(net.to(device).eval())
        if not self.nets:
            raise FileNotFoundError(f"no artifact_seg*.pt in {model_dir}")
        self.device = device

    @torch.no_grad()
    def _prob(self, img):
        """Probability map in the net's own stride-2 grid, and the letterbox matrix that maps the image there."""
        import cv2
        h, w = img.shape[:2]
        m = K.letterbox_matrix(h, w)
        x = cv2.warpAffine(np.ascontiguousarray(img).astype(np.float32), m, (K.SIZE, K.SIZE),
                           flags=cv2.INTER_LINEAR) / 255.0
        t = torch.from_numpy(x.astype(np.float32))[None, None].to(self.device)
        p = torch.stack([torch.sigmoid(net(t)[0])[0, 0] for net in self.nets]).mean(0).cpu().numpy()
        return p, m

    def area(self, img) -> float:
        """Number of pixels the net calls metal (in its own stride-2 grid)."""
        return float((self._prob(img)[0] > 0.5).sum())

    def predict(self, img):
        """(area, mask): the area exactly as area() — the decision layer's feature — and the same answer
        as a boolean mask in image coordinates, for the contour the doctor sees on the overlay."""
        import cv2
        p, m = self._prob(img)
        h, w = img.shape[:2]
        full = cv2.resize(p, (K.SIZE, K.SIZE), interpolation=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(full, cv2.invertAffineTransform(m), (w, h), flags=cv2.INTER_LINEAR) > 0.5
        return float((p > 0.5).sum()), mask
