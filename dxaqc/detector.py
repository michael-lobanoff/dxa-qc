"""Inference wrapper around the trained landmark models (models/kp_spine.pt, models/kp_hip.pt)."""
import os
from pathlib import Path

import numpy as np
import torch

from . import kpmodel as K


def best_device():
    """DXAQC_DEVICE=cpu|cuda|mps overrides auto-detection (e.g. to measure the CPU-only configuration)."""
    forced = os.environ.get("DXAQC_DEVICE")
    if forced:
        return forced
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


class KeypointDetector:
    def __init__(self, model_dir="models", device=None):
        self.device = device or best_device()
        self.models = {}
        for region in ("spine", "hip"):
            ckpt = torch.load(Path(model_dir) / f"kp_{region}.pt", map_location="cpu", weights_only=False)
            assert ckpt["points"] == K.POINTS[region] and ckpt["size"] == K.SIZE and ckpt["margin"] == K.MARGIN
            net = K.KPNet(len(ckpt["points"]))
            net.load_state_dict(ckpt["state_dict"])
            self.models[region] = net.to(self.device).eval()

    def __call__(self, img: np.ndarray, region: str, vis_threshold=0.5):
        """img: uint8 array as stored in the DICOM; region: 'spine' | 'hip_left' | 'hip_right'.

        Returns {name: [x, y] or None} in original pixels and {name: in-frame confidence}.
        """
        kind = "spine" if region == "spine" else "hip"
        (p, conf), = K.predict(self.models[kind], [img], kind, self.device, flips=[region == "hip_left"],
                               vis_threshold=vis_threshold)
        names = K.POINTS[kind]
        points = {k: (None if np.isnan(q).any() else [float(q[0]), float(q[1])]) for k, q in zip(names, p)}
        return points, {k: float(c) for k, c in zip(names, conf)}
