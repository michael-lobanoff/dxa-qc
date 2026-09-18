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
    """kp_<region>.pt, plus kp_<region>_<n>.pt when present: an ensemble of detectors trained with different
    seeds and augmentations, whose heatmaps are averaged (kpmodel.predict). A single detector is one random
    draw — two draws with identical point error differed by 0.05 AUC on hip rotation downstream, because
    the rotation model reads crops placed by these points; the average of several draws is steadier."""

    def __init__(self, model_dir="models", device=None):
        self.device = device or best_device()
        self.models = {}
        for region in ("spine", "hip"):
            nets = []
            for f in sorted(Path(model_dir).glob(f"kp_{region}*.pt")):
                ckpt = torch.load(f, map_location="cpu", weights_only=False)
                assert ckpt["points"] == K.POINTS[region] and ckpt["size"] == K.SIZE and ckpt["margin"] == K.MARGIN
                net = K.KPNet(len(ckpt["points"]))
                net.load_state_dict(ckpt["state_dict"])
                nets.append(net.to(self.device).eval())
            if not nets:
                raise FileNotFoundError(f"no landmark model kp_{region}*.pt in {model_dir}")
            self.models[region] = nets

    def __call__(self, img: np.ndarray, region: str, vis_threshold=0.5, members=False):
        """img: uint8 array as stored in the DICOM; region: 'spine' | 'hip_left' | 'hip_right'.

        Returns {name: [x, y] or None} in original pixels and {name: in-frame confidence}; with
        members=True also a list with the same pair for every model of the ensemble.
        """
        kind = "spine" if region == "spine" else "hip"
        names = K.POINTS[kind]

        def as_dict(p, conf):
            return ({k: (None if np.isnan(q).any() else [float(q[0]), float(q[1])]) for k, q in zip(names, p)},
                    {k: float(c) for k, c in zip(names, conf)})

        res = K.predict(self.models[kind], [img], kind, self.device, flips=[region == "hip_left"],
                        vis_threshold=vis_threshold, members=members)
        if not members:
            return as_dict(*res[0])
        ens, each = res
        return (*as_dict(*ens[0]), [as_dict(*m[0]) for m in each])
