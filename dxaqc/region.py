"""Which anatomy is on the image: spine, left hip or right hip — plus "is this our kind of image at all".

Content-based (HOG of the downsized image + logistic regression) rather than image width, which
only identifies one device model. Left/right hips differ only by mirroring, so the classifier also
sees mirrored copies with swapped labels during training.

A classifier always returns one of its three classes, confidently, even for a chest X-ray or a blank
frame, so it is paired with a novelty check: the distance in HOG space to the nearest training scans,
scaled by how far real scans sit from each other. (Reconstruction error from a PCA was tried first and
rejected: white noise has a flat, "average" HOG and reconstructs better than real anatomy.) The
organisers say foreign images should not appear in the closed test set, but a service that silently
reports a quality verdict for one would be worse than a service that says "this is not a DXA spine or
hip".
"""
import pickle

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .hog import hog

CLASSES = ["spine", "hip_left", "hip_right"]
MIRROR = {"spine": "spine", "hip_left": "hip_right", "hip_right": "hip_left"}
NEIGHBOURS = 5
# Refusal thresholds. The structure measure is the correlation at a 4 px lag after a light blur, which
# is what separates anatomy from noise without depending on how strongly an export was smoothed:
# organisers' scans 0.95-0.97, public DXA scans from another centre 0.74-0.93, white noise 0.00-0.44.
# Raw neighbouring-pixel correlation was tried first and rejected: it refused every image of the
# external DXA set (0.91 against our 0.997) purely because their export carries finer pixel noise.
MIN_STD = 1.0             # a blank frame carries no anatomy
AUTOCORR_LAG = 4
AUTOCORR_BLUR = 1.0
MIN_AUTOCORR = 0.60       # halfway between the noisiest real scan (0.74) and the smoothest noise (0.44)
MAX_NOVELTY = 1.05        # distance to the nearest training scans; 1.0 = the oddest scan we trained on


def features(img):
    return hog(cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA))


class RegionClassifier:
    def __init__(self):
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))
        self.bank = None            # training HOG features, for the novelty distance
        self.novelty_ref = None

    def fit(self, images, labels):
        X, y = [], []
        for img, lab in zip(images, labels):
            X.append(features(img)); y.append(lab)
            X.append(features(np.ascontiguousarray(img[:, ::-1]))); y.append(MIRROR[lab])
        X = np.stack(X)
        self.model.fit(X, y)
        self.bank = X.astype(np.float32)
        d = self._neighbour_distance(X, skip_self=True)
        self.novelty_ref = float(np.quantile(d, 0.99))     # 1.0 = as far out as the oddest training scan
        return self

    def _neighbour_distance(self, X, skip_self=False):
        """Mean distance to the NEIGHBOURS closest training scans."""
        d = np.linalg.norm(X[:, None, :] - self.bank[None, :, :], axis=2)
        d.sort(axis=1)
        start = 1 if skip_self else 0                      # a training row matches itself at distance 0
        return d[:, start:start + NEIGHBOURS].mean(1)

    def novelty(self, img):
        """How unlike the training scans this image is; 1.0 = the 99th percentile of training."""
        if self.bank is None or not self.novelty_ref:
            return 0.0
        return float(self._neighbour_distance(features(img)[None].astype(np.float32))[0] / self.novelty_ref)

    def check(self, img):
        """(ok, reason, stats) — is this a DXA spine / proximal femur scan at all?

        The service reports a refusal instead of a quality verdict: a confident verdict about a chest
        X-ray would be worse than an explicit "not evaluated".
        """
        a = np.asarray(img, dtype=np.float32)
        std = float(a.std())
        stats = {"std": std, "autocorr": 0.0, "novelty": 0.0}
        if std < MIN_STD:
            return False, "пустое изображение", stats
        b = cv2.GaussianBlur(a, (0, 0), AUTOCORR_BLUR)
        z = (b - b.mean()) / (b.std() + 1e-6)
        k = AUTOCORR_LAG
        stats["autocorr"] = float(((z[:, :-k] * z[:, k:]).mean() + (z[:-k] * z[k:]).mean()) / 2)
        stats["novelty"] = self.novelty(img)
        if stats["autocorr"] < MIN_AUTOCORR:
            return False, "изображение не похоже на рентгеновский снимок", stats
        if stats["novelty"] > MAX_NOVELTY:
            return False, "снимок не похож на денситометрию поясничного отдела или бедра", stats
        return True, "", stats

    def predict(self, img):
        proba = self.model.predict_proba(features(img)[None])[0]
        k = int(np.argmax(proba))
        return self.model.classes_[k], float(proba[k])

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "bank": self.bank, "novelty_ref": self.novelty_ref}, f)

    @classmethod
    def load(cls, path):
        obj = cls()
        with open(path, "rb") as f:
            d = pickle.load(f)
        if isinstance(d, dict):
            obj.model, obj.bank, obj.novelty_ref = d["model"], d["bank"], d["novelty_ref"]
        else:
            obj.model = d            # model files from before the novelty check
        return obj
