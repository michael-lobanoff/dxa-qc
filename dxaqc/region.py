"""Which anatomy is on the image: spine, left hip or right hip.

Content-based (HOG of the downsized image + logistic regression) rather than image width, which
only identifies one device model. Left/right hips differ only by mirroring, so the classifier also
sees mirrored copies with swapped labels during training.
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


def features(img):
    return hog(cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA))


class RegionClassifier:
    def __init__(self):
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))

    def fit(self, images, labels):
        X, y = [], []
        for img, lab in zip(images, labels):
            X.append(features(img)); y.append(lab)
            X.append(features(np.ascontiguousarray(img[:, ::-1]))); y.append(MIRROR[lab])
        self.model.fit(np.stack(X), y)
        return self

    def predict(self, img):
        proba = self.model.predict_proba(features(img)[None])[0]
        k = int(np.argmax(proba))
        return self.model.classes_[k], float(proba[k])

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, path):
        obj = cls()
        with open(path, "rb") as f:
            obj.model = pickle.load(f)
        return obj
