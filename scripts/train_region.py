"""Train the region classifier (models/region.pkl) and its novelty check.

Cross-validated by study, then refitted on everything. Also reports how the novelty score separates
the training scans from obviously foreign inputs (noise, blank frames, rotated scans), which is what
the service uses to refuse images that are not a DXA spine or proximal femur.

Usage: python scripts/train_region.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from dxaqc.region import RegionClassifier


def load():
    idx = pd.read_csv("data/train/image_index.csv")
    idx["id"] = idx.n.map("{:03d}".format) + "_" + idx.region
    imgs = [np.asarray(Image.open(f"data/annotation/images/{i}.png").convert("L")) for i in idx.id]
    return idx, imgs


def foreign(imgs, rng):
    """Inputs the service must refuse: noise, blank, rotated, and a strong-gradient 'photo'."""
    out = {}
    h, w = imgs[0].shape
    out["шум"] = rng.integers(0, 255, (h, w), dtype=np.uint8)
    out["пустой кадр"] = np.full((h, w), 40, np.uint8)
    out["градиент"] = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    for k, i in enumerate((0, 5, 17)):
        out[f"повёрнут на 90° #{k + 1}"] = np.ascontiguousarray(np.rot90(imgs[i]))
    return out


def main():
    idx, imgs = load()
    groups = idx.study.factorize()[0]
    y = idx.region.to_numpy()
    acc = []
    for seed in range(3):
        ok = 0
        for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups):
            m = RegionClassifier().fit([imgs[i] for i in tr], y[tr])
            ok += sum(m.predict(imgs[i])[0] == y[i] for i in va)
        acc.append(ok / len(y))
    print(f"область: точность по кросс-валидации {np.mean(acc):.4f} ({int(np.mean(acc) * len(y))}/{len(y)})")

    model = RegionClassifier().fit(imgs, y)
    nov = np.array([model.novelty(im) for im in imgs])
    print(f"новизна на обучающих снимках: медиана {np.median(nov):.2f}, p99 {np.quantile(nov, 0.99):.2f}, max {nov.max():.2f}")
    for name, im in foreign(imgs, np.random.default_rng(0)).items():
        print(f"  чужое «{name}»: новизна {model.novelty(im):.2f}, класс {model.predict(im)[0]}")
    Path("models").mkdir(exist_ok=True)
    model.save("models/region.pkl")
    print("saved models/region.pkl")


if __name__ == "__main__":
    main()
