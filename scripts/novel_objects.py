"""Ловит ли сервис посторонние предметы, которых не было в обучении.

На чистые снимки позвоночника (эксперт не отметил посторонних предметов) кладём металл семи видов,
которых нет ни в разметке, ни в синтетике обучения: монета, кольцо, пуговица с отверстиями, пластина,
пятно неправильной формы, цепочка, молния. Металл рисуем так же, как синтетика обучения (_stamp):
яркость у верха шкалы, края чуть размыты. Место — любое в пределах тела, в том числе поверх позвонков.

Меряем две вещи: выносит ли сервис вердикт «посторонние предметы» и попадает ли маска сети на предмет.
Фон — те же снимки без предмета.

    python scripts/novel_objects.py
"""
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dxaqc import kpmodel as K  # noqa: E402
from dxaqc.service import QCService, read_dicom_image  # noqa: E402
from dxaqc.synth import _stamp  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path("data/train")
MM = 0.607


def body_point(img, rng):
    """Случайная точка внутри тела (не на чёрном фоне кадра)."""
    ys, xs = np.where(img > np.percentile(img, 35))
    k = rng.integers(len(xs))
    return float(xs[k]), float(ys[k])


def coin(m, c, rng):
    cv2.circle(m, (int(c[0]), int(c[1])), int(rng.uniform(9, 20) / MM / 2), 1.0, -1, cv2.LINE_AA)


def ring(m, c, rng):
    r = int(rng.uniform(16, 22) / MM / 2)
    cv2.circle(m, (int(c[0]), int(c[1])), r, 1.0, int(rng.uniform(2, 3.5)), cv2.LINE_AA)


def button(m, c, rng):
    r = int(rng.uniform(10, 16) / MM / 2)
    cv2.circle(m, (int(c[0]), int(c[1])), r, 1.0, -1, cv2.LINE_AA)
    for k in range(int(rng.choice([2, 4]))):
        a = 2 * np.pi * k / 4 + rng.uniform(0, 1)
        cv2.circle(m, (int(c[0] + 0.35 * r * np.cos(a)), int(c[1] + 0.35 * r * np.sin(a))), max(1, r // 5), 0.0, -1)


def plate(m, c, rng):
    w, h = rng.uniform(8, 18) / MM, rng.uniform(20, 45) / MM
    box = cv2.boxPoints(((c[0], c[1]), (w, h), rng.uniform(0, 180))).astype(np.int32)
    cv2.fillConvexPoly(m, box, 1.0, cv2.LINE_AA)


def blob(m, c, rng):
    n = int(rng.integers(5, 10))
    ang = np.sort(rng.uniform(0, 2 * np.pi, n))
    rad = rng.uniform(5, 14, n) / MM
    pts = np.c_[c[0] + rad * np.cos(ang), c[1] + rad * np.sin(ang)].astype(np.int32)
    cv2.fillPoly(m, [pts], 1.0, cv2.LINE_AA)


def chain(m, c, rng):
    a0, ln = rng.uniform(0, np.pi), rng.uniform(40, 90) / MM
    bend = rng.uniform(-0.6, 0.6)
    for t in np.linspace(-0.5, 0.5, int(ln / 4)):
        a = a0 + bend * t
        x, y = c[0] + t * ln * np.cos(a), c[1] + t * ln * np.sin(a) + bend * 30 * t * t
        cv2.circle(m, (int(x), int(y)), 1, 1.0, -1, cv2.LINE_AA)


def zipper(m, c, rng):
    ln = rng.uniform(40, 90) / MM
    a = rng.uniform(-0.3, 0.3) + np.pi / 2
    d = np.array([np.cos(a), np.sin(a)]); n = np.array([-d[1], d[0]])
    for t in np.arange(-ln / 2, ln / 2, 3):
        p = np.array(c) + t * d
        q1, q2 = p - 2.5 * n, p + 2.5 * n
        cv2.line(m, tuple(q1.astype(int)), tuple(q2.astype(int)), 1.0, 1, cv2.LINE_AA)


KINDS = {"монета": coin, "кольцо": ring, "пуговица": button, "пластина": plate,
         "пятно": blob, "цепочка": chain, "молния": zipper}


@torch.no_grad()
def seg_prob(svc, img):
    h, w = img.shape
    m = K.letterbox_matrix(h, w)
    x = cv2.warpAffine(np.ascontiguousarray(img).astype(np.float32), m, (K.SIZE, K.SIZE), flags=cv2.INTER_LINEAR) / 255.0
    t = torch.from_numpy(x.astype(np.float32))[None, None]
    seg = svc.segmenter
    p = torch.stack([torch.sigmoid(net(t.to(seg.device))[0])[0, 0] for net in seg.nets]).mean(0).cpu().numpy()
    p = cv2.resize(p, (K.SIZE, K.SIZE))                       # сеть отдаёт карту в половинном разрешении
    return cv2.warpAffine(p, cv2.invertAffineTransform(m), (w, h))


def main(model_dir="models", tag=""):
    idx = pd.read_csv(ROOT / "image_index.csv")
    clean = idx[(idx.region == "spine") & (idx.v_artifact == 0)]
    svc = QCService(model_dir)
    rng = np.random.default_rng(0)
    rows = []
    for _, r in clean.iterrows():
        img, _ = read_dicom_image(ROOT / r.path)
        base = svc.analyse(img, (MM, MM))
        rows.append({"kind": "без предмета", "flag": "v_artifact" in base["violations"], "hit": np.nan})
        for name, draw in KINDS.items():
            mask = np.zeros(img.shape, np.float32)
            draw(mask, body_point(img, rng), rng)
            if mask.sum() < 10:
                continue
            obj, om = _stamp(img, mask, rng, return_mask=True)
            res = svc.analyse(obj, (MM, MM))
            p = seg_prob(svc, obj)
            rows.append({"kind": name, "flag": "v_artifact" in res["violations"], "hit": float(p[om].max() > 0.5)})
    d = pd.DataFrame(rows)
    out = d.groupby("kind", sort=False).agg(n=("flag", "size"), verdict=("flag", "mean"), mask_hit=("hit", "mean"))
    out["verdict"] = (100 * out.verdict).round(0)
    out["mask_hit"] = (100 * out.mask_hit).round(0)
    print(out.rename(columns={"n": "снимков", "verdict": "вердикт «предмет», %", "mask_hit": "маска на предмете, %"}).to_string())
    d.to_csv(f"data/train/novel_objects{'_' + tag if tag else '_baseline'}.csv", index=False)


if __name__ == "__main__":
    main(*(sys.argv[1:3]))
