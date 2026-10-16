"""Отдельный канал «металл необычного вида»: крупнейший связный участок маски сети сегментации.

Вердикт по посторонним предметам остаётся прежним (прежняя сеть и решающий слой — их метрика считается
отдельно и ронять её нельзя). Канал добавляется сбоку и срабатывает, только когда сеть уверенно видит
сплошной металлический участок крупнее, чем на любом чистом снимке организаторов. Порог — максимум по
82 чистым позвоночникам с запасом 20 %: на обучающих данных ложных срабатываний нет по построению.

    python scripts/metal_channel.py
"""
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dxaqc.artifactnet import ArtifactSegmenter  # noqa: E402
from dxaqc.service import QCService, read_dicom_image  # noqa: E402
from dxaqc.synth import _stamp  # noqa: E402
from novel_objects import KINDS, body_point  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path("data/train")
MM = 0.607


def biggest_metal(seg, img):
    """Площадь крупнейшего участка, который сеть уверенно считает металлом, в пикселях снимка."""
    import torch
    from dxaqc import kpmodel as K
    h, w = img.shape
    m = K.letterbox_matrix(h, w)
    x = cv2.warpAffine(np.ascontiguousarray(img).astype(np.float32), m, (K.SIZE, K.SIZE), flags=cv2.INTER_LINEAR) / 255.0
    t = torch.from_numpy(x.astype(np.float32))[None, None].to(seg.device)
    with torch.no_grad():
        p = torch.stack([torch.sigmoid(net(t)[0])[0, 0] for net in seg.nets]).mean(0).cpu().numpy()
    p = cv2.warpAffine(cv2.resize(p, (K.SIZE, K.SIZE)), cv2.invertAffineTransform(m), (w, h))
    n, lab, st, _ = cv2.connectedComponentsWithStats((p > 0.5).astype(np.uint8), 8)
    return float(st[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0.0


def main():
    idx = pd.read_csv(ROOT / "image_index.csv")
    spines = idx[idx.region == "spine"]
    svc = QCService("models")
    nets = {"прежняя": svc.segmenter, "новая": ArtifactSegmenter("data/train/models_cand", svc.segmenter.device)}
    rows = []
    for _, r in spines.iterrows():
        img, _ = read_dicom_image(ROOT / r.path)
        base = {"label": int(r.v_artifact == 1)}
        rows.append({**base, "kind": "реальный снимок", **{k: biggest_metal(s, img) for k, s in nets.items()},
                     "verdict": "v_artifact" in svc.analyse(img, (MM, MM))["violations"]})
        if r.v_artifact != 0:
            continue
        rng = np.random.default_rng(int(r.n))
        for name, draw in KINDS.items():
            mask = np.zeros(img.shape, np.float32)
            draw(mask, body_point(img, rng), rng)
            if mask.sum() < 10:
                continue
            obj = _stamp(img, mask, rng)
            rows.append({**base, "kind": name, **{k: biggest_metal(s, obj) for k, s in nets.items()},
                         "verdict": "v_artifact" in svc.analyse(obj, (MM, MM))["violations"]})
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "metal_channel.csv", index=False)
    clean = d[(d.kind == "реальный снимок") & (d.label == 0)]
    thr = {k: 1.2 * clean[k].max() for k in nets}
    print("порог канала (1.2 × максимум на чистых):", {k: round(v, 1) for k, v in thr.items()})
    held = {"кольцо", "пуговица", "цепочка"}
    out = []
    for kind, g in d[d.kind != "реальный снимок"].groupby("kind", sort=False):
        out.append({"предмет": kind + (" (не было в обучении)" if kind in held else ""),
                    "вердикт сейчас, %": round(100 * g.verdict.mean()),
                    "канал, прежняя сеть, %": round(100 * (g["прежняя"] > thr["прежняя"]).mean()),
                    "канал, новая сеть, %": round(100 * (g["новая"] > thr["новая"]).mean()),
                    "вердикт или канал (новая), %": round(100 * (g.verdict | (g["новая"] > thr["новая"])).mean())})
    print(pd.DataFrame(out).to_string(index=False))
    real_pos = d[(d.kind == "реальный снимок") & (d.label == 1)]
    print(f"\nреальные посторонние предметы ({len(real_pos)}): канал новой сети срабатывает на "
          f"{int((real_pos['новая'] > thr['новая']).sum())}")


if __name__ == "__main__":
    main()
