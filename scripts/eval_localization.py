"""Localisation metrics (ТЗ 8.4: «Dice, IoU, расстояние между ключевыми точками — при наличии
соответствующего функционала»).

Two kinds of localisation are shipped, and each is measured the way its annotation allows:

* landmarks — distance between the predicted and the annotated point, in millimetres, using
  out-of-fold predictions (the model never saw that patient);
* foreign bodies — the hand annotation marks a BOX around each object while the service marks the
  metal pixels themselves, so a pixel-wise Dice between them is meaningless (it comes out at 0.04 for
  a thin wire inside a generous box). Reported instead: how many annotated objects are hit, and how
  much of what we point at lies on an object.

Usage: python scripts/eval_localization.py
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from dxaqc.artifacts import artifact_mask


def landmark_errors(kp, oof, idx):
    rows = []
    for i, d in oof.items():
        mm = float(idx.loc[i, "mm_per_px"])
        for k, p in d["points"].items():
            q = kp[i]["points"].get(k)
            if p and q:
                rows.append({"point": k, "mm": float(np.hypot(p[0] - q[0], p[1] - q[1]) * mm)})
    e = pd.DataFrame(rows).groupby("point")["mm"].agg(["median", lambda s: s.quantile(0.9), "size"])
    e.columns = ["median_mm", "p90_mm", "n"]
    return e.sort_values("median_mm")


def artifact_localisation(kp, oof):
    hit_boxes = total_boxes = comp_in = comp_all = imgs = imgs_hit = 0
    for i, d in kp.items():
        if not i.endswith("spine") or not d.get("boxes"):
            continue
        img = np.asarray(Image.open(f"data/annotation/images/{i}.png").convert("L"))
        pred = artifact_mask(img, oof[i]["points"]).astype(np.uint8)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(pred, 8)
        boxes = [(int(a), int(b), int(c), int(e)) for a, b, c, e in d["boxes"]]
        imgs += 1
        any_hit = False
        for (x0, y0, x1, y1) in boxes:
            total_boxes += 1
            if pred[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1].any():
                hit_boxes += 1
                any_hit = True
        imgs_hit += any_hit
        for k in range(1, n):
            comp_all += 1
            cx, cy = cent[k]
            comp_in += any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in boxes)
    return {"images": imgs, "images_hit": imgs_hit, "objects": total_boxes, "objects_hit": hit_boxes,
            "components": comp_all, "components_on_object": comp_in}


def main():
    idx = pd.read_csv("data/train/image_index.csv")
    idx["id"] = idx.n.map("{:03d}".format) + "_" + idx.region
    idx = idx.set_index("id")
    kp = json.loads(Path("data/train/keypoints.json").read_text())
    oof = {**json.loads(Path("data/train/kp_oof_spine.json").read_text()),
           **json.loads(Path("data/train/kp_oof_hip.json").read_text())}

    e = landmark_errors(kp, oof, idx)
    print("Ключевые точки: расстояние до размеченной точки (вне обучения)")
    print(e.round(1).to_string())
    print(f"  средняя медиана по точкам: {e.median_mm.mean():.1f} мм\n")

    a = artifact_localisation(kp, oof)
    print("Инородные тела: локализация")
    print(f"  снимков с размеченным металлом: {a['images']}, объектов: {a['objects']}")
    print(f"  найден хотя бы один объект: {a['images_hit']}/{a['images']} ({a['images_hit'] / a['images']:.0%})")
    print(f"  покрыто объектов: {a['objects_hit']}/{a['objects']} ({a['objects_hit'] / a['objects']:.0%})")
    print(f"  найденное лежит на объекте: {a['components_on_object']}/{a['components']} "
          f"({a['components_on_object'] / max(a['components'], 1):.0%})")
    out = {"landmarks_mm": e.round(2).to_dict(orient="index"), "artifacts": a}
    Path("data/train/localization_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
