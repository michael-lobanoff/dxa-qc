"""Error analysis for the pitch: per violation type, the most confident misses (expert: violation) and
false alarms (expert: normal) from the honest out-of-fold run (data/train/pipeline_oof.csv).

Usage: python scripts/error_gallery.py  ->  docs/night/errors_<type>.png
"""
import json

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dxaqc.decision import hip_measurements, spine_measurements  # noqa: E402
from dxaqc.visualize import render_overlay  # noqa: E402

TYPES = {"v_axis": "spine", "v_pos": "spine", "v_artifact": "spine", "v_roi": "hip", "v_posrot": "hip"}


def main():
    df = pd.read_csv("data/train/pipeline_oof.csv").set_index("id")
    oof = {**json.load(open("data/train/kp_oof_spine.json")), **json.load(open("data/train/kp_oof_hip.json"))}
    idx = pd.read_csv("data/train/image_index.csv")
    idx["id"] = idx.n.map("{:03d}".format) + "_" + idx.region
    idx = idx.set_index("id")
    for vt, region in TYPES.items():
        d = df[df[vt].notna()]
        fn = d[(d[vt] == 1) & (d[f"d_{vt}"] == 0)].sort_values(f"p_{vt}").head(4)
        fp = d[(d[vt] == 0) & (d[f"d_{vt}"] == 1)].sort_values(f"p_{vt}", ascending=False).head(4)
        fig, axs = plt.subplots(2, 4, figsize=(20, 11))
        for a in axs.ravel():
            a.axis("off")
        for row, (name, sub) in enumerate((("пропуск (эксперт: нарушение)", fn), ("ложное срабатывание (эксперт: норма)", fp))):
            for ax, (i, r) in zip(axs[row], sub.iterrows()):
                img = np.asarray(Image.open(f"data/annotation/images/{i}.png"))
                p, c = oof[i]["points"], oof[i]["conf"]
                mm = idx.loc[i, "mm_per_px"]
                meas = (spine_measurements(img, p, c, mm) if region == "spine"
                        else hip_measurements(img, p, c, i[4:], mm))
                viol = [t for t in TYPES if TYPES[t] == region and r.get(f"d_{t}") == 1]
                res = {"region": "spine" if region == "spine" else i[4:], "points": p, "measurements": meas,
                       "violations": viol, "score": r.score, "implant": False}
                comment = idx.loc[i, "comment"] if isinstance(idx.loc[i, "comment"], str) else ""
                ax.imshow(render_overlay(img, res))
                ax.set_title(f"{i}: {name}\np={r[f'p_{vt}']:.2f}  {comment}", fontsize=11)
        fig.suptitle(f"{vt}: верхний ряд — пропуски, нижний — ложные срабатывания (OOF)", fontsize=14)
        plt.tight_layout(); plt.savefig(f"docs/night/errors_{vt}.png", dpi=50); plt.close()


if __name__ == "__main__":
    main()
