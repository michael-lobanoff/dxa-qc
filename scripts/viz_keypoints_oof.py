"""Manual vs out-of-fold keypoints. Usage: python scripts/viz_keypoints_oof.py {spine,hip} docs/night/oof_<region>.png"""
import json, sys, numpy as np, cv2, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
region, out = sys.argv[1], sys.argv[2]
kp = json.load(open("data/train/keypoints.json"))
oof = json.load(open(f"data/train/kp_oof_{region}.json"))
COL = ["#4cc2ff", "#5ad17a", "#ffd84c", "#ff9f4c", "#ff6b6b", "#c77dff", "#4cffe0", "#ff4cc2", "#b0ff4c"]
def err(i):
    e = [np.hypot(*np.subtract(g, p)) for k, p in oof[i]["points"].items() if p and (g := kp[i]["points"].get(k))]
    return max(e) if e else 0
ids = sorted(oof, key=err, reverse=True)
pick = ids[:6] + ids[len(ids) // 2: len(ids) // 2 + 6]
fig, axs = plt.subplots(2, 6, figsize=(26, 10)); axs = axs.ravel()
for ax, i in zip(axs, pick):
    raw = np.asarray(Image.open(f"data/annotation/images/{i}.png")); h, w = raw.shape
    img = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4)).apply(raw) if region == "hip" else raw
    ax.imshow(img, cmap="gray", vmin=0, vmax=255); ax.axis("off")
    for j, (k, p) in enumerate(oof[i]["points"].items()):
        g = kp[i]["points"].get(k)
        if g: ax.plot(*g, "o", ms=9, mfc="none", mec=COL[j], mew=2)
        if p: ax.plot(*p, "o", ms=5, mfc=COL[j], mec="k")
        if g and p: ax.plot([g[0], p[0]], [g[1], p[1]], "-", color=COL[j], lw=1)
    ax.set_title(f"{i}  max err {err(i):.0f}px", fontsize=11)
fig.suptitle("○ ручная разметка   ● предсказание сети (кросс-валидация). Верхний ряд — худшие случаи, нижний — типичные", fontsize=13)
plt.tight_layout(); plt.savefig(out, dpi=55)
