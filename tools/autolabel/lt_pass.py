import json, numpy as np, cv2, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d
IMG = "/Users/user/Desktop/lct/data/annotation/images/"
pts = json.load(open("hip_manual_pts.json"))
out = {}
ids = sorted(k for k, v in pts.items() if v.get("fn_c") and v.get("shaft_p") and v.get("shaft_d"))
for iid in ids:
    kp = pts[iid]; raw = np.asarray(Image.open(IMG + iid + ".png"), float); h, w = raw.shape
    left = iid.endswith("left")
    a = raw[:, ::-1] if left else raw                      # normalise: medial side to the image right
    X = lambda x: (w - 1 - x) if left else x
    g = gaussian_filter(a, 1.2)
    fn, sp, sd = kp["fn_c"], kp["shaft_p"], kp["shaft_d"]
    y0, y1 = int(fn[1] + 10), int(min(sp[1] + 25, h - 3))
    edge = []
    for y in range(y0, y1):
        t = (y - fn[1]) / max(sp[1] - fn[1], 1)
        cx = X(fn[0]) + (X(sp[0]) - X(fn[0])) * min(max(t, 0), 1) if y < sp[1] else X(sp[0]) + (X(sd[0]) - X(sp[0])) * (y - sp[1]) / max(sd[1] - sp[1], 1)
        cx = int(cx); row = g[y]; bone = np.percentile(row[max(cx - 10, 0):cx + 10], 50); bg = np.percentile(row, 5)
        thr = bg + 0.45 * (bone - bg); x = cx
        while x < w - 1 and row[x] > thr: x += 1
        edge.append((y, x))
    if len(edge) < 25: continue
    ys = np.array([e[0] for e in edge]); xs = gaussian_filter1d(np.array([e[1] for e in edge], float), 1.5)
    k = 9; prot = np.zeros_like(xs)
    for i in range(k, len(xs) - k): prot[i] = xs[i] - (xs[i - k] + xs[i + k]) / 2
    i = int(np.argmax(prot)); cand = [float(X(xs[i])), float(ys[i])]
    out[iid] = {"cand": cand, "prot": round(float(prot[i]), 2), "edge": [[float(X(x)), float(y)] for y, x in zip(ys, xs)],
                "box": [int(min(X(sp[0]) - 45, X(sp[0]) + 75) if left else X(sp[0]) - 45), int(fn[1] - 15), 0, int(y1 + 5)]}
json.dump(out, open("lt_cands.json", "w"))
def sheet(chunk, fname):
    fig, axs = plt.subplots(3, 4, figsize=(16, 13)); axs = axs.ravel()
    for ax in axs: ax.axis("off")
    for ax, iid in zip(axs, chunk):
        kp = pts[iid]; c = out[iid]; raw = np.asarray(Image.open(IMG + iid + ".png")); h, w = raw.shape
        img = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6)).apply(raw)
        sx = kp["shaft_p"][0]; left = iid.endswith("left")
        x0, x1 = (int(sx - 80), int(sx + 45)) if left else (int(sx - 45), int(sx + 80))
        y0, y1 = int(kp["fn_c"][1] - 15), int(min(kp["shaft_p"][1] + 35, h))
        x0, x1 = max(x0, 0), min(x1, w)
        ax.axis("on"); ax.imshow(img[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255, extent=(x0, x1, y1, y0))
        e = np.array(c["edge"]); ax.plot(e[:, 0], e[:, 1], "-", color="#ffd84c", lw=1, alpha=.8)
        ax.plot(*c["cand"], "o", ms=9, mfc="none", mec="#c77dff", mew=2)
        ax.set_xticks(range(x0 - x0 % 10, x1, 10)); ax.set_yticks(range(y0 - y0 % 10, y1, 10)); ax.tick_params(labelsize=6)
        ax.grid(color="#ff4040", lw=0.3, alpha=.5)
        ax.set_title(f"{iid}  prot={c['prot']}  cand=({c['cand'][0]:.0f},{c['cand'][1]:.0f})", fontsize=9)
    plt.tight_layout(); plt.savefig(fname, dpi=60); plt.close()
ids2 = sorted(out)
for i in range(0, len(ids2), 12): sheet(ids2[i:i+12], f"lt_{i//12:02d}.png")
print(len(out), "candidates;", (len(ids2) + 11) // 12, "sheets; prot quantiles:", np.percentile([v["prot"] for v in out.values()], [10, 50, 90]).round(2))
