"""Spine pre-annotation proposals (blind to expert labels): column centre line + iliac crest tops."""
import json, numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d, label, median_filter
from skimage_free_otsu import otsu

def centre_line(a):
    h, w = a.shape
    P = np.stack([gaussian_filter1d(a[y], 6) for y in range(h)])
    lo, hi = 40, w - 40
    S = np.full((h, w), -1e9); S[0, lo:hi] = P[0, lo:hi]; back = np.zeros((h, w), int)
    for y in range(1, h):
        for dx in (-2, -1, 0, 1, 2):
            prev = np.roll(S[y-1], dx) - 3.0 * abs(dx)
            better = prev > S[y]
            S[y] = np.where(better, prev, S[y]); back[y] = np.where(better, -dx, back[y])
        S[y, :lo] = S[y, hi:] = -1e9; S[y, lo:hi] += P[y, lo:hi]
    x = np.zeros(h, int); x[-1] = int(np.argmax(S[-1]))
    for y in range(h - 1, 0, -1): x[y-1] = x[y] + back[y, x[y]]
    # centre = midpoint of the column edges around the ridge
    cx, hw = np.zeros(h), np.zeros(h)
    for y in range(h):
        p = P[y]; peak = p[x[y]]; base = np.percentile(p, 20); t = base + 0.5 * (peak - base)
        l = x[y]; r = x[y]
        while l > 0 and p[l] > t: l -= 1
        while r < w - 1 and p[r] > t: r += 1
        cx[y], hw[y] = (l + r) / 2, (r - l) / 2
    return median_filter(cx, 15, mode="nearest"), median_filter(hw, 15, mode="nearest")

def crests(a, cx, hw):
    h, w = a.shape; y0 = int(h * 0.55)
    reg = a[y0:]; t = max(otsu(reg), 70)
    m = reg > t
    for yy in range(m.shape[0]):
        y = yy + y0; m[yy, max(0, int(cx[y] - hw[y] - 28)):min(w, int(cx[y] + hw[y] + 28))] = False
    lab, n = label(m); out = {}
    for side, xs in (("crest_a", slice(0, w // 2)), ("crest_b", slice(w // 2, w))):
        best = None
        for i in range(1, n + 1):
            comp = lab == i; comp_side = np.zeros_like(comp); comp_side[:, xs] = comp[:, xs]
            area = comp_side.sum()
            if area < 150: continue
            ys, xs_ = np.where(comp_side)
            touches = ys.max() >= m.shape[0] - 2 or (xs_.min() <= 1 if side == "crest_a" else xs_.max() >= w - 2)
            if not touches: continue
            top = ys.min(); xt = xs_[ys == top].mean()
            if abs(xt - np.median(cx[int(h*.4):int(h*.7)])) < 75: continue   # sacrum / L5 processes, not a wing
            if best is None or area > best[2]: best = (float(xt), float(top + y0), area)
        out[side] = None if best is None else [round(best[0], 1), round(best[1], 1)]
    return out, t

def propose(path):
    a = gaussian_filter(np.asarray(Image.open(path), float), 1.5)
    h, w = a.shape; cx, hw = centre_line(a)
    cr, t = crests(a, cx, hw)
    ys = [p[1] for p in cr.values() if p]
    yb = int(round(np.mean(ys))) if ys else h - 3
    yb = min(max(yb, 0), h - 1)
    # The column centre right at the crest level is masked by L5 processes and the sacrum: take it just above.
    band = cx[max(yb - 40, 0):max(yb - 10, 1)]
    xb = float(np.median(band)) if len(band) else float(cx[yb])
    return {"col_top": [round(float(np.median(cx[2:12])), 1), 2.0], "col_bottom": [round(xb, 1), float(yb)], **cr}
