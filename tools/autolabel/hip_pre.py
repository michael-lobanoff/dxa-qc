"""Hip pre-annotation proposals (blind to labels). Works on a normalised view: head on the image right."""
import numpy as np, cv2
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d, label, binary_opening, binary_fill_holes
from skimage_free_otsu import otsu

def runs(mask_row):
    r, s = [], None
    for i, v in enumerate(np.r_[mask_row, False]):
        if v and s is None: s = i
        if not v and s is not None: r.append((s, i - 1)); s = None
    return r

def propose(path, side):
    raw = np.asarray(Image.open(path), float)
    a = raw[:, ::-1] if side == "hip_left" else raw          # normalise: head on the right
    g = gaussian_filter(a, 1.5); h, w = g.shape
    t = otsu(g[g > 5]); bone = binary_opening(g > t, iterations=1)
    # --- shaft: track the femur run from the bottom row upwards
    ys = list(range(h - 4, int(h * 0.35), -1)); edges = {}
    prev = None
    for y in ys:
        rr = [(l, r) for l, r in runs(bone[y]) if r - l > 12]
        if not rr: break
        if prev is None:
            l, r = max(rr, key=lambda lr: lr[1] - lr[0] - abs((lr[0] + lr[1]) / 2 - w * 0.45) * 0.5)
        else:
            c = (prev[0] + prev[1]) / 2
            cand = [lr for lr in rr if lr[0] <= c <= lr[1]] or [min(rr, key=lambda lr: abs((lr[0] + lr[1]) / 2 - c))]
            l, r = cand[0]
        edges[y] = (l, r); prev = (l, r)
    yb = h - 4; shaft_d = [(edges[yb][0] + edges[yb][1]) / 2, float(yb)] if yb in edges else None
    base_w = np.median([edges[y][1] - edges[y][0] for y in list(edges)[:25]]) if edges else 40
    # medial edge profile (right edge) in the lower-middle part; stop where the run widens a lot (neck / pelvis)
    med = {y: r for y, (l, r) in edges.items() if r - l < base_w * 1.9}
    yy = np.array(sorted(med)); xm = np.array([med[y] for y in yy], float)
    lt, shaft_p = None, None
    if len(yy) > 30:
        xs = gaussian_filter1d(xm, 2)
        prot = np.zeros_like(xs)
        for i in range(12, len(xs) - 12): prot[i] = xs[i] - (xs[i - 12] + xs[i + 12]) / 2
        i = int(np.argmax(prot[: len(xs) // 1]))
        if prot[i] > 1.2: lt = [float(xs[i]), float(yy[i])]
        yp = (lt[1] + 22) if lt else yy[int(len(yy) * 0.35)]
        yp = int(min(max(yp, yy.min()), h - 6))
        if yp in edges: shaft_p = [(edges[yp][0] + edges[yp][1]) / 2, float(yp)]
    # lateral contour: leftmost bone pixel in the upper 70 %
    lab, n = label(bone); femur_lab = lab[h - 4, int(shaft_d[0])] if shaft_d else 0
    fem = lab == femur_lab if femur_lab else bone
    up = fem[: int(h * 0.7)]
    cols = np.where(up.any(0))[0]
    gt_lat = gt_top = None
    if len(cols):
        xl = cols.min(); yl = float(np.where(up[:, xl])[0].mean()); gt_lat = [float(xl), yl]
        band = fem[:, xl: xl + 45]; ytops = [np.where(band[:, j])[0].min() if band[:, j].any() else h for j in range(band.shape[1])]
        j = int(np.argmin(ytops)); gt_top = [float(xl + j), float(ytops[j])]
    # femoral head: Hough circle in the upper-right region
    fh_c = fh_top = None
    u8 = np.clip(g, 0, 255).astype(np.uint8)
    circ = cv2.HoughCircles(u8, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=60, param2=14, minRadius=18, maxRadius=40)
    if circ is not None:
        best = None
        for cx, cy, r in circ[0]:
            if not (w * 0.45 < cx < w * 0.97 and h * 0.08 < cy < h * 0.6): continue
            if gt_top and cx < gt_top[0] + 30: continue
            score = -abs(cy - (gt_top[1] if gt_top else h * 0.35)) * 0.5 - abs(r - 28)
            if best is None or score > best[0]: best = (score, float(cx), float(cy), float(r))
        if best:
            fh_c = [best[1], best[2]]; fh_top = [best[1], best[2] - best[3]]
    # neck centre: 45 % of the way from the head centre towards the intertrochanteric midpoint
    fn_c = None
    if fh_c and gt_top:
        tx = (gt_top[0] + (lt[0] if lt else gt_top[0] + 40)) / 2; ty = (gt_top[1] + (lt[1] if lt else gt_top[1] + 60)) / 2
        fn_c = [fh_c[0] + 0.45 * (tx - fh_c[0]), fh_c[1] + 0.45 * (ty - fh_c[1])]
    # ischium: lowest bone pixel medial to the femur, below the head
    isch = None
    if fh_c:
        m = bone.copy()
        for y, (l, r) in edges.items(): m[y, : r + 4] = False
        m[: int(fh_c[1] + 25)] = False; m[:, : int(fh_c[0] - 15)] = False
        lab2, n2 = label(m)
        if n2:
            sizes = np.bincount(lab2.ravel())[1:]; k = int(np.argmax(sizes)) + 1
            if sizes[k - 1] > 80:
                yy2, xx2 = np.where(lab2 == k); yl2 = yy2.max(); isch = [float(xx2[yy2 == yl2].mean()), float(yl2)]
    kp = {"fh_c": fh_c, "fh_top": fh_top, "fn_c": fn_c, "gt_top": gt_top, "gt_lat": gt_lat, "lt": lt,
          "isch": isch, "shaft_p": shaft_p, "shaft_d": shaft_d}
    if side == "hip_left":  # back to the original orientation
        kp = {k: (None if v is None else [round(w - 1 - v[0], 1), round(v[1], 1)]) for k, v in kp.items()}
    else:
        kp = {k: (None if v is None else [round(v[0], 1), round(v[1], 1)]) for k, v in kp.items()}
    return kp
