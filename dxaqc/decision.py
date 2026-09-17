"""Per-violation decisions and the image verdict, shared by the evaluation script and the service.

Every violation type has a small monotone model over 1-3 interpretable measurements (or the
rotation classifier for hips), a Platt calibration and an F1-optimal threshold. The image is
"bad" if any type fires; its score is the noisy-OR of the per-type probabilities.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from .artifacts import artifact_features
from .features import hip_features, spine_curvature
from .geometry import MM_PER_PX, axes_mm, spine_rules
from .roi import hip_roi, spine_rois
from .vertebrae import find as find_vertebrae

# How the per-type scores are combined into the image score (quality_prob, the column ROC-AUC is
# computed from). Spine has three types with 6-17 positives each, where the probability scale is
# unstable, and percentiles rank better (AUC 0.859 against 0.845). Hip is dominated by rotation with
# 36 positives, where the probability itself carries information and percentiles destroy it (0.62).
FUSION = {"spine": "percentile", "hip": "probability"}

TYPES = {
    "spine": {"v_axis": (["abs_tilt", "curvature"], [1, -1]), "v_pos": (["crest_conf"], [-1]),
              # artifacts: one measurement beats three — the extra two are weaker (0.84, 0.80 against
              # 0.90) and with 17 positives they cost more than they add (per-fold AUC 0.869 -> 0.893)
              "v_artifact": (["area_top"], [1])},
    "hip": {"v_roi": (["margin_bottom", "margin_top"], [-1, -1]), "v_posrot": (["rotation"], [1])},
}
# Wording fixed by the organisers (разъяснения V2, вопрос 6): exactly these strings, several joined
# by "; ", empty when there is no violation. macro-F1 is computed over this list.
NAMES_RU = {"v_axis": "Не выравнена ось позвоночника", "v_pos": "Некорректная укладка",
            "v_artifact": "Присутствуют посторонние предметы", "v_roi": "Некорректная область интереса",
            "v_posrot": "Некорректная укладка"}


def spine_measurements(img, points, conf, mm_per_px=None):
    mm_per_px = MM_PER_PX if mm_per_px is None else mm_per_px
    vert = find_vertebrae(img, points, mm_per_px)
    if vert is not None:
        crests = [points[k][1] for k in ("crest_a", "crest_b") if points.get(k)]
        if crests:   # the crest line sits at ~L4/L5, so keep at most one vertebra below it (L5), not the sacrum
            vert["points"] = [p for p in vert["points"] if p[1] < min(crests) + 1.4 * vert["pitch_px"]]
    r = spine_rules(points, img.shape, mm_per_px, None if vert is None else vert["pitch_px"])
    a = artifact_features(img, points)
    return {"abs_tilt": abs(r["tilt_deg"] or 0.0), "tilt_deg": r["tilt_deg"],
            "curvature": spine_curvature(img, points, mm_per_px=mm_per_px) or 0.0,
            "span_mm": r["span_mm"], "span_vert": r["span_vert"], "top_coverage_ok": r["top_coverage_ok"],
            "vert_pitch_mm": None if vert is None else vert["pitch_px"] * axes_mm(mm_per_px)[1],
            "vert_n": None if vert is None else vert["n"], "vert_points": None if vert is None else vert["points"],
            "rois": spine_rois(img, points, vert),
            "crest_conf": min(conf.get("crest_a", 0.0), conf.get("crest_b", 0.0)),
            "area_top": a["area_top"], "n_out": a["n_out"], "max_len": a["max_len"],
            "printed_markup": a["n_overlay_lines"] >= 3}


def hip_measurements(img, points, conf, side, mm_per_px=None):
    f = hip_features(points, img.shape, side, MM_PER_PX if mm_per_px is None else mm_per_px)
    roi = hip_roi(points, img.shape)
    return {"margin_bottom": f["margin_bottom"], "margin_top": f["margin_top"], "margin_lateral": f["margin_lateral"],
            "shaft_angle": f.get("shaft_angle"), "lt_conf": conf.get("lt", 0.0),
            "rois": [roi] if roi else []}


def best_threshold(y, p):
    grid = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 49)))
    return max(grid, key=lambda t: f1_score(y, p >= t, zero_division=0))


class Monotone:
    """Score = weighted sum of standardised features with known direction (+1 = larger is worse),
    then calibrated. With 6-17 positives an unconstrained model can learn the wrong sign; this cannot.

    The Platt step is applied to the PERCENTILE of the score within the training set, not to the score
    itself. With so few positives a sigmoid fitted directly on the score varies a lot between folds, so
    probabilities from different folds end up on different scales — pooling them cost up to 0.12 AUC
    (v_pos 0.758 -> 0.878), and the noisy-OR across violation types inherits the same distortion.
    A percentile always lives on [0, 1], so the calibration is comparable whatever the fold.
    """

    def __init__(self, signs, lr_weights=True, percentile=True):
        self.signs, self.lr_weights, self.percentile = np.asarray(signs, float), lr_weights, percentile

    def fit(self, X, y):
        Z = X * self.signs
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-9
        Z = (Z - self.mu) / self.sd
        if self.lr_weights and Z.shape[1] > 1:
            w = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(Z, y).coef_[0]
            self.w = np.clip(w, 0, None) if (w > 0).any() else np.ones(Z.shape[1])
        else:
            self.w = np.ones(Z.shape[1])
        s = Z @ self.w
        self.ref = np.sort(s)
        self.cal = LogisticRegression(C=10.0, max_iter=2000).fit(self._scale(s)[:, None], y)
        if self.cal.coef_[0, 0] <= 0:
            self.cal.coef_[0, 0] = 1e-3
        return self

    def _scale(self, s):
        if not self.percentile:
            return s
        return np.searchsorted(self.ref, s) / max(len(self.ref), 1)

    def predict_proba(self, X):
        s = ((X * self.signs - self.mu) / self.sd) @ self.w
        p = self.cal.predict_proba(self._scale(s)[:, None])[:, 1]
        return np.stack([1 - p, p], 1)

    def rank_score(self, X):
        """Where this image falls in the training distribution of the score, in [0, 1]."""
        s = ((X * self.signs - self.mu) / self.sd) @ self.w
        return np.searchsorted(self.ref, s) / max(len(self.ref), 1)


def blend_rotation(own, other, alpha):
    """Rotation probability of a hip, mixed with the other hip of the same study.

    Both hips are positioned in one session, and the expert's rotation labels of left and right agree
    far more often than chance (r = 0.59; P(bad | other bad) = 0.67 against a base rate of 0.24).
    Mixing the two probabilities barely moves AUC (0.828 -> 0.831) but makes the score much steadier
    around the threshold, which is where it counts: F1 0.591 -> 0.657.
    """
    if other is None or not np.isfinite(other):
        return float(own)
    return float((1 - alpha) * own + alpha * other)


def sensitivity_threshold(y, p, target=0.80):
    """Most specific threshold that still reaches the target sensitivity.

    With 6-36 positives per type the F1-optimal threshold jumps between folds; a sensitivity target
    picks the same region of the curve far more stably (see scripts/threshold_policy.py).
    """
    grid = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97)))
    pos = max(int((y == 1).sum()), 1)
    ok = [t for t in grid if ((p >= t) & (y == 1)).sum() / pos >= target]
    return float(max(ok)) if ok else float(grid.min())


class DecisionModel:
    """Fitted per-type models + thresholds. feats: {type: X (n, k)} in the TYPES column order.

    The image verdict uses its own threshold on the noisy-OR score: "any type fired" inherits the
    instability of every per-type threshold at once, and measured out-of-fold it is both less
    sensitive and no better on F1 (0.63 vs 0.65).
    """

    def fit(self, region, feats, labels, y_img=None):
        self.region, self.models, self.thresholds = region, {}, {}
        probs = {}
        for vt, (cols, signs) in TYPES[region].items():
            m = Monotone(signs).fit(feats[vt], labels[vt])
            self.models[vt] = m
            probs[vt] = m.predict_proba(feats[vt])[:, 1]
            self.thresholds[vt] = float(best_threshold(labels[vt], probs[vt]))
        fuse = {vt: (self.models[vt].rank_score(feats[vt]) if FUSION[region] == "percentile" else probs[vt])
                for vt in probs}
        raw = 1 - np.prod([1 - fuse[vt] for vt in fuse], axis=0)
        y_for_cal = np.max([labels[vt] for vt in labels], axis=0) if y_img is None else np.asarray(y_img)
        self.image_cal = LogisticRegression(C=10.0, max_iter=2000).fit(raw[:, None], y_for_cal)
        if self.image_cal.coef_[0, 0] <= 0:
            self.image_cal.coef_[0, 0] = 1e-3
        self.scores_ = self.image_cal.predict_proba(raw[:, None])[:, 1]
        self.y_img_ = np.max([labels[vt] for vt in labels], axis=0) if y_img is None else np.asarray(y_img)
        self.set_image_thresholds(self.y_img_, self.scores_)
        return self

    def set_image_thresholds(self, y_img, score):
        """One operating point for the whole service. Measured out-of-fold, a threshold shared by spine
        and hip beats separate ones (F1 0.656 against 0.636): after the percentile calibration the
        probabilities are on the same scale, and one threshold is estimated from twice the data."""
        self.image_thresholds = {"screening": sensitivity_threshold(np.asarray(y_img), np.asarray(score), 0.80),
                                 "balanced": float(best_threshold(np.asarray(y_img), np.asarray(score)))}
        return self

    def predict(self, meas):
        """meas: dict of measurements for one image -> (probabilities, decisions, image score)."""
        probs, decs, fuse = {}, {}, {}
        for vt, (cols, _) in TYPES[self.region].items():
            x = np.array([[np.nan_to_num(float(meas.get(c) if meas.get(c) is not None else np.nan), nan=0.0) for c in cols]])
            probs[vt] = float(self.models[vt].predict_proba(x)[0, 1])
            decs[vt] = probs[vt] >= self.thresholds[vt]
            fuse[vt] = (float(self.models[vt].rank_score(x)[0])
                        if FUSION[self.region] == "percentile" and hasattr(self.models[vt], "rank_score")
                        else probs[vt])
        raw = 1 - float(np.prod([1 - p for p in fuse.values()]))
        return probs, decs, self.calibrate(raw)

    def calibrate(self, raw):
        """Map the combined score to a probability. One calibration per region, fitted on every
        labelled image, so quality_prob still reads as a probability after percentile fusion."""
        cal = getattr(self, "image_cal", None)
        return float(cal.predict_proba([[raw]])[0, 1]) if cal is not None else float(raw)

    def verdict(self, probs, decs, score, policy="screening"):
        """(quality_class, violation types) for one image: bad when any violation type fires.

        A threshold on the combined score was tried instead and shipped for a while. Once the
        per-type probabilities were put on a common scale (percentile calibration) and the rotation
        model improved, the plain union became better on every metric ТЗ 8.4 lists — per image
        F1 0.741 against 0.702, per study 0.800 against 0.752, macro-F1 over types 0.611 against
        0.588 — and it needs no "most probable type" hedge: every flagged image names its reason.
        The combined score is still reported as quality_prob, which is what ROC-AUC is computed from.
        """
        types = [vt for vt, d in decs.items() if d]
        return int(bool(types)), types
