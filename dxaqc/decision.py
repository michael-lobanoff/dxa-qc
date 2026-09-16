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
from .geometry import MM_PER_PX, spine_rules
from .vertebrae import find as find_vertebrae

TYPES = {
    "spine": {"v_axis": (["abs_tilt", "curvature"], [1, -1]), "v_pos": (["crest_conf"], [-1]),
              "v_artifact": (["area_top", "n_out", "max_len"], [1, 1, 1])},
    "hip": {"v_roi": (["margin_bottom", "margin_top"], [-1, -1]), "v_posrot": (["rotation"], [1])},
}
NAMES_RU = {"v_axis": "наклон оси позвоночника > допустимого", "v_pos": "некорректная укладка (не захвачены гребни подвздошных костей)",
            "v_artifact": "инородные тела / артефакты", "v_roi": "недостаточное поле вокруг области интереса",
            "v_posrot": "позиционирование / ротация бедра"}


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
            "vert_pitch_mm": None if vert is None else vert["pitch_px"] * mm_per_px,
            "vert_n": None if vert is None else vert["n"], "vert_points": None if vert is None else vert["points"],
            "crest_conf": min(conf.get("crest_a", 0.0), conf.get("crest_b", 0.0)),
            "area_top": a["area_top"], "n_out": a["n_out"], "max_len": a["max_len"]}


def hip_measurements(img, points, conf, side, mm_per_px=None):
    f = hip_features(points, img.shape, side, MM_PER_PX if mm_per_px is None else mm_per_px)
    return {"margin_bottom": f["margin_bottom"], "margin_top": f["margin_top"], "margin_lateral": f["margin_lateral"],
            "lt_conf": conf.get("lt", 0.0)}


def best_threshold(y, p):
    grid = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 49)))
    return max(grid, key=lambda t: f1_score(y, p >= t, zero_division=0))


class Monotone:
    """Score = weighted sum of standardised features with known direction (+1 = larger is worse),
    Platt-calibrated. With 6-17 positives an unconstrained model can learn the wrong sign; this cannot."""

    def __init__(self, signs, lr_weights=True):
        self.signs, self.lr_weights = np.asarray(signs, float), lr_weights

    def fit(self, X, y):
        Z = X * self.signs
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-9
        Z = (Z - self.mu) / self.sd
        if self.lr_weights and Z.shape[1] > 1:
            w = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(Z, y).coef_[0]
            self.w = np.clip(w, 0, None) if (w > 0).any() else np.ones(Z.shape[1])
        else:
            self.w = np.ones(Z.shape[1])
        self.cal = LogisticRegression(C=10.0, max_iter=2000).fit((Z @ self.w)[:, None], y)
        if self.cal.coef_[0, 0] <= 0:
            self.cal.coef_[0, 0] = 1e-3
        return self

    def predict_proba(self, X):
        s = ((X * self.signs - self.mu) / self.sd) @ self.w
        p = self.cal.predict_proba(s[:, None])[:, 1]
        return np.stack([1 - p, p], 1)


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
        score = 1 - np.prod([1 - probs[vt] for vt in probs], axis=0)
        y_img = np.max([labels[vt] for vt in labels], axis=0) if y_img is None else np.asarray(y_img)
        self.image_thresholds = {"screening": sensitivity_threshold(y_img, score, 0.80),
                                 "balanced": float(best_threshold(y_img, score))}
        return self

    def predict(self, meas):
        """meas: dict of measurements for one image -> (probabilities, decisions, image score)."""
        probs, decs = {}, {}
        for vt, (cols, _) in TYPES[self.region].items():
            x = np.array([[np.nan_to_num(float(meas.get(c) if meas.get(c) is not None else np.nan), nan=0.0) for c in cols]])
            probs[vt] = float(self.models[vt].predict_proba(x)[0, 1])
            decs[vt] = probs[vt] >= self.thresholds[vt]
        score = 1 - float(np.prod([1 - p for p in probs.values()]))
        return probs, decs, score

    def verdict(self, probs, decs, score, policy="screening"):
        """(quality_class, violation types) for one image.

        The class comes from the image threshold; the listed types are those over their own
        thresholds. When the image is flagged but no single type crosses its threshold, the most
        probable type is named — a technician needs to know what to re-check, and an empty
        "нарушение без типа" row would be useless.
        """
        t = self.image_thresholds.get(policy, self.image_thresholds["screening"])
        bad = score >= t
        types = [vt for vt, d in decs.items() if d] if bad else []
        if bad and not types and probs:
            types = [max(probs, key=probs.get)]
        return int(bad), types
