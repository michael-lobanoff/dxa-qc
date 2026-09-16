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


class DecisionModel:
    """Fitted per-type models + thresholds. feats: {type: X (n, k)} in the TYPES column order."""

    def fit(self, region, feats, labels):
        self.region, self.models, self.thresholds = region, {}, {}
        for vt, (cols, signs) in TYPES[region].items():
            m = Monotone(signs).fit(feats[vt], labels[vt])
            self.models[vt] = m
            self.thresholds[vt] = float(best_threshold(labels[vt], m.predict_proba(feats[vt])[:, 1]))
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
