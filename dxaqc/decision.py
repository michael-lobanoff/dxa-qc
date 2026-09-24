"""Per-violation decisions and the image verdict, shared by the evaluation script and the service.

Every violation type has a small monotone model over 1-3 interpretable measurements (or the
rotation classifier for hips), a Platt calibration and an F1-optimal threshold. The image is
"bad" if any type fires; its score is the noisy-OR of the per-type probabilities.
"""
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from .artifacts import artifact_features, overlay_mask
from .features import hip_features, pelvis_ratio, pelvis_relative_tilt, spine_curvature
from .geometry import MM_PER_PX, axes_mm, spine_rules
from .markup import read as read_markup, review as review_markup
from .roi import hip_roi, spine_rois
from .vertebrae import find as find_vertebrae

# How the per-type scores are combined into the image score (quality_prob, the column ROC-AUC is
# computed from). Spine has three types with 6-17 positives each, where the probability scale is
# unstable, and percentiles rank better (AUC 0.859 against 0.845). Hip is dominated by rotation with
# 36 positives, where the probability itself carries information and percentiles destroy it (0.62).
FUSION = {"spine": "percentile", "hip": "probability"}

TYPES = {
    "spine": {"v_axis": (["abs_tilt", "tilt_pelvis", "curvature"], [1, -1, -1]), "v_pos": (["crest_conf", "pelvis"], [-1, -1]),
              # artifacts: the hand-crafted top-hat area plus the segmentation net. Two weaker
              # hand features (n_out, max_len) were dropped — they cost more than they added; the net
              # is a genuinely different opinion and does add (0.898 -> 0.911 on matched folds).
              "v_artifact": (["area_top", "seg_area"], [1, 1])},
    "hip": {"v_roi": (["margin_bottom", "margin_top"], [-1, -1]), "v_posrot": (["rotation"], [1])},
}
# Types whose feature weights also learn from synthetic violations (scripts/synth_decision.py): the two
# rarest geometric ones, 10 and 7 positives. With so few, the weights of the two features jump from fold
# to fold; synthetic scans labelled by the ТЗ rule anchor them. Re-confirmed on 20 fresh CV repeats:
# v_axis AUC 0.871 -> 0.882, F1 0.357 -> 0.398; v_roi AUC 0.874 -> 0.892, F1 0.537 -> 0.557. Flat for
# weights 0.05-0.2; at 1.0 the rule starts to override the experts (whose tilt positives start at 3°).
SYNTH_TYPES = ("v_axis", "v_roi")
SYNTH_WEIGHT = 0.1
# Wording fixed by the organisers (разъяснения V2, вопрос 6): exactly these strings, several joined
# by "; ", empty when there is no violation. macro-F1 is computed over this list.
NAMES_RU = {"v_axis": "Не выравнена ось позвоночника", "v_pos": "Некорректная укладка",
            "v_artifact": "Присутствуют посторонние предметы", "v_roi": "Некорректная область интереса",
            "v_posrot": "Некорректная укладка"}


def _or(value, fallback):
    return fallback if value is None else value


def spine_measurements(img, points, conf, mm_per_px=None, seg_area=0.0):
    """seg_area: pixels the segmentation net calls metal (artifactnet.ArtifactSegmenter)."""
    mm_per_px = MM_PER_PX if mm_per_px is None else mm_per_px
    vert = find_vertebrae(img, points, mm_per_px)
    if vert is not None:
        crests = [points[k][1] for k in ("crest_a", "crest_b") if points.get(k)]
        if crests:   # the crest line sits at ~L4/L5, so keep at most one vertebra below it (L5), not the sacrum
            vert["points"] = [p for p in vert["points"] if p[1] < min(crests) + 1.4 * vert["pitch_px"]]
    r = spine_rules(points, img.shape, mm_per_px, None if vert is None else vert["pitch_px"])
    a = artifact_features(img, points)
    rois = spine_rois(img, points, vert)
    # optional (ТЗ 2.6): when the densitometer printed its own L1-L4 boxes on the image, read them and
    # judge them. Only runs on such images, so it costs nothing on the organisers' scans, which have none.
    printed = a["n_overlay_lines"] >= 3
    mk = read_markup(img) if printed else None
    return {"abs_tilt": abs(r["tilt_deg"] or 0.0), "tilt_deg": r["tilt_deg"],
            "curvature": spine_curvature(img, points, mm_per_px=mm_per_px) or 0.0,
            "span_mm": r["span_mm"], "span_vert": r["span_vert"], "top_coverage_ok": r["top_coverage_ok"],
            "vert_pitch_mm": None if vert is None else vert["pitch_px"] * axes_mm(mm_per_px)[1],
            "vert_n": None if vert is None else vert["n"], "vert_points": None if vert is None else vert["points"],
            "rois": rois, "markup": mk,
            "markup_review": review_markup(img, mk, mm_per_px) if mk else None,
            "crest_conf": min(conf.get("crest_a", 0.0), conf.get("crest_b", 0.0)),
            "pelvis": _or(pelvis_ratio(img, points), 0.0),
            # when the crests are out of frame there is no pelvis to compare against; assume it lies
            # straight, which makes the relative tilt equal the plain one (a missing value would read as
            # "spine perfectly aligned with the pelvis" and nudge the model the wrong way)
            "tilt_pelvis": _or(pelvis_relative_tilt(points, mm_per_px), abs(r["tilt_deg"] or 0.0)),
            "area_top": a["area_top"], "n_out": a["n_out"], "max_len": a["max_len"], "seg_area": float(seg_area),
            "printed_markup": a["n_overlay_lines"] >= 3}


def hip_measurements(img, points, conf, side, mm_per_px=None):
    f = hip_features(points, img.shape, side, MM_PER_PX if mm_per_px is None else mm_per_px)
    roi = hip_roi(points, img.shape)
    # the densitometer can print its own ROI boxes onto the export; report it for hips too, not only
    # for spines (Q&A 17.09: such markup exists on some scans and a specialist corrects it)
    return {"margin_bottom": f["margin_bottom"], "margin_top": f["margin_top"], "margin_lateral": f["margin_lateral"],
            "shaft_angle": f.get("shaft_angle"), "lt_conf": conf.get("lt", 0.0),
            "printed_markup": overlay_mask(img)[1] >= 3,
            "rois": [roi] if roi else []}


# How the threshold of a violation type is chosen. "f1" takes the F1 optimum on the training part;
# "prev" flags as many scans as the training prevalence. With 6-36 positives per type the F1 optimum
# sits wherever a couple of positives happen to land and generalises badly (see docs/night_report.md).
THRESHOLD_RULE = os.environ.get("DXAQC_THRESH_RULE", "f1")


def best_threshold(y, p, rule=None):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if (rule or THRESHOLD_RULE) == "prev":
        rate = float(np.mean(y))
        return float(np.quantile(p, 1 - rate)) if 0 < rate < 1 else float(p.max() + 1)
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

    INTERP = True

    def __init__(self, signs, lr_weights=True, percentile=True):
        self.signs, self.lr_weights, self.percentile = np.asarray(signs, float), lr_weights, percentile
        self.interp = self.INTERP

    def fit(self, X, y, X_syn=None, y_syn=None, w_syn=0.0):
        """X_syn, y_syn: synthetic scans (scripts/synth_decision.py), each weighing w_syn of a real one.
        They only inform the relative weights of the features; standardisation, the percentile reference
        and the calibration are fitted on real scans, so the probability scale stays that of real data."""
        Z = X * self.signs
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-9
        Z = (Z - self.mu) / self.sd
        if self.lr_weights and Z.shape[1] > 1:
            if X_syn is not None and len(X_syn) and w_syn > 0:
                Zs = (np.asarray(X_syn, float) * self.signs - self.mu) / self.sd
                yy = np.concatenate([y, y_syn])
                # the real scans keep the "balanced" class weights; every synthetic scan weighs w_syn of them
                cw = np.where(yy == 1, len(y) / (2 * max(y.sum(), 1)), len(y) / (2 * max((y == 0).sum(), 1)))
                sw = np.concatenate([np.ones(len(Z)), np.full(len(Zs), float(w_syn))]) * cw
                w = LogisticRegression(C=1.0, max_iter=2000).fit(np.vstack([Z, Zs]), yy, sample_weight=sw).coef_[0]
            else:
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
        return self._pct(s)

    def _pct(self, s):
        """Percentile of s among the training scores.

        A plain searchsorted gives a new image the rank of the NEXT training score above it, so every
        unseen image is shifted one step towards "bad" relative to the training images the threshold
        was chosen on. Where the threshold sits right next to a handful of positives (6-10 for the rare
        types) that step is exactly what separates a clean scan from a flagged one. Interpolating
        between mid-ranks treats training and new images alike. Models pickled before this change
        have no `interp` attribute and keep the old behaviour until they are refitted.
        """
        n = max(len(self.ref), 1)
        if not getattr(self, "interp", False):
            return np.searchsorted(self.ref, s) / n
        u, first, cnt = np.unique(self.ref, return_index=True, return_counts=True)
        return np.interp(s, u, (first + cnt / 2) / n, left=0.0, right=1.0)

    def predict_proba(self, X):
        s = ((X * self.signs - self.mu) / self.sd) @ self.w
        p = self.cal.predict_proba(self._scale(s)[:, None])[:, 1]
        return np.stack([1 - p, p], 1)

    def rank_score(self, X):
        """Where this image falls in the training distribution of the score, in [0, 1]."""
        s = ((X * self.signs - self.mu) / self.sd) @ self.w
        return self._pct(s)


ROTATION_BLEND = 0.2   # weight of the other hip in blend_rotation (fixed, not tuned on the data)
SENS_TARGET = 0.80     # sensitivity per violation type in the "screening" operating point
# Share of the HOG features the rotation forest looks at per split. 0.1 was chosen on the old
# landmark-aligned crops; on crops aligned by the traced shaft axis 0.2 is better — confirmed on a detector
# run and CV seeds that took no part in the choice (AUC 0.835 -> 0.852, PR-AUC 0.708 -> 0.732).
ROTATION_MAX_FEATURES = 0.2


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

    def fit(self, region, feats, labels, y_img=None, synth=None):
        """synth: {type: (X_syn, y_syn)} synthetic scans for the SYNTH_TYPES, measured like real ones."""
        self.region, self.models, self.thresholds, self.thresholds_screening = region, {}, {}, {}
        probs = {}
        synth = synth or {}
        for vt, (cols, signs) in TYPES[region].items():
            xs, ys = synth.get(vt, (None, None))
            m = Monotone(signs).fit(feats[vt], labels[vt], xs, ys, SYNTH_WEIGHT if vt in SYNTH_TYPES else 0.0)
            self.models[vt] = m
            probs[vt] = m.predict_proba(feats[vt])[:, 1]
            self.thresholds[vt] = float(best_threshold(labels[vt], probs[vt]))
            # second operating point: catch SENS_TARGET of this violation type, as specific as that allows —
            # but never stricter than the F1 one, so "screening" can only add findings, never drop them
            # (for v_axis the F1 threshold is already past that sensitivity).
            self.thresholds_screening[vt] = min(self.thresholds[vt],
                                                float(sensitivity_threshold(labels[vt], probs[vt], SENS_TARGET)))
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

    def predict(self, meas, policy="balanced"):
        """meas: measurements of one image -> (probabilities, decisions, image score).

        policy picks the per-type thresholds: "balanced" = F1-optimal (default, what the reported
        metrics are measured at), "screening" = the most specific threshold that still catches
        SENS_TARGET of that violation type."""
        thr = self.thresholds_for(policy)
        probs, decs, fuse = {}, {}, {}
        for vt, (cols, _) in TYPES[self.region].items():
            x = np.array([[np.nan_to_num(float(meas.get(c) if meas.get(c) is not None else np.nan), nan=0.0) for c in cols]])
            probs[vt] = float(self.models[vt].predict_proba(x)[0, 1])
            decs[vt] = probs[vt] >= thr[vt]
            fuse[vt] = (float(self.models[vt].rank_score(x)[0])
                        if FUSION[self.region] == "percentile" and hasattr(self.models[vt], "rank_score")
                        else probs[vt])
        raw = 1 - float(np.prod([1 - p for p in fuse.values()]))
        return probs, decs, self.calibrate(raw)

    def thresholds_for(self, policy):
        """Per-type thresholds of an operating point; falls back to the F1 ones for models pickled
        before the screening point existed."""
        if policy == "screening":
            return getattr(self, "thresholds_screening", None) or self.thresholds
        return self.thresholds

    def calibrate(self, raw):
        """Map the combined score to a probability. One calibration per region, fitted on every
        labelled image, so quality_prob still reads as a probability after percentile fusion."""
        cal = getattr(self, "image_cal", None)
        return float(cal.predict_proba([[raw]])[0, 1]) if cal is not None else float(raw)

    def verdict(self, probs, decs, score, policy="balanced"):
        """(quality_class, violation types) for one image: bad when any violation type fires.

        A threshold on the combined score was tried instead and shipped for a while. Once the
        per-type probabilities were put on a common scale (percentile calibration) and the rotation
        model improved, the plain union became better on every metric ТЗ 8.4 lists — per image
        F1 0.741 against 0.702, per study 0.800 against 0.752, macro-F1 over types 0.611 against
        0.588 — and it needs no "most probable type" hedge: every flagged image names its reason.
        The combined score is still reported as quality_prob, which is what ROC-AUC is computed from.
        The operating point is chosen when the per-type decisions are made (predict), not here.
        """
        types = [vt for vt, d in decs.items() if d]
        return int(bool(types)), types
