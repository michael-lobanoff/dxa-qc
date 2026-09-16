"""Build a per-image index of the training set.

Studies contain repeated exports of the same scan, so images are deduplicated
by pixel hash. Each unique image gets a region (spine / hip_left / hip_right),
an implant flag, and the study-level expert labels for that region.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

from dxaqc.pixels import pixel_spacing_mm

LABEL_COLS = ["n", "study", "sp_pos", "sp_axis", "sp_artifact", "rh_posrot", "rh_roi",
              "lh_posrot", "lh_roi", "y_spine", "y_rhip", "y_lhip", "comment"]
REGION_LABELS = {
    "spine": {"y": "y_spine", "v_pos": "sp_pos", "v_axis": "sp_axis", "v_artifact": "sp_artifact"},
    "hip_right": {"y": "y_rhip", "v_posrot": "rh_posrot", "v_roi": "rh_roi"},
    "hip_left": {"y": "y_lhip", "v_posrot": "lh_posrot", "v_roi": "lh_roi"},
}
METAL_FRACTION = 0.015  # share of saturated pixels; implants ~0.018-0.12, normal hips <0.012


def read_labels(xlsx: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx, header=None, skiprows=2, usecols=range(len(LABEL_COLS)), names=LABEL_COLS)
    return df.dropna(subset=["study"]).set_index("study")


def hip_side(img: np.ndarray) -> float:
    """>0: femoral head on the image right (right hip in AP view), <0: left hip."""
    a = img.astype(np.float32)
    h, w = a.shape
    thr = np.percentile(a, 70)
    xs = np.arange(w)

    def cx(mask):
        return (mask.sum(0) * xs).sum() / max(mask.sum(), 1)

    return cx(a[: int(h * 0.35)] > thr) - cx(a[int(h * 0.65):] > thr)


def detect_region(img: np.ndarray) -> str:
    # All training images come from GE Lunar Prodigy: spine exports are 300 px wide, hips 280/248.
    if img.shape[1] == 300:
        return "spine"
    return "hip_right" if hip_side(img) > 0 else "hip_left"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/train"))
    ap.add_argument("--out", type=Path, default=Path("data/train/image_index.csv"))
    args = ap.parse_args()

    labels = read_labels(args.root / "разметка.xlsx")
    rows, seen = [], set()
    for path in sorted((args.root / "Исследования").rglob("*.dcm")):
        study = path.relative_to(args.root / "Исследования").parts[0]
        ds = pydicom.dcmread(path)
        img = ds.pixel_array
        digest = hashlib.md5(img.tobytes() + str(img.shape).encode()).hexdigest()
        if (study, digest) in seen:
            continue
        seen.add((study, digest))

        region = detect_region(img)
        mm, mm_src = pixel_spacing_mm(ds, img.shape)
        implant = bool((img >= img.max() - 2).mean() > METAL_FRACTION) and region != "spine"
        row = {"path": str(path.relative_to(args.root)), "study": study, "n": int(labels.loc[study, "n"]),
               "study_uid": ds.StudyInstanceUID, "image_uid": ds.SOPInstanceUID,
               "rows": img.shape[0], "cols": img.shape[1], "region": region, "implant": implant,
               "mm_per_px": round(mm, 4), "mm_per_px_source": mm_src}
        lab = labels.loc[study]
        for key, col in REGION_LABELS[region].items():
            row[key] = lab[col]
        row["comment"] = lab["comment"]
        rows.append(row)

    df = pd.DataFrame(rows)
    assert not df.duplicated(["study", "region"]).any(), "several unique images of one region in a study"
    df.to_csv(args.out, index=False)
    print(f"{len(df)} unique images from {df.study.nunique()} studies -> {args.out}")
    print(df.groupby("region").agg(n=("y", "size"), violations=("y", "sum"), unlabeled=("y", lambda s: s.isna().sum())))


if __name__ == "__main__":
    main()
