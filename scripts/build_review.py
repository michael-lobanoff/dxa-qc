"""Export expert labels for tools/viewer (review only: never ship this file to annotators)."""
import json
from pathlib import Path

import pandas as pd

LABELS = {"spine": ["y", "v_axis", "v_pos", "v_artifact"], "hip": ["y", "v_posrot", "v_roi"]}


def main():
    idx = pd.read_csv("data/train/image_index.csv")
    out = {}
    for r in idx.itertuples():
        cols = LABELS["spine" if r.region == "spine" else "hip"]
        vals = {c: (None if pd.isna(getattr(r, c)) else int(getattr(r, c))) for c in cols}
        out[f"{r.n:03d}_{r.region}"] = {**vals, "comment": None if pd.isna(r.comment) else str(r.comment).strip()}
    Path("data/annotation/review.js").write_text("window.REVIEW = " + json.dumps(out, ensure_ascii=False) + ";\n")
    print(f"{len(out)} images -> data/annotation/review.js")


if __name__ == "__main__":
    main()
