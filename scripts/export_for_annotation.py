"""Export unique training images as PNG plus a manifest for tools/annotator."""
import argparse
import json
from pathlib import Path

import pandas as pd
import pydicom
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/train"))
    ap.add_argument("--out", type=Path, default=Path("data/annotation"))
    args = ap.parse_args()

    index = pd.read_csv(args.root / "image_index.csv")
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    items = []
    for r in index.sort_values(["region", "n"]).itertuples():
        image_id = f"{r.n:03d}_{r.region}"
        img = pydicom.dcmread(args.root / r.path).pixel_array
        Image.fromarray(img).save(args.out / "images" / f"{image_id}.png")
        # Labels are deliberately not exported: annotators must not see the expert verdict.
        items.append({"id": image_id, "region": r.region, "n": int(r.n), "w": int(img.shape[1]),
                      "h": int(img.shape[0]), "implant": bool(r.implant)})
    (args.out / "manifest.js").write_text("window.MANIFEST = " + json.dumps(items, ensure_ascii=False) + ";\n")
    print(f"{len(items)} images -> {args.out}")


if __name__ == "__main__":
    main()
