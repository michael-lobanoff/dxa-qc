"""Merge annotator exports from tools/annotator into one keypoint file.

--policy average: independent annotators. Images annotated by several people (every
10th study is shared on purpose) are averaged and reported as inter-annotator agreement.
--policy latest: review / correction rounds. The most recently edited record of each
image wins, so a reviewer's fix replaces the original instead of being averaged with it.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from dxaqc.geometry import MM_PER_PX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exports", nargs="+", type=Path, help="dxa_keypoints_*.json files")
    ap.add_argument("--manifest", type=Path, default=Path("data/annotation/manifest.js"))
    ap.add_argument("--out", type=Path, default=Path("data/train/keypoints.json"))
    ap.add_argument("--policy", choices=["average", "latest"], default="average")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text().split("=", 1)[1].rstrip().rstrip(";"))
    by_image = defaultdict(dict)  # image id -> annotator -> record (latest wins)
    for path in args.exports:
        data = json.loads(path.read_text())
        for image_id, rec in data["annotations"].items():
            who = rec.get("annotator") or data.get("annotator") or path.stem
            old = by_image[image_id].get(who)
            if old is None or rec.get("updated", "") > old.get("updated", ""):
                by_image[image_id][who] = rec

    size = {m["id"]: (m["w"], m["h"]) for m in manifest}
    if args.policy == "latest":
        merged = {}
        for image_id, recs in by_image.items():
            who, rec = max(recs.items(), key=lambda kv: kv[1].get("updated", ""))
            w, h = size[image_id]
            points = {k: (None if v is None else [min(max(v[0], 0.0), w - 1.0), min(max(v[1], 0.0), h - 1.0)])
                      for k, v in rec.get("points", {}).items()}
            merged[image_id] = {"points": points, "boxes": rec.get("boxes", []), "flags": rec.get("flags", {}),
                                "choice": rec.get("choice"), "annotators": sorted(recs), "source": who}
        args.out.write_text(json.dumps(merged, ensure_ascii=False, indent=1))
        print(f"merged {len(merged)} images (latest wins) -> {args.out}")
        print("records taken from:", dict(Counter(v["source"] for v in merged.values())))
        return

    merged, dist, n_conflict, lt_pairs = {}, defaultdict(list), defaultdict(int), []
    for image_id, recs in by_image.items():
        recs = [r for r in recs.values() if r.get("points") or r.get("flags", {}).get("skip")]
        if not recs:
            continue
        keys = set().union(*(r["points"].keys() for r in recs))
        points = {}
        for k in keys:
            vals = [r["points"].get(k) for r in recs if k in r["points"]]
            placed = [v for v in vals if v is not None]
            if len(placed) not in (0, len(vals)):
                n_conflict[k] += 1  # one annotator saw the point, another marked it absent
            points[k] = np.mean(placed, axis=0).round(1).tolist() if len(placed) * 2 >= len(vals) and placed else None
            if len(placed) >= 2:
                dist[k].extend(np.linalg.norm(np.subtract(a, b)) for i, a in enumerate(placed) for b in placed[i + 1:])
        choices = [r.get("choice") for r in recs if r.get("choice")]
        if len(choices) >= 2:
            lt_pairs.append(len(set(choices)) == 1)
        merged[image_id] = {
            "points": points,
            "boxes": [b for r in recs for b in r.get("boxes", [])],
            "flags": {f: any(r.get("flags", {}).get(f) for r in recs) for f in {f for r in recs for f in r.get("flags", {})}},
            "choice": max(set(choices), key=choices.count) if choices else None,
            "annotators": sorted(by_image[image_id].keys()),
        }

    args.out.write_text(json.dumps(merged, ensure_ascii=False, indent=1))
    regions = defaultdict(lambda: [0, 0])
    for item in manifest:
        reg = "spine" if item["region"] == "spine" else "hip"
        regions[reg][1] += 1
        regions[reg][0] += item["id"] in merged
    print(f"merged {len(merged)} images -> {args.out}")
    print("coverage:", {r: f"{a}/{b}" for r, (a, b) in regions.items()})
    if dist or n_conflict:
        print("\ninter-annotator agreement (overlap images):")
        for k in sorted(set(dist) | set(n_conflict)):
            d = dist.get(k, [])
            spread = (f"median {np.median(d):5.1f}px ({np.median(d) * MM_PER_PX:4.1f} mm)  p90 {np.percentile(d, 90):5.1f}px"
                      if d else "no pairs placed by both")
            print(f"  {k:8s} n={len(d):3d}  {spread}  absent-vs-placed conflicts: {n_conflict[k]}")
    if lt_pairs:
        print(f"lesser trochanter class agreement: {np.mean(lt_pairs):.0%} (n={len(lt_pairs)})")


if __name__ == "__main__":
    main()
