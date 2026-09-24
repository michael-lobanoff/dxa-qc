"""Does the decision layer survive a change of densitometer?

The region check has been tested on the public Multan set (350 of 350 recognised), but that says nothing
about the verdicts: to test those you need labels, and the set has none. So we make labels by
construction — take an external scan, inject a violation whose ground truth is defined by the rule of
ТЗ 2.3, and see whether the service says so. The same generator that builds synthetic violations for
training is used (dxaqc.synth), but here it runs on images from a scanner the models have never seen.

What is measured:
  - «поймано» — the share of injected violations the service flags with that exact type;
  - «фон»     — how often the service flags the untouched image, i.e. how much of the detection is just
                the service disliking a foreign scanner.

    python scripts/check_synth_external.py data/external/mendeley_dexa_spine --limit 60
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from dxaqc import synth
from dxaqc.service import QCService


def points_of(res):
    return {k: v for k, v in (res.get("points") or {}).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--models", default="models")
    ap.add_argument("--limit", type=int, default=60)
    args = ap.parse_args()
    svc = QCService(args.models)
    rng = np.random.default_rng(0)

    files = [p for p in sorted(args.folder.iterdir()) if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    stats = {k: [0, 0] for k in ("tilt", "cut_bottom", "artifact")}      # [поймано, всего]
    base_bad, base_types, n = 0, {}, 0
    for p in files:
        img = np.asarray(Image.open(p).convert("L"))
        if img.shape == (224, 224):                 # downscaled copies: not the resolution we support
            continue
        base = svc.analyse(img)
        if base.get("unsupported") or base["region"] != "spine":
            continue
        n += 1
        base_bad += int(bool(base["violations"]))
        for v in base["violations"]:
            base_types[v] = base_types.get(v, 0) + 1
        kp = {k: (list(v) if v else None) for k, v in points_of(base).items()}
        for kind, vt in (("tilt", "v_axis"), ("cut_bottom", "v_pos"), ("artifact", "v_artifact")):
            if vt in base["violations"]:            # already flagged before the injection: not a test
                continue
            try:
                simg, skp, lab = synth.spine_sample(img, kp, kind, rng)
            except (ValueError, KeyError, TypeError):
                continue
            r = svc.analyse(simg)
            stats[kind][1] += 1
            stats[kind][0] += int(vt in r["violations"])
        if n >= args.limit:
            break

    print(f"снимков с чужого аппарата: {n}")
    print(f"фон (сервис считает исходный снимок некачественным): {base_bad} ({base_bad / max(n, 1):.0%}), "
          f"по типам {base_types}")
    names = {"tilt": "наклон оси > 5°", "cut_bottom": "гребни ушли из кадра", "artifact": "нарисована косточка белья"}
    for k, (hit, tot) in stats.items():
        if tot:
            print(f"внесено «{names[k]}»: поймано {hit} из {tot} ({hit / tot:.0%})")


if __name__ == "__main__":
    main()
