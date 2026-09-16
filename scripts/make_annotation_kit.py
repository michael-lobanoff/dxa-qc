"""Pack tools/annotator + exported images into a zip that teammates unpack and run locally."""
import zipfile
from pathlib import Path

START_SH = """#!/bin/sh
cd "$(dirname "$0")"
URL=http://127.0.0.1:8765/tools/annotator/
( sleep 1; (command -v open >/dev/null && open "$URL") || xdg-open "$URL" ) &
python3 -m http.server 8765 --bind 127.0.0.1
"""
START_BAT = """@echo off
cd /d %~dp0
start http://127.0.0.1:8765/tools/annotator/
python -m http.server 8765 --bind 127.0.0.1
"""


def main():
    out = Path("dist/annotation_kit.zip")
    out.parent.mkdir(exist_ok=True)
    files = [Path("tools/annotator/index.html"), Path("docs/annotation_scheme.md"), Path("docs/annotation_guide.html"),
             Path("data/annotation/manifest.js")]
    files += sorted(Path("data/annotation/images").glob("*.png")) + sorted(Path("data/annotation/guide").glob("*.png"))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f"annotation_kit/{f}")
        for name, body in [("start_annotator.sh", START_SH), ("start_annotator.bat", START_BAT)]:
            info = zipfile.ZipInfo(f"annotation_kit/{name}")
            info.external_attr = 0o755 << 16
            z.writestr(info, body)
    print(f"{out}: {len(files) + 2} files, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
