"""Batch processing shared by the CLI (scripts/predict.py) and the HTTP API (dxaqc/api.py)."""
import os
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

COLUMNS = ["path_to_study", "study_uid", "image_uid", "anatomical_region", "quality_class", "violation_type",
           "processing_status", "time_of_processing", "quality_score", "violation_codes", "details", "mm_per_px", "mm_per_px_source", "file", "error"]


def _zip_name(info):
    """Archives made without the UTF-8 flag store Cyrillic names in the local code page."""
    if info.flag_bits & 0x800:
        return info.filename
    raw = info.filename.encode("cp437")
    for enc in ("utf-8", "cp866", "cp1251"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return info.filename


def extract_zip(path: Path, dest: Path):
    dest = dest.resolve()
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            name = _zip_name(info)
            target = (dest / name).resolve()
            if info.is_dir() or not str(target).startswith(str(dest) + os.sep):  # skip dirs and path traversal
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as out:
                out.write(src.read())


def study_folders(rows):
    """path_to_study: the deepest folder shared by all files of the same StudyInstanceUID."""
    by_study = {}
    for r in rows:
        by_study.setdefault(r["study_uid"] or r["file"], []).append(Path(r["file"]).parent)
    common = {k: Path(os.path.commonpath([str(p) for p in v])) if len(v) > 1 else v[0] for k, v in by_study.items()}
    for r in rows:
        r["path_to_study"] = str(common[r["study_uid"] or r["file"]])


def iter_dicoms(root: Path):
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name.startswith(".") or "__MACOSX" in p.parts:
            continue
        with open(p, "rb") as f:
            head = f.read(132)
        if p.suffix.lower() in (".dcm", ".dicom") or head[128:132] == b"DICM":
            yield p


def run_batch(root: Path, service, vis_dir=None) -> pd.DataFrame:
    """All DICOM files under root (folder or .zip) -> one results row per image."""
    root = Path(root)
    with tempfile.TemporaryDirectory() as tmp:
        if root.suffix.lower() == ".zip":
            extract_zip(root, Path(tmp))
            root = Path(tmp)
        rows = []
        for p in iter_dicoms(root):
            rel = p.relative_to(root)
            study = rel.parts[0] if len(rel.parts) > 1 else ""
            rows.append(service.process_file(p, study_dir=study, vis_dir=(Path(vis_dir) / study) if vis_dir else None))
            rows[-1]["file"] = str(rel)
    study_folders(rows)
    return pd.DataFrame(rows, columns=COLUMNS)


def zip_folder(folder: Path, archive: Path):
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(Path(folder).rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(folder))
