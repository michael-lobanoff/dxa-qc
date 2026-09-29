"""Batch processing shared by the CLI (scripts/predict.py) and the HTTP API (dxaqc/api.py)."""
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

# ТЗ 2.5 asks for the first eight columns; the rest is ours and helps whoever checks the result.
# "projection" answers ТЗ 2.2 ("область и проекция"), "markup_suggestion" carries the proposed
# correction of the printed markup that the specialist confirms in the web interface (ТЗ 2.6).
COLUMNS = ["path_to_study", "study_uid", "image_uid", "anatomical_region", "projection", "quality_class",
           "violation_type", "processing_status", "time_of_processing", "quality_prob", "violation_codes",
           "details", "markup_suggestion", "mm_per_px", "mm_per_px_source", "file", "error"]


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


def zip_top_level(path: Path):
    """Имя единственной папки верхнего уровня в архиве, если она там одна.

    Архив, собранный из папки исследования, уже несёт её имя внутри: раскрывать его ещё и в
    папку по имени архива значит получить «исследование/исследование» в path_to_study.
    """
    tops = set()
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            name = _zip_name(info)
            if info.is_dir() or "__MACOSX" in name or Path(name).name.startswith("."):
                continue
            parts = Path(name).parts
            if len(parts) < 2:
                return None
            tops.add(parts[0])
    return tops.pop() if len(tops) == 1 else None


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


def contralateral_rotation(rows):
    """{row index: rotation probability of the other hip in the same study}.

    Both hips of a patient are positioned in one session, so their rotation labels agree far more
    often than chance; using the other hip stabilises the verdict (see decision.blend_rotation).
    """
    by_study = {}
    for k, r in enumerate(rows):
        if r.get("_region") in ("hip_left", "hip_right") and r.get("_rotation") is not None:
            by_study.setdefault(r.get("study_uid") or r.get("path_to_study"), {})[r["_region"]] = (k, r["_rotation"])
    out = {}
    for sides in by_study.values():
        for side, (k, _) in sides.items():
            other = sides.get("hip_right" if side == "hip_left" else "hip_left")
            if other is not None:
                out[k] = other[1]
    return out


def run_batch(root: Path, service, vis_dir=None, policy=None) -> pd.DataFrame:
    """All DICOM files under root (folder or .zip) -> one results row per image.

    Two passes: every image is analysed on its own, then hips whose study contains the other side are
    re-decided with the two rotation probabilities mixed. Overlays are rendered in the second pass so
    the picture always matches the verdict in the table.
    """
    root = Path(root)
    with tempfile.TemporaryDirectory() as tmp:
        if root.suffix.lower() == ".zip":
            extract_zip(root, Path(tmp))
            root = Path(tmp)
        elif any(p.is_file() and p.suffix.lower() == ".zip" for p in root.rglob("*")):
            # архивы лежат вперемешку с отдельными файлами: их надо распаковать, иначе снимки
            # внутри просто не попадут в отчёт, а строка итога скажет «0 ошибок»
            work = Path(tmp)
            for p in sorted(root.rglob("*")):
                if not p.is_file():
                    continue
                rel = p.relative_to(root)
                if p.suffix.lower() == ".zip":
                    inner = zip_top_level(p)
                    extract_zip(p, work / rel.parent if inner else work / rel.with_suffix(""))
                else:
                    (work / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, work / rel)
            root = work
        files, rows = [], []
        for p in iter_dicoms(root):
            rel = p.relative_to(root)
            study = rel.parts[0] if len(rel.parts) > 1 else ""
            files.append((p, rel, study))
            rows.append(service.process_file(p, study_dir=study, policy=policy))
            rows[-1]["file"] = str(rel)
        alpha = getattr(service, "rotation_blend", 0.0)
        contra = contralateral_rotation(rows) if alpha else {}
        for k, (p, rel, study) in enumerate(files):
            need_vis = vis_dir is not None
            if k not in contra and not need_vis:
                continue
            override = None
            if k in contra:
                from .decision import blend_rotation
                override = blend_rotation(rows[k]["_rotation"], contra[k], alpha)
                if abs(override - rows[k]["_rotation"]) < 1e-9 and not need_vis:
                    continue
            rows[k] = service.process_file(p, study_dir=study, rotation_override=override, policy=policy,
                                           vis_dir=(Path(vis_dir) / study) if need_vis else None)
            rows[k]["file"] = str(rel)
    study_folders(rows)
    return pd.DataFrame(rows, columns=COLUMNS)


def zip_folder(folder: Path, archive: Path):
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(Path(folder).rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(folder))
