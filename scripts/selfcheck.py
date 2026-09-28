"""Проверка, что сборка живая — без единого медицинского снимка.

Нужна тому, кто клонировал репозиторий и хочет убедиться, что образ собрался правильно, прежде чем
подставлять настоящие исследования. Проверяет четыре вещи:

  1. модели на месте и загружаются;
  2. конвейер отрабатывает на синтетическом изображении и не падает;
  3. защита «свой/чужой» срабатывает — шум это не денситометрия, сервис обязан отказаться;
  4. пакетная обработка собирает таблицу в формате ТЗ со всеми обязательными колонками.

    python scripts/selfcheck.py          # локально
    ./run.sh check                        # в контейнере
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

REQUIRED = ["path_to_study", "study_uid", "image_uid", "anatomical_region", "quality_class",
            "violation_type", "processing_status", "time_of_processing"]


def synthetic_dicom(path, shape=(300, 300), seed=0):
    """Шумовой снимок в корректном DICOM: анатомии нет, и сервис обязан это заметить."""
    rng = np.random.default_rng(seed)
    arr = (rng.normal(60, 25, shape).clip(0, 255)).astype(np.uint8)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID = generate_uid(), generate_uid()
    ds.Modality, ds.PhotometricInterpretation, ds.SamplesPerPixel = "CR", "MONOCHROME2", 1
    ds.Rows, ds.Columns = shape
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    ds.PixelData = arr.tobytes()
    ds.save_as(path, enforce_file_format=True)
    return path


def main():
    from dxaqc.batch import run_batch
    from dxaqc.service import QCService

    print("1/4 загрузка моделей…", flush=True)
    svc = QCService("models")
    names = sorted(p.name for p in Path("models").glob("*"))
    print(f"    модели: {', '.join(names)}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        synthetic_dicom(tmp / "synthetic.dcm")

        print("2/4 прогон конвейера на синтетическом снимке…", flush=True)
        df = run_batch(tmp, svc, None)
        row = df.iloc[0]
        print(f"    статус обработки: {row.processing_status}, время {row.time_of_processing} с")

        print("3/4 защита «свой/чужой»…", flush=True)
        refused = row.violation_codes == "unsupported"
        print(f"    ответ сервиса: {row.details[:80] if row.details else '—'}")

        print("4/4 формат таблицы…", flush=True)
        missing = [c for c in REQUIRED if c not in df.columns]
        print(f"    колонки ТЗ 2.5: {'все на месте' if not missing else 'НЕТ ' + ', '.join(missing)}")

    ok = row.processing_status == "Success" and refused and not missing
    print()
    if ok:
        print("Сборка исправна: модели загружаются, конвейер работает, посторонние снимки отклоняются,")
        print("таблица собирается в формате ТЗ. Можно подставлять настоящие исследования:")
        print("    ./run.sh predict /путь/к/исследованиям")
        return 0
    print("ПРОВЕРКА НЕ ПРОЙДЕНА — смотрите вывод выше", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
