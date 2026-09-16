"""DICOM Basic Text SR with the QC findings (ТЗ 2.6: textual description of violations in DICOM SR)."""
import datetime

import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

BASIC_TEXT_SR = "1.2.840.10008.5.1.4.1.1.88.11"


def _code(value, scheme, meaning):
    c = Dataset()
    c.CodeValue, c.CodingSchemeDesignator, c.CodeMeaning = value, scheme, meaning
    return c


def _text_item(meaning_code, text):
    item = Dataset()
    item.RelationshipType, item.ValueType = "CONTAINS", "TEXT"
    item.ConceptNameCodeSequence = [meaning_code]
    item.TextValue = text
    return item


def to_structured_report(row, ref: pydicom.Dataset):
    """row: a results row from QCService.process_file (Russian wording); ref: the source image dataset."""
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = BASIC_TEXT_SR
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SpecificCharacterSet = "ISO_IR 192"  # UTF-8: findings are in Russian
    for tag in ("PatientID", "PatientName", "PatientBirthDate", "PatientSex", "StudyInstanceUID", "StudyDate",
                "StudyTime", "AccessionNumber", "StudyID", "ReferringPhysicianName"):
        if tag in ref:
            setattr(ds, tag, getattr(ref, tag))
    now = datetime.datetime.now()
    ds.SOPClassUID, ds.SOPInstanceUID = BASIC_TEXT_SR, meta.MediaStorageSOPInstanceUID
    ds.Modality, ds.SeriesInstanceUID, ds.SeriesNumber, ds.InstanceNumber = "SR", generate_uid(), 998, 1
    ds.SeriesDescription = "DXA QC report"
    ds.ContentDate, ds.ContentTime = now.strftime("%Y%m%d"), now.strftime("%H%M%S")
    ds.CompletionFlag, ds.VerificationFlag = "COMPLETE", "UNVERIFIED"
    ds.ValueType, ds.ContinuityOfContent = "CONTAINER", "SEPARATE"
    ds.ConceptNameCodeSequence = [_code("18748-4", "LN", "Diagnostic imaging report")]

    ev = Dataset()
    ev.StudyInstanceUID = getattr(ref, "StudyInstanceUID", "")
    series = Dataset()
    series.SeriesInstanceUID = getattr(ref, "SeriesInstanceUID", "")
    sop = Dataset()
    sop.ReferencedSOPClassUID, sop.ReferencedSOPInstanceUID = getattr(ref, "SOPClassUID", ""), getattr(ref, "SOPInstanceUID", "")
    series.ReferencedSOPSequence = [sop]
    ev.ReferencedSeriesSequence = [series]
    ds.CurrentRequestedProcedureEvidenceSequence = [ev]

    finding = _code("121071", "DCM", "Finding")
    verdict = "есть нарушение качества" if row.get("quality_class") == 1 else "качественное"
    items = [_text_item(_code("121072", "DCM", "Impression"), f"Контроль качества DXA: {verdict}."),
             _text_item(finding, f"Область: {row.get('anatomical_region', '')}.")]
    if row.get("violation_type"):
        items.append(_text_item(finding, f"Нарушения: {row['violation_type']}."))
    if row.get("details"):
        items.append(_text_item(finding, f"Измерения: {row['details']}."))
    if row.get("quality_score") is not None:
        items.append(_text_item(finding, f"Вероятность нарушения по модели: {row['quality_score']:.2f}."))
    img = Dataset()
    img.RelationshipType, img.ValueType = "CONTAINS", "IMAGE"
    img.ConceptNameCodeSequence = [_code("121112", "DCM", "Source of Measurement")]
    img.ReferencedSOPSequence = [sop]
    items.append(img)
    items.append(_text_item(_code("121106", "DCM", "Comment"),
                            "Автоматическая оценка, требует подтверждения специалистом."))
    ds.ContentSequence = items
    return ds
