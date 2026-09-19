"""HTTP API tests (FastAPI TestClient, no server needed)."""
import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from dxaqc.api import app

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = sorted((ROOT / "data/test_sample").glob("*.dcm"))

# The organisers' sample DICOMs are medical data and are not in the repository: put them in
# data/test_sample to run the tests that need them; everything else runs without.
needs_samples = pytest.mark.skipif(not SAMPLES, reason="нужны тестовые DICOM организаторов в data/test_sample")
client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and "kp_hip.pt" in r.json()["models"]


@needs_samples
def test_predict_files_json():
    files = [("files", (p.name, p.read_bytes(), "application/dicom")) for p in SAMPLES]
    rows = client.post("/predict", files=files).json()
    assert len(rows) == 3 and all(r["processing_status"] == "Success" for r in rows)


@needs_samples
def test_predict_zip_csv_and_vis():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for p in SAMPLES:
            z.write(p, f"study1/{p.name}")
    r = client.post("/predict?format=csv", files=[("files", ("batch.zip", buf.getvalue(), "application/zip"))])
    df = pd.read_csv(io.BytesIO(r.content))
    assert len(df) == 3 and set(df.path_to_study) == {"study1"}
    r = client.post("/predict?format=zip", files=[("files", ("batch.zip", buf.getvalue(), "application/zip"))])
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert "results.csv" in names and sum(n.endswith("_qc.dcm") for n in names) == 3


def test_predict_folder_rejects_outside_paths():
    assert client.post("/predict_folder", json={"input": "/etc", "output": "/tmp/x.csv"}).status_code == 403


def test_web_interface_is_served_and_offline():
    """The page loads from the package and references no external host (ТЗ 3.2: works without internet)."""
    r = client.get("/")
    assert r.status_code == 200 and "Контроль качества денситометрии" in r.text
    for marker in ("http://", "https://", "//cdn"):
        assert marker not in r.text.replace("http://localhost:8000", ""), marker


@needs_samples
def test_analyse_returns_rows_with_overlay():
    """The endpoint behind the web interface returns a row per image plus a base64 overlay."""
    with open(SAMPLES[0], "rb") as f:
        r = client.post("/analyse", files={"files": (SAMPLES[0].name, f.read(), "application/dicom")})
    assert r.status_code == 200
    rows = r.json()["rows"]
    assert len(rows) == 1
    row = rows[0]
    assert row["file"] == SAMPLES[0].name
    assert row["processing_status"] == "Success"
    assert row["quality_class"] in (0, 1) and 0 <= row["quality_prob"] <= 1
    assert len(row.get("overlay", "")) > 1000        # PNG, base64
