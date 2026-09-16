"""HTTP API tests (FastAPI TestClient, no server needed)."""
import io
import zipfile
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from dxaqc.api import app

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = sorted((ROOT / "data/test_sample").glob("*.dcm"))
client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and "kp_hip.pt" in r.json()["models"]


def test_predict_files_json():
    files = [("files", (p.name, p.read_bytes(), "application/dicom")) for p in SAMPLES]
    rows = client.post("/predict", files=files).json()
    assert len(rows) == 3 and all(r["processing_status"] == "Success" for r in rows)


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
