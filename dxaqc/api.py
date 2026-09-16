"""HTTP API for batch QC (ТЗ 3.2).

Run:  uvicorn dxaqc.api:app --host 0.0.0.0 --port 8000
  GET  /health                      models loaded, version
  POST /predict?format=json|csv|xlsx|zip   multipart files: DICOM files or one .zip archive
  POST /predict_folder              {"input": "/data", "output": "/out/results.csv", "vis": true}
                                    server-side batch over a mounted folder (paths limited to DXAQC_ALLOWED_ROOTS)
"""
import io
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from .batch import run_batch, zip_folder
from .service import QCService

MODEL_DIR = Path(os.environ.get("DXAQC_MODELS", "models"))
ALLOWED_ROOTS = [Path(p).resolve() for p in os.environ.get("DXAQC_ALLOWED_ROOTS", f"/data:/out:{os.getcwd()}").split(":") if p]
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

app = FastAPI(title="DXA QC", version="0.1.0", description="Контроль качества денситометрических исследований")
_service = None


def service():
    global _service
    if _service is None:
        _service = QCService(MODEL_DIR)
    return _service


def _allowed(path: Path) -> bool:
    p = path.resolve()
    return any(p == r or r in p.parents for r in ALLOWED_ROOTS)


def _zip_bytes(folder: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(folder.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(folder))
    return buf.getvalue()


@app.get("/health")
def health():
    service()
    return {"status": "ok", "version": app.version, "models": sorted(p.name for p in MODEL_DIR.glob("*") if p.is_file())}


@app.post("/predict")
async def predict(files: list[UploadFile] = File(...), format: str = Query("json", pattern="^(json|csv|xlsx|zip)$")):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "in"
        root.mkdir()
        for k, f in enumerate(files):
            name = Path(f.filename or f"file{k}.dcm").name or f"file{k}.dcm"  # never trust client paths
            (root / f"{k:04d}_{name}").write_bytes(await f.read())
        items = list(root.iterdir())
        src = items[0] if len(items) == 1 and items[0].suffix.lower() == ".zip" else root
        vis = Path(tmp) / "vis" if format == "zip" else None
        df = run_batch(src, service(), vis)
        if format == "json":
            return JSONResponse(json.loads(df.to_json(orient="records", force_ascii=False)))
        if format == "csv":
            return Response(df.to_csv(index=False).encode("utf-8-sig"), media_type="text/csv",
                            headers={"Content-Disposition": "attachment; filename=results.csv"})
        if format == "xlsx":
            buf = io.BytesIO(); df.to_excel(buf, index=False)
            return Response(buf.getvalue(), media_type=XLSX, headers={"Content-Disposition": "attachment; filename=results.xlsx"})
        out = Path(tmp) / "out"
        out.mkdir()
        df.to_csv(out / "results.csv", index=False, encoding="utf-8-sig")
        df.to_excel(out / "results.xlsx", index=False)
        if vis and vis.exists():
            shutil.copytree(vis, out / "overlays")
        return Response(_zip_bytes(out), media_type="application/zip", headers={"Content-Disposition": "attachment; filename=results.zip"})


class FolderJob(BaseModel):
    input: str
    output: str = "/out/results.csv"
    vis: bool = False


@app.post("/predict_folder")
def predict_folder(job: FolderJob):
    src, out = Path(job.input), Path(job.output)
    if not _allowed(src) or not _allowed(out.parent):
        raise HTTPException(403, f"paths must be inside {', '.join(map(str, ALLOWED_ROOTS))}")
    if not src.exists():
        raise HTTPException(404, f"{src} not found")
    out.parent.mkdir(parents=True, exist_ok=True)
    vis = out.with_name(out.stem + "_vis") if job.vis else None
    df = run_batch(src, service(), vis)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    df.to_excel(out.with_suffix(".xlsx"), index=False)
    result = {"images": len(df), "success": int((df.processing_status == "Success").sum()), "output": str(out)}
    if vis and vis.exists():
        zip_folder(vis, out.with_name(out.stem + "_vis.zip"))
        result["overlays"] = str(out.with_name(out.stem + "_vis.zip"))
    return result
