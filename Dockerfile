# DXA QC service.
# Batch CLI: docker run -v <studies>:/data -v <out>:/out dxa-qc /data --out /out/results.csv --vis /out/vis
# HTTP API:  docker run -p 8000:8000 -v <studies>:/data -v <out>:/out --entrypoint uvicorn dxa-qc dxaqc.api:app --host 0.0.0.0 --port 8000
FROM python:3.12.2-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DXAQC_MODELS=/app/models
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY dxaqc ./dxaqc
COPY scripts/predict.py ./scripts/predict.py
COPY models ./models
RUN pip install --no-cache-dir --no-deps -e .

ENTRYPOINT ["python", "scripts/predict.py", "--models", "/app/models"]
