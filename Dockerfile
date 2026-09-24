# DXA QC service.
# Batch CLI: docker run -v <studies>:/data -v <out>:/out dxa-qc /data --out /out/results.csv --vis /out/vis
# HTTP API:  docker run -p 8000:8000 -v <studies>:/data -v <out>:/out --entrypoint uvicorn dxa-qc dxaqc.api:app --host 0.0.0.0 --port 8000
# Built for linux/amd64 (the organisers' test host); on Apple Silicon add --platform linux/amd64.
FROM python:3.12.2-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DXAQC_MODELS=/app/models \
    PIP_NO_CACHE_DIR=1 PIP_DEFAULT_TIMEOUT=180 PIP_RETRIES=10
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# CPU build of torch by default: the service needs 0.43 s per image on a CPU (limit is 3 minutes per
# study), while the CUDA wheel pulls ~2.5 GB — that is what made the first build time out. The final
# testing host has GPUs (ТЗ 3.1: 2xH200), so the index is a build argument:
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 -t dxa-qc .
# Nothing else changes: the service picks CUDA -> MPS -> CPU by itself (DXAQC_DEVICE overrides).
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
COPY requirements.txt pyproject.toml ./
RUN pip install --index-url ${TORCH_INDEX} torch==2.14.0 \
    && grep -v '^torch==' requirements.txt > /tmp/req.txt \
    && pip install -r /tmp/req.txt

COPY dxaqc ./dxaqc
COPY scripts/predict.py ./scripts/predict.py
COPY models ./models
RUN pip install --no-deps -e .

ENTRYPOINT ["python", "scripts/predict.py", "--models", "/app/models"]
