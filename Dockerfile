FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DEVICE=cpu \
    MODEL_USE_FP16=false \
    HF_HOME=/app/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt /app/requirements-docker.txt

RUN pip install --upgrade pip \
    && pip install -r requirements-docker.txt

COPY src /app/src

RUN mkdir -p /app/data/database /app/.cache/huggingface

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]