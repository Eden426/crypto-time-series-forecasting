FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    OMP_NUM_THREADS=1 \
    TF_NUM_INTRAOP_THREADS=1 \
    TF_NUM_INTEROP_THREADS=1 \
    HF_HOME=/tmp/huggingface \
    HF_REPO_ID=Eden-N/crypto-time-series-gru

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend /app/backend
COPY frontend/src/index.html frontend/src/style.css frontend/src/index.js /app/frontend/src/

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.src.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
