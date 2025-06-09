FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgomp1 \
    liblapack3 \
    ffmpeg \
    libavcodec-extra \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local

RUN mkdir -p /app/vosk_models/ru


ENV VOSK_MODEL_PATH=/app/vosk_models/ru \
    PATH=/root/.local/bin:$PATH \
    PYTHONPATH=/app \
    DATABASE_URL=postgresql+asyncpg://admin:admin@postgres:5432/app_db

COPY app/ ./app/
COPY alembic/ ./alembic/
COPY templates/ ./templates/
COPY static/ ./static/
COPY alembic.ini .
COPY setup.py .
COPY requirements.txt .
COPY data_loader.py .
COPY historical_dataset.csv .
COPY app/ai/history/model/anomaly_detector /app/app/ai/history/model/anomaly_detector

CMD bash -c "alembic upgrade head && uvicorn app.main:fastapi_app --host 0.0.0.0 --port 8000 "