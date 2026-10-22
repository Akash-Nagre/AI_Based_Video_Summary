FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

WORKDIR /app

# FFmpeg handles audio/video conversion. These shared libraries are needed by
# OpenCV's Python wheel in a slim Linux image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY backend ./backend
COPY modules ./modules
COPY frontend ./frontend

RUN mkdir -p /app/data/jobs /app/data/cache /models

EXPOSE 8000
VOLUME ["/app/data", "/models"]

# Keep one worker: the current job registry is in memory.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
