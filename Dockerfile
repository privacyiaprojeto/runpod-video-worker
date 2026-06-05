FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/huggingface \
    INSIGHTFACE_HOME=/runpod-volume/insightface \
    MODEL_DIR=/runpod-volume/models \
    TMPDIR=/runpod-volume/tmp

WORKDIR /app

RUN set -eux; \
    apt-get update -o Acquire::Retries=5; \
    apt-get install -y --no-install-recommends \
      ffmpeg \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
      git \
      curl \
      wget \
      unzip \
      ca-certificates; \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py

# Download dos Modelos de IA (O que o GPT havia esquecido nesta rodada)
RUN mkdir -p /runpod-volume/models /runpod-volume/tmp /runpod-volume/insightface/models/buffalo_l /runpod-volume/huggingface

RUN curl -L -o /runpod-volume/models/inswapper_128.onnx https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx

RUN curl -L -o /runpod-volume/insightface/models/buffalo_l.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip && \
    unzip /runpod-volume/insightface/models/buffalo_l.zip -d /runpod-volume/insightface/models/buffalo_l && \
    rm /runpod-volume/insightface/models/buffalo_l.zip

CMD ["python", "-u", "/app/handler.py"]