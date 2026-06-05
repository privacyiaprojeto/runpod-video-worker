FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/huggingface \
    INSIGHTFACE_HOME=/runpod-volume/insightface \
    MODEL_DIR=/runpod-volume/models \
    TMPDIR=/runpod-volume/tmp

RUN rm -rf /etc/apt/sources.list.d/* && \
    apt-get clean && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    git \
    curl \
    wget \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install -r /app/requirements.txt

COPY handler.py /app/handler.py

# Cria as pastas necessárias
RUN mkdir -p /runpod-volume/models /runpod-volume/tmp /runpod-volume/insightface/models/buffalo_l /runpod-volume/huggingface

# Faz o download do modelo de FaceSwap (inswapper_128.onnx)
RUN curl -L -o /runpod-volume/models/inswapper_128.onnx https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx

# Faz o download do modelo de detecção facial (buffalo_l) e extrai
RUN curl -L -o /runpod-volume/insightface/models/buffalo_l.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip && \
    unzip /runpod-volume/insightface/models/buffalo_l.zip -d /runpod-volume/insightface/models/buffalo_l && \
    rm /runpod-volume/insightface/models/buffalo_l.zip

CMD ["python3", "/app/handler.py"]