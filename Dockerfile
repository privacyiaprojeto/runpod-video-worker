FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/huggingface \
    INSIGHTFACE_HOME=/runpod-volume/insightface \
    MODEL_DIR=/runpod-volume/models \
    TMPDIR=/runpod-volume/tmp

# Hotfix: do not mask apt-get update failures with "|| true".
# The CUDA runtime image already contains the CUDA runtime libs we need, so we remove
# NVIDIA/CUDA apt source files that commonly break apt in GitHub Actions runners.
RUN set -eux; \
    rm -f /etc/apt/sources.list.d/cuda*.list \
          /etc/apt/sources.list.d/nvidia*.list \
          /etc/apt/sources.list.d/nvidia-ml.list || true; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-dev \
      build-essential \
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

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install -r /app/requirements.txt

COPY handler.py /app/handler.py

RUN mkdir -p /runpod-volume/models /runpod-volume/tmp /runpod-volume/insightface /runpod-volume/huggingface

CMD ["python3", "/app/handler.py"]
