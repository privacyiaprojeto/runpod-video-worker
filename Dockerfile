FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/huggingface \
    INSIGHTFACE_HOME=/runpod-volume/insightface \
    MODEL_DIR=/runpod-volume/models \
    TMPDIR=/runpod-volume/tmp

# Hotfix v2:
# Some NVIDIA/CUDA base images now ship apt sources as .sources, not only .list.
# If these stale NVIDIA/CUDA source files remain, apt-get update/install can fail with exit code 100.
# We only need the CUDA runtime libraries already baked into the base image, so we remove
# external CUDA/NVIDIA apt source definitions before installing OS packages.
RUN set -eux; \
    find /etc/apt/sources.list.d -maxdepth 1 -type f \
      \( -iname '*cuda*' -o -iname '*nvidia*' \) \
      -print -delete || true; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*; \
    apt-get update -o Acquire::Retries=5; \
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
