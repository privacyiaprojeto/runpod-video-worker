ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG COMFYUI_REF=v0.27.0
ARG VIDEO_HELPER_REF=main

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_ROOT=/app \
    COMFYUI_ROOT=/opt/ComfyUI \
    WORKFLOW_ROOT=/app/workflows \
    RUNTIME_ROOT=/runpod-volume/privacy-wan-runtime \
    HF_HOME=/runpod-volume/huggingface \
    COMFYUI_START_LOCAL=true

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      ffmpeg \
      git \
      git-lfs \
      libgl1 \
      libglib2.0-0 \
      libsm6 \
      libxext6; \
    rm -rf /var/lib/apt/lists/*; \
    ffmpeg -version; \
    ffprobe -version

RUN set -eux; \
    git clone --filter=blob:none https://github.com/Comfy-Org/ComfyUI.git /opt/ComfyUI; \
    git -C /opt/ComfyUI checkout "${COMFYUI_REF}"; \
    python -m pip install --upgrade pip setuptools wheel; \
    grep -Ev '^(torch|torchvision|torchaudio)([<>=!~].*)?$' /opt/ComfyUI/requirements.txt > /tmp/comfyui-requirements-no-torch.txt; \
    python -m pip install --no-cache-dir -r /tmp/comfyui-requirements-no-torch.txt; \
    python -m pip install --no-cache-dir torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128; \
    python -c "import torch, torchvision, torchaudio; flags = getattr(torch._C, '_cuda_getArchFlags', lambda: '')() or ''; assert torch.__version__.startswith('2.8.0'), torch.__version__; assert torch.version.cuda == '12.8', torch.version.cuda; assert torchvision.__version__.startswith('0.23.0'), torchvision.__version__; assert torchaudio.__version__.startswith('2.8.0'), torchaudio.__version__; assert 'sm_120' in flags, flags; print('PYTORCH_BLACKWELL_RUNTIME_READY', torch.__version__, torch.version.cuda, flags)"

RUN set -eux; \
    mkdir -p /opt/ComfyUI/custom_nodes; \
    git clone --filter=blob:none https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
      /opt/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite; \
    git -C /opt/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite checkout "${VIDEO_HELPER_REF}"; \
    if [ -f /opt/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt ]; then \
      python -m pip install --no-cache-dir \
        -r /opt/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt; \
    fi

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py
COPY privacy_worker /app/privacy_worker
COPY workflows /app/workflows
COPY scripts /app/scripts
COPY custom_nodes/privacy_lora_attestation /opt/ComfyUI/custom_nodes/privacy_lora_attestation
COPY custom_nodes/privacy_motion_structure /opt/ComfyUI/custom_nodes/privacy_motion_structure
COPY extra_model_paths.yaml /app/extra_model_paths.yaml

RUN set -eux; \
    mkdir -p \
      /runpod-volume/models/diffusion_models \
      /runpod-volume/models/text_encoders \
      /runpod-volume/models/vae \
      /runpod-volume/models/clip_vision \
      /runpod-volume/models/loras \
      /runpod-volume/huggingface \
      /runpod-volume/privacy-wan-runtime/input \
      /runpod-volume/privacy-wan-runtime/output \
      /runpod-volume/privacy-wan-runtime/temp; \
    python -m compileall -q /app/handler.py /app/privacy_worker /opt/ComfyUI/custom_nodes/privacy_lora_attestation /opt/ComfyUI/custom_nodes/privacy_motion_structure; \
    python /app/scripts/validate_workflows.py; \
    ffmpeg -version >/dev/null; \
    ffprobe -version >/dev/null

CMD ["python", "-u", "/app/handler.py"]
