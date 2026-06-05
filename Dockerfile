FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/huggingface \
    INSIGHTFACE_HOME=/runpod-volume/insightface \
    MODEL_DIR=/runpod-volume/models \
    TMPDIR=/runpod-volume/tmp

WORKDIR /app

# IMPORTANTE:
# Este worker NÃO usa apt-get.
# Motivo: as imagens CUDA/RunPod podem vir com fontes APT quebradas no buildx/GitHub Actions,
# causando exit code 100 antes mesmo do Python iniciar.
# Usamos apenas Python/pip e imageio-ffmpeg para fornecer o binário ffmpeg.

COPY requirements.txt /app/requirements.txt

RUN set -eux; \
    python -m pip install --upgrade pip setuptools wheel; \
    python -m pip install --no-cache-dir -r /app/requirements.txt; \
    python -m pip install --no-cache-dir imageio-ffmpeg; \
    python - <<'PY'
import os
import shutil
from pathlib import Path
import imageio_ffmpeg

ffmpeg_src = imageio_ffmpeg.get_ffmpeg_exe()
ffmpeg_dst = Path('/usr/local/bin/ffmpeg')
ffprobe_dst = Path('/usr/local/bin/ffprobe')
ffmpeg_dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(ffmpeg_src, ffmpeg_dst)
os.chmod(ffmpeg_dst, 0o755)

# Alguns fluxos procuram ffprobe. O imageio-ffmpeg não entrega ffprobe,
# então deixamos um fallback que evita erro de "command not found".
if not ffprobe_dst.exists():
    ffprobe_dst.symlink_to(ffmpeg_dst)

print('ffmpeg instalado em:', ffmpeg_dst)
PY

COPY handler.py /app/handler.py

RUN set -eux; \
    mkdir -p /runpod-volume/models \
             /runpod-volume/tmp \
             /runpod-volume/insightface/models \
             /runpod-volume/huggingface; \
    python - <<'PY'
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

model_dir = Path('/runpod-volume/models')
insightface_models = Path('/runpod-volume/insightface/models')
model_dir.mkdir(parents=True, exist_ok=True)
insightface_models.mkdir(parents=True, exist_ok=True)

inswapper_path = model_dir / 'inswapper_128.onnx'
if not inswapper_path.exists() or inswapper_path.stat().st_size < 1024 * 1024:
    print('Baixando inswapper_128.onnx...')
    urlretrieve(
        'https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx',
        inswapper_path,
    )

buffalo_dir = insightface_models / 'buffalo_l'
buffalo_zip = insightface_models / 'buffalo_l.zip'
if not buffalo_dir.exists() or not any(buffalo_dir.glob('*.onnx')):
    print('Baixando buffalo_l.zip...')
    urlretrieve(
        'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip',
        buffalo_zip,
    )
    buffalo_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(buffalo_zip, 'r') as zip_ref:
        zip_ref.extractall(buffalo_dir)
    buffalo_zip.unlink(missing_ok=True)

print('Modelos preparados.')
PY

CMD ["python", "-u", "/app/handler.py"]
