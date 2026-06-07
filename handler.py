import base64
import json
import mimetypes
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import boto3
import cv2
import numpy as np
import requests
import runpod
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# -----------------------------
# Environment / defaults
# -----------------------------
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/runpod-volume/models"))
TMP_ROOT = Path(os.getenv("TMPDIR", "/runpod-volume/tmp"))
INSIGHTFACE_HOME = Path(os.getenv("INSIGHTFACE_HOME", "/runpod-volume/insightface"))

SWAPPER_MODEL_PATH = Path(os.getenv("SWAPPER_MODEL_PATH", str(MODEL_DIR / "inswapper_128.onnx")))
SWAPPER_MODEL_URL = os.getenv("SWAPPER_MODEL_URL", "").strip()

FACE_DET_SIZE = int(os.getenv("FACE_DET_SIZE", "640"))
FACE_DET_THRESH = float(os.getenv("FACE_DET_THRESH", "0.5"))
DOWNLOAD_TIMEOUT_SECONDS = int(os.getenv("DOWNLOAD_TIMEOUT_SECONDS", "180"))
MAX_DOWNLOAD_MB = int(os.getenv("MAX_DOWNLOAD_MB", "300"))
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "0"))  # 0 = no hard frame limit
DEFAULT_MAX_SECONDS = float(os.getenv("DEFAULT_MAX_SECONDS", "12"))
PRESERVE_AUDIO = os.getenv("PRESERVE_AUDIO", "true").lower() == "true"
SWAP_ALL_FACES = os.getenv("SWAP_ALL_FACES", "false").lower() == "true"
RETURN_BASE64_DEFAULT = os.getenv("RETURN_BASE64_DEFAULT", "true").lower() == "true"
MAX_BASE64_RETURN_MB = int(os.getenv("MAX_BASE64_RETURN_MB", "80"))
JPEG_SOURCE_QUALITY = int(os.getenv("JPEG_SOURCE_QUALITY", "95"))
VIDEO_CRF = os.getenv("VIDEO_CRF", "20").strip() or "20"
VIDEO_PRESET = os.getenv("VIDEO_PRESET", "veryfast").strip() or "veryfast"
AUDIO_BITRATE = os.getenv("AUDIO_BITRATE", "128k").strip() or "128k"


# R2 direct upload is optional. If configured, large MP4s can be returned as URL instead of base64.
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL", "").strip()
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "").strip()
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "").strip()
R2_PUBLIC_BASE_URL = os.getenv("R2_PUBLIC_BASE_URL", "").strip().rstrip("/")
R2_PREFIX = os.getenv("R2_PREFIX", "faceswap/tmp").strip().strip("/")

_FACE_APP = None
_SWAPPER = None


# -----------------------------
# Utility helpers
# -----------------------------
def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "sim"}


def _is_http_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def _ensure_dirs() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    INSIGHTFACE_HOME.mkdir(parents=True, exist_ok=True)


def _download_file(url: str, out_path: Path, label: str) -> Path:
    if not _is_http_url(url):
        raise ValueError(f"{label} precisa ser uma URL http(s) válida.")

    max_bytes = MAX_DOWNLOAD_MB * 1024 * 1024
    downloaded = 0

    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        response.raise_for_status()
        content_length = int(response.headers.get("content-length") or 0)
        if content_length > max_bytes:
            raise ValueError(f"{label} excede o limite de {MAX_DOWNLOAD_MB}MB.")

        with out_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                downloaded += len(chunk)
                if downloaded > max_bytes:
                    raise ValueError(f"{label} excede o limite de {MAX_DOWNLOAD_MB}MB durante download.")
                handle.write(chunk)

    if out_path.stat().st_size <= 0:
        raise ValueError(f"Download vazio para {label}.")

    return out_path


def _download_model_if_needed() -> None:
    if SWAPPER_MODEL_PATH.exists() and SWAPPER_MODEL_PATH.stat().st_size > 0:
        return

    if not SWAPPER_MODEL_URL:
        raise RuntimeError(
            "Modelo FaceSwap não encontrado. Coloque inswapper_128.onnx em "
            f"{SWAPPER_MODEL_PATH} ou defina SWAPPER_MODEL_URL."
        )

    print(f"[BOOT] baixando swapper model para {SWAPPER_MODEL_PATH}")
    _download_file(SWAPPER_MODEL_URL, SWAPPER_MODEL_PATH, "SWAPPER_MODEL_URL")


def _largest_face(faces):
    if not faces:
        return None

    def area(face):
        x1, y1, x2, y2 = face.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    return max(faces, key=area)


def _encode_base64(path: Path) -> str:
    with path.open("rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")


def _r2_is_configured() -> bool:
    return all([R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME, R2_PUBLIC_BASE_URL])


def _upload_to_r2(path: Path, content_type: str = "video/mp4") -> str:
    if not _r2_is_configured():
        raise RuntimeError("R2 não configurado no worker.")

    key = f"{R2_PREFIX}/{uuid.uuid4()}.mp4"
    client = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
    )

    with path.open("rb") as body:
        client.put_object(
            Bucket=R2_BUCKET_NAME,
            Key=key,
            Body=body,
            ContentType=content_type,
            CacheControl="public, max-age=31536000, immutable",
        )

    return f"{R2_PUBLIC_BASE_URL}/{key}"


def _get_ffmpeg_exe() -> Optional[str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as error:
        print(f"[WARN] imageio_ffmpeg indisponível: {error}")
        return None


def _run_command(command) -> Tuple[bool, str]:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    return result.returncode == 0, output[-2000:]


def _run_ffmpeg_finalize(raw_video_path: Path, source_video_path: Path, final_video_path: Path, fps: float) -> bool:
    """
    Reempacota/reencoda o vídeo bruto do OpenCV para um MP4 realmente tocável no navegador.

    O OpenCV VideoWriter pode gerar arquivos .mp4 que têm frames, mas ficam com metadata/duração
    ruim em alguns browsers/R2. Por isso o worker sempre tenta normalizar com ffmpeg:
    - H.264
    - yuv420p
    - movflags +faststart
    - PTS/fps coerentes
    - áudio opcional do vídeo base
    """
    ffmpeg = _get_ffmpeg_exe()
    if not ffmpeg:
        print("[WARN] ffmpeg não encontrado; usando vídeo bruto do OpenCV.")
        return False

    fps_value = max(float(fps or 24.0), 1.0)
    fps_text = f"{fps_value:.3f}"

    base_command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-fflags",
        "+genpts",
        "-r",
        fps_text,
        "-i",
        str(raw_video_path),
        "-i",
        str(source_video_path),
        "-map",
        "0:v:0",
    ]

    if PRESERVE_AUDIO:
        base_command += ["-map", "1:a:0?"]

    h264_command = base_command + [
        "-c:v",
        "libx264",
        "-preset",
        VIDEO_PRESET,
        "-crf",
        VIDEO_CRF,
        "-pix_fmt",
        "yuv420p",
        "-r",
        fps_text,
        "-movflags",
        "+faststart",
    ]

    if PRESERVE_AUDIO:
        h264_command += ["-c:a", "aac", "-b:a", AUDIO_BITRATE, "-shortest"]
    else:
        h264_command += ["-an"]

    h264_command += [str(final_video_path)]

    ok, output = _run_command(h264_command)
    if ok and final_video_path.exists() and final_video_path.stat().st_size > 0:
        return True

    print(f"[WARN] ffmpeg h264 falhou; tentando fallback mpeg4. stderr={output}")

    fallback_command = base_command + [
        "-c:v",
        "mpeg4",
        "-q:v",
        "4",
        "-pix_fmt",
        "yuv420p",
        "-r",
        fps_text,
        "-movflags",
        "+faststart",
    ]

    if PRESERVE_AUDIO:
        fallback_command += ["-c:a", "aac", "-b:a", AUDIO_BITRATE, "-shortest"]
    else:
        fallback_command += ["-an"]

    fallback_command += [str(final_video_path)]

    ok, output = _run_command(fallback_command)
    if not ok:
        print(f"[WARN] ffmpeg fallback falhou; usando vídeo bruto. stderr={output}")
        return False

    return final_video_path.exists() and final_video_path.stat().st_size > 0


def _probe_video_with_cv(video_path: Path) -> Dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Vídeo final inválido: OpenCV não conseguiu abrir {video_path.name}.")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()

    duration_seconds = frames / fps if fps > 0 and frames > 0 else 0

    if frames <= 0 or duration_seconds <= 0:
        raise RuntimeError(
            f"Vídeo final com duração inválida. frames={frames}, fps={fps:.3f}, duration={duration_seconds:.3f}s"
        )

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frames,
        "duration_seconds": duration_seconds,
    }


def _create_intermediate_writer(work_dir: Path, fps: float, width: int, height: int):
    """
    Usa AVI/MJPG como intermediário preferencial.
    É mais confiável que escrever MP4 direto pelo OpenCV e depois facilita a normalização via ffmpeg.
    """
    candidates = [
        (work_dir / "faceswap_raw.avi", "MJPG"),
        (work_dir / "faceswap_raw.mp4", "mp4v"),
    ]

    for path, codec in candidates:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if writer.isOpened():
            print(f"[JOB] intermediate writer codec={codec} path={path.name}")
            return writer, path, codec
        writer.release()

    raise RuntimeError("Não foi possível inicializar VideoWriter intermediário.")


# -----------------------------
# Model loading
# -----------------------------
def _load_models() -> Tuple[FaceAnalysis, Any]:
    global _FACE_APP, _SWAPPER

    if _FACE_APP is not None and _SWAPPER is not None:
        return _FACE_APP, _SWAPPER

    _ensure_dirs()
    _download_model_if_needed()

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    print("[BOOT] carregando FaceAnalysis buffalo_l")
    app = FaceAnalysis(
        name="buffalo_l",
        root=str(INSIGHTFACE_HOME),
        providers=providers,
    )
    app.prepare(ctx_id=0, det_size=(FACE_DET_SIZE, FACE_DET_SIZE), det_thresh=FACE_DET_THRESH)

    print(f"[BOOT] carregando swapper model path={SWAPPER_MODEL_PATH}")
    swapper = get_model(str(SWAPPER_MODEL_PATH), providers=providers)

    _FACE_APP = app
    _SWAPPER = swapper

    print("[BOOT] FaceSwap worker pronto")
    return _FACE_APP, _SWAPPER


# Lazy warmup at import time. If model is missing, handler returns clear error instead of crashing the container.
try:
    _load_models()
except Exception as boot_error:
    print(f"[BOOT][WARN] carregamento inicial adiado/falhou: {boot_error}")


# -----------------------------
# Core faceswap
# -----------------------------
def _faceswap_video(source_image_path: Path, target_video_path: Path, work_dir: Path, input_payload: Dict[str, Any]) -> Path:
    started_at = _now_ms()
    app, swapper = _load_models()

    source_image = cv2.imread(str(source_image_path))
    if source_image is None:
        raise ValueError("Não foi possível abrir source_image_url como imagem.")

    source_faces = app.get(source_image)
    source_face = _largest_face(source_faces)
    if source_face is None:
        raise ValueError("Nenhum rosto encontrado na imagem de origem.")

    capture = cv2.VideoCapture(str(target_video_path))
    if not capture.isOpened():
        raise ValueError("Não foi possível abrir target_video_url como vídeo.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError("Vídeo base sem largura/altura válida.")

    max_seconds = float(input_payload.get("max_seconds") or DEFAULT_MAX_SECONDS)
    frame_limit_by_seconds = int(max_seconds * fps) if max_seconds > 0 else 0
    frame_limit_candidates = [value for value in [MAX_FRAMES, frame_limit_by_seconds, total_frames] if value and value > 0]
    frame_limit = min(frame_limit_candidates) if frame_limit_candidates else 0

    final_output_path = work_dir / "faceswap_final.mp4"

    try:
        writer, raw_output_path, raw_codec = _create_intermediate_writer(work_dir, fps, width, height)
    except Exception:
        capture.release()
        raise

    processed_frames = 0
    swapped_frames = 0

    print(
        f"[JOB] faceswap start width={width} height={height} fps={fps:.2f} "
        f"total_frames={total_frames} frame_limit={frame_limit or 'none'} swap_all={SWAP_ALL_FACES}"
    )

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_limit and processed_frames >= frame_limit:
            break

        try:
            faces = app.get(frame)
            if faces:
                targets = faces if SWAP_ALL_FACES else [_largest_face(faces)]
                for target_face in targets:
                    if target_face is None:
                        continue
                    frame = swapper.get(frame, target_face, source_face, paste_back=True)
                swapped_frames += 1
        except Exception as frame_error:
            print(f"[WARN] falha em frame={processed_frames}: {frame_error}")

        writer.write(frame)
        processed_frames += 1

        if processed_frames % 30 == 0:
            print(f"[JOB] progress frames={processed_frames} swapped_frames={swapped_frames}")

    capture.release()
    writer.release()

    if processed_frames == 0:
        raise ValueError("Nenhum frame processado no vídeo base.")

    if not raw_output_path.exists() or raw_output_path.stat().st_size <= 0:
        raise RuntimeError("Arquivo MP4 bruto não foi gerado.")

    if _run_ffmpeg_finalize(raw_output_path, target_video_path, final_output_path, fps):
        output_path = final_output_path
    else:
        output_path = raw_output_path

    final_probe = _probe_video_with_cv(output_path)
    elapsed_ms = _now_ms() - started_at
    print(
        f"[JOB] faceswap completed elapsed_ms={elapsed_ms} frames={processed_frames} "
        f"swapped_frames={swapped_frames} output_mb={output_path.stat().st_size / 1024 / 1024:.2f} "
        f"final_fps={final_probe['fps']:.2f} final_frames={final_probe['frames']} "
        f"duration={final_probe['duration_seconds']:.2f}s"
    )

    return output_path


# -----------------------------
# RunPod entrypoint
# -----------------------------
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    job_input = event.get("input") or {}
    request_id = event.get("id") or str(uuid.uuid4())
    started_at = _now_ms()

    source_image_url = str(job_input.get("source_image_url") or job_input.get("reference_image_url") or "").strip()
    target_video_url = str(job_input.get("target_video_url") or job_input.get("video_url") or "").strip()
    safety_mode = str(job_input.get("safety_mode") or "").strip()
    consent_confirmed = _safe_bool(job_input.get("consent_confirmed"), default=False)
    return_base64 = _safe_bool(job_input.get("return_base64"), default=RETURN_BASE64_DEFAULT)
    force_r2 = _safe_bool(job_input.get("force_r2"), default=False)

    if not source_image_url:
        raise ValueError("source_image_url obrigatório")

    if not target_video_url:
        raise ValueError("target_video_url obrigatório")

    if safety_mode != "licensed_or_consented_assets_only" and not consent_confirmed:
        raise ValueError("safety_mode=licensed_or_consented_assets_only ou consent_confirmed=true obrigatório")

    with tempfile.TemporaryDirectory(dir=str(TMP_ROOT), prefix=f"faceswap_{request_id}_") as tmp:
        work_dir = Path(tmp)
        source_path = work_dir / "source.jpg"
        target_path = work_dir / "target.mp4"

        print(f"[JOB] request_id={request_id} download start")
        _download_file(source_image_url, source_path, "source_image_url")
        _download_file(target_video_url, target_path, "target_video_url")

        output_path = _faceswap_video(source_path, target_path, work_dir, job_input)
        output_size_mb = output_path.stat().st_size / 1024 / 1024

        if force_r2 or (output_size_mb > MAX_BASE64_RETURN_MB and _r2_is_configured()):
            output_url = _upload_to_r2(output_path, "video/mp4")
            elapsed_ms = _now_ms() - started_at
            return {
                "video_url": output_url,
                "url": output_url,
                "mime_type": "video/mp4",
                "extension": "mp4",
                "size_bytes": output_path.stat().st_size,
                "elapsed_ms": elapsed_ms,
                "request_id": request_id,
            }

        if not return_base64 and _r2_is_configured():
            output_url = _upload_to_r2(output_path, "video/mp4")
            elapsed_ms = _now_ms() - started_at
            return {
                "video_url": output_url,
                "url": output_url,
                "mime_type": "video/mp4",
                "extension": "mp4",
                "size_bytes": output_path.stat().st_size,
                "elapsed_ms": elapsed_ms,
                "request_id": request_id,
            }

        if output_size_mb > MAX_BASE64_RETURN_MB:
            raise ValueError(
                f"Vídeo final tem {output_size_mb:.2f}MB e excede MAX_BASE64_RETURN_MB={MAX_BASE64_RETURN_MB}. "
                "Configure R2 no worker ou reduza max_seconds."
            )

        video_base64 = _encode_base64(output_path)
        elapsed_ms = _now_ms() - started_at

        return {
            "video_base64": video_base64,
            "mime_type": "video/mp4",
            "extension": "mp4",
            "size_bytes": output_path.stat().st_size,
            "elapsed_ms": elapsed_ms,
            "request_id": request_id,
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
