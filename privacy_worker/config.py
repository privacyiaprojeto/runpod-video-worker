from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        return max(int(os.getenv(name, str(default))), minimum)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    app_root: Path = Path(os.getenv("APP_ROOT", "/app"))
    comfyui_root: Path = Path(os.getenv("COMFYUI_ROOT", "/opt/ComfyUI"))
    workflow_root: Path = Path(os.getenv("WORKFLOW_ROOT", "/app/workflows"))
    runtime_root: Path = Path(os.getenv("RUNTIME_ROOT", "/runpod-volume/privacy-wan-runtime"))
    model_root: Path = Path(os.getenv("MODEL_ROOT", "/runpod-volume/models"))
    comfyui_host: str = os.getenv("COMFYUI_HOST", "127.0.0.1")
    comfyui_port: int = _int("COMFYUI_PORT", 8188)
    comfyui_start_local: bool = _bool("COMFYUI_START_LOCAL", True)
    comfyui_start_timeout_seconds: int = _int("COMFYUI_START_TIMEOUT_SECONDS", 180)
    comfyui_job_timeout_seconds: int = _int("COMFYUI_JOB_TIMEOUT_SECONDS", 3900)
    comfyui_poll_interval_seconds: int = _int("COMFYUI_POLL_INTERVAL_SECONDS", 3)
    download_timeout_seconds: int = _int("DOWNLOAD_TIMEOUT_SECONDS", 180)
    max_image_download_mb: int = _int("MAX_IMAGE_DOWNLOAD_MB", 80)
    max_video_download_mb: int = _int("MAX_VIDEO_DOWNLOAD_MB", 2048)
    max_output_mb: int = _int("MAX_OUTPUT_MB", 2048)
    max_base64_return_mb: int = _int("MAX_BASE64_RETURN_MB", 80)
    allow_private_download_hosts: bool = _bool("ALLOW_PRIVATE_DOWNLOAD_HOSTS", False)
    skip_model_validation: bool = _bool("SKIP_MODEL_VALIDATION", False)
    media_allowed_hosts: tuple[str, ...] = tuple(
        item.strip().lower()
        for item in os.getenv("MEDIA_ALLOWED_HOSTS", "").split(",")
        if item.strip()
    )
    output_mode: str = os.getenv("OUTPUT_MODE", "auto").strip().lower()
    r2_endpoint_url: str = os.getenv("R2_ENDPOINT_URL", "").strip()
    r2_access_key_id: str = os.getenv("R2_ACCESS_KEY_ID", "").strip()
    r2_secret_access_key: str = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
    r2_bucket_name: str = os.getenv("R2_BUCKET_NAME", "").strip()
    r2_prefix: str = os.getenv("R2_PREFIX", "wan/private-tmp").strip().strip("/")
    r2_signed_url_ttl_seconds: int = _int("R2_SIGNED_URL_TTL_SECONDS", 900)
    i2v_model_name: str = os.getenv("WAN_I2V_MODEL_NAME", "wan2.1_i2v_480p_14B_fp16.safetensors")
    v2v_model_name: str = os.getenv("WAN_V2V_MODEL_NAME", "wan2.1_vace_14B_fp16.safetensors")
    text_encoder_name: str = os.getenv("WAN_TEXT_ENCODER_NAME", "umt5_xxl_fp8_e4m3fn_scaled.safetensors")
    vae_name: str = os.getenv("WAN_VAE_NAME", "wan_2.1_vae.safetensors")
    clip_vision_name: str = os.getenv("WAN_CLIP_VISION_NAME", "clip_vision_h.safetensors")

    @property
    def comfyui_base_url(self) -> str:
        return f"http://{self.comfyui_host}:{self.comfyui_port}"

    @property
    def input_dir(self) -> Path:
        return self.runtime_root / "input"

    @property
    def output_dir(self) -> Path:
        return self.runtime_root / "output"

    @property
    def temp_dir(self) -> Path:
        return self.runtime_root / "temp"

    @property
    def r2_configured(self) -> bool:
        return all(
            [
                self.r2_endpoint_url,
                self.r2_access_key_id,
                self.r2_secret_access_key,
                self.r2_bucket_name,
            ]
        )

    def ensure_runtime_dirs(self) -> None:
        for path in (self.runtime_root, self.input_dir, self.output_dir, self.temp_dir):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
