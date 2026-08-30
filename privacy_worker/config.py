from __future__ import annotations

import os
import math
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


def _float(name: str, default: float, *, minimum: float = 0.0) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} precisa ser numérico.") from exc
    if not math.isfinite(parsed) or parsed < minimum:
        raise ValueError(f"{name} precisa ser maior ou igual a {minimum}.")
    return parsed


def _storage_path(name: str, *, legacy: str, cached: str) -> Path:
    configured = os.getenv(name)
    configured_path = Path(configured) if configured else None
    mode = os.getenv("MODEL_SOURCE_MODE", "network_volume").strip().lower()
    if mode in {"cached_model", "r2_registry"} and (
        configured_path is None or configured_path == Path(legacy)
    ):
        return Path(cached)
    return configured_path or Path(legacy)


@dataclass(frozen=True)
class Settings:
    app_root: Path = Path(os.getenv("APP_ROOT", "/app"))
    comfyui_root: Path = Path(os.getenv("COMFYUI_ROOT", "/opt/ComfyUI"))
    workflow_root: Path = Path(os.getenv("WORKFLOW_ROOT", "/app/workflows"))
    model_source_mode: str = os.getenv("MODEL_SOURCE_MODE", "network_volume").strip().lower()
    runtime_root: Path = _storage_path(
        "RUNTIME_ROOT",
        legacy="/runpod-volume/privacy-wan-runtime",
        cached="/tmp/privacy-wan-runtime",
    )
    model_root: Path = _storage_path(
        "MODEL_ROOT",
        legacy="/runpod-volume/models",
        cached="/tmp/privacy-models",
    )
    cached_model_id: str = os.getenv("CACHED_MODEL_ID", "").strip()
    cached_model_revision: str = os.getenv("CACHED_MODEL_REVISION", "").strip()
    cached_model_cache_root: Path = Path(
        os.getenv("CACHED_MODEL_CACHE_ROOT", "/runpod-volume/huggingface-cache/hub")
    )
    ephemeral_min_free_gb: float = _float("EPHEMERAL_MIN_FREE_GB", 20.0)
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

    # M4 canonical model registry — independent from customer/media R2.
    model_registry_r2_endpoint_url: str = os.getenv(
        "MODEL_REGISTRY_R2_ENDPOINT_URL", ""
    ).strip()
    model_registry_r2_access_key_id: str = os.getenv(
        "MODEL_REGISTRY_R2_ACCESS_KEY_ID", ""
    ).strip()
    model_registry_r2_secret_access_key: str = os.getenv(
        "MODEL_REGISTRY_R2_SECRET_ACCESS_KEY", ""
    ).strip()
    model_registry_r2_bucket_name: str = os.getenv(
        "MODEL_REGISTRY_R2_BUCKET_NAME", "ia-adulta-model-registry"
    ).strip()
    identity_one_shot_lock_backend: str = os.getenv(
        "IDENTITY_ONE_SHOT_LOCK_BACKEND", "filesystem"
    ).strip().lower()
    identity_one_shot_lock_prefix: str = os.getenv(
        "IDENTITY_ONE_SHOT_LOCK_PREFIX",
        os.getenv("R2_PREFIX", "wan/private-tmp"),
    ).strip().strip("/")
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

    @property
    def model_registry_r2_configured(self) -> bool:
        return all(
            [
                self.model_registry_r2_endpoint_url,
                self.model_registry_r2_access_key_id,
                self.model_registry_r2_secret_access_key,
                self.model_registry_r2_bucket_name,
            ]
        )

    def ensure_runtime_dirs(self) -> None:
        for path in (self.runtime_root, self.input_dir, self.output_dir, self.temp_dir):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
