from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid

from pathlib import Path
from typing import Any, Iterable

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config as BotoConfig

from .config import Settings
from .errors import ModelStorageError
from .telemetry import log_event


CANONICAL_MODEL_REGISTRY_BUCKET = "ia-adulta-model-registry"
CANONICAL_MODEL_BUNDLE_VERSION = "wan-m4-v1"
MODEL_REGISTRY_MARKER_NAME = ".privacy-r2-model-registry-wan-m4-v1.json"

CANONICAL_MODEL_OBJECTS = (
    {
        "key": "models/diffusion_models/wan2.1_vace_14B_fp16.safetensors",
        "relative_path": "diffusion_models/wan2.1_vace_14B_fp16.safetensors",
        "size_bytes": 34675323640,
        "sha256": "f202a5c59b8a91ada1862c46a038214f1f7f216c61ec8350d25f69b919da4307",
    },
    {
        "key": "models/diffusion_models/wan2.1_i2v_480p_14B_fp16.safetensors",
        "relative_path": "diffusion_models/wan2.1_i2v_480p_14B_fp16.safetensors",
        "size_bytes": 32791377504,
        "sha256": "27988f6b510eb8d5fdd7485671b54897f8683f2bba7a772c5671be21d3491253",
    },
    {
        "key": "models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "relative_path": "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "size_bytes": 6735906897,
        "sha256": "c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68",
    },
    {
        "key": "models/clip_vision/clip_vision_h.safetensors",
        "relative_path": "clip_vision/clip_vision_h.safetensors",
        "size_bytes": 1264219396,
        "sha256": "64a7ef761bfccbadbaa3da77366aac4185a6c58fa5de5f589b42a65bcc21f161",
    },
    {
        "key": "models/vae/wan_2.1_vae.safetensors",
        "relative_path": "vae/wan_2.1_vae.safetensors",
        "size_bytes": 253815318,
        "sha256": "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b",
    },
)

CANONICAL_MODEL_BUNDLE_BYTES = sum(
    int(item["size_bytes"]) for item in CANONICAL_MODEL_OBJECTS
)


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _safe_relative_path(value: str) -> Path:
    relative = Path(str(value or "").strip())
    if (
        not str(relative)
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.parts[0] not in {
            "diffusion_models",
            "text_encoders",
            "vae",
            "clip_vision",
        }
    ):
        raise ModelStorageError(
            "Objeto do Model Registry possui destino local inválido.",
            details={"relative_path": str(value or "")},
        )
    return relative


def _validate_specs(specs: Iterable[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    normalized: list[dict[str, Any]] = []
    keys: set[str] = set()
    paths: set[str] = set()

    for raw in specs:
        key = str(raw.get("key") or "").strip().lstrip("/")
        relative = _safe_relative_path(str(raw.get("relative_path") or ""))
        sha256 = str(raw.get("sha256") or "").strip().lower()
        try:
            size_bytes = int(raw.get("size_bytes"))
        except (TypeError, ValueError) as exc:
            raise ModelStorageError("Tamanho inválido no Model Registry.") from exc

        if not key.startswith("models/"):
            raise ModelStorageError(
                "Objeto do Model Registry precisa permanecer sob models/.",
                details={"key": key},
            )
        if size_bytes <= 0:
            raise ModelStorageError(
                "Objeto do Model Registry precisa ter tamanho positivo.",
                details={"key": key},
            )
        if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
            raise ModelStorageError(
                "SHA-256 inválido no Model Registry.",
                details={"key": key},
            )
        if key in keys or str(relative) in paths:
            raise ModelStorageError("Model Registry contém objeto duplicado.")

        keys.add(key)
        paths.add(str(relative))
        normalized.append(
            {
                "key": key,
                "relative_path": str(relative),
                "size_bytes": size_bytes,
                "sha256": sha256,
            }
        )

    if not normalized:
        raise ModelStorageError("Model Registry não possui objetos.")
    return tuple(normalized)


def model_registry_r2_client(settings: Settings):
    if not settings.model_registry_r2_configured:
        raise ModelStorageError("R2 MASTER do Model Registry não está configurado.")
    if settings.model_registry_r2_bucket_name != CANONICAL_MODEL_REGISTRY_BUCKET:
        raise ModelStorageError(
            "Bucket do Model Registry diverge do MASTER canônico.",
            details={
                "expected": CANONICAL_MODEL_REGISTRY_BUCKET,
                "configured": settings.model_registry_r2_bucket_name,
            },
        )

    return boto3.client(
        "s3",
        endpoint_url=settings.model_registry_r2_endpoint_url,
        aws_access_key_id=settings.model_registry_r2_access_key_id,
        aws_secret_access_key=settings.model_registry_r2_secret_access_key,
        region_name="auto",
        config=BotoConfig(
            connect_timeout=15,
            read_timeout=120,
            retries={
                "total_max_attempts": 1,
                "mode": "standard",
            },
        ),
    )


def _marker_payload(specs: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    return {
        "version": CANONICAL_MODEL_BUNDLE_VERSION,
        "bucket": CANONICAL_MODEL_REGISTRY_BUCKET,
        "objects": [
            {
                "key": item["key"],
                "relative_path": item["relative_path"],
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
            }
            for item in specs
        ],
    }


def _marker_matches(marker_path: Path, specs: tuple[dict[str, Any], ...]) -> bool:
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload == _marker_payload(specs)


def _fast_local_bundle_ready(
    model_root: Path,
    marker_path: Path,
    specs: tuple[dict[str, Any], ...],
) -> bool:
    if not _marker_matches(marker_path, specs):
        return False
    for item in specs:
        path = model_root / item["relative_path"]
        try:
            if not path.is_file() or path.stat().st_size != int(item["size_bytes"]):
                return False
        except OSError:
            return False
    return True


def _assert_bootstrap_disk_capacity(
    model_root: Path,
    specs: tuple[dict[str, Any], ...],
    *,
    reserve_gb: float,
) -> dict[str, int | float]:
    model_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(model_root)
    bundle_bytes = sum(int(item["size_bytes"]) for item in specs)
    reserve_bytes = int(float(reserve_gb) * 1024**3)
    required_bytes = bundle_bytes + reserve_bytes

    if int(usage.free) < required_bytes:
        raise ModelStorageError(
            "Disco efêmero insuficiente para hidratar o R2 MASTER.",
            details={
                "free_bytes": int(usage.free),
                "bundle_bytes": bundle_bytes,
                "reserve_bytes": reserve_bytes,
                "required_bytes": required_bytes,
            },
        )

    return {
        "free_bytes": int(usage.free),
        "bundle_bytes": bundle_bytes,
        "reserve_bytes": reserve_bytes,
        "required_bytes": required_bytes,
    }


def _head_and_validate(client, bucket: str, item: dict[str, Any]) -> None:
    try:
        head = client.head_object(Bucket=bucket, Key=item["key"])
    except Exception as exc:
        raise ModelStorageError(
            "Falha ao validar objeto do R2 MASTER.",
            details={"key": item["key"]},
        ) from exc

    size = int(head.get("ContentLength") or 0)
    if size != int(item["size_bytes"]):
        raise ModelStorageError(
            "Tamanho do objeto no R2 MASTER diverge do lock canônico.",
            details={
                "key": item["key"],
                "expected_size": int(item["size_bytes"]),
                "observed_size": size,
            },
        )

    metadata = {
        str(key).strip().lower(): str(value).strip().lower()
        for key, value in (head.get("Metadata") or {}).items()
    }
    metadata_sha = (
        metadata.get("sha256")
        or metadata.get("sha-256")
        or metadata.get("checksum-sha256")
        or ""
    )
    if metadata_sha and metadata_sha != item["sha256"]:
        raise ModelStorageError(
            "SHA-256 metadata do R2 MASTER diverge do lock canônico.",
            details={"key": item["key"]},
        )


def _download_one(
    client,
    *,
    bucket: str,
    model_root: Path,
    item: dict[str, Any],
) -> tuple[Path, bool]:
    destination = model_root / item["relative_path"]
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_file() and destination.stat().st_size == int(item["size_bytes"]):
        if sha256_file(destination) == item["sha256"]:
            return destination, False

    if destination.exists():
        destination.unlink()

    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.part"
    )

    try:
        client.download_file(
            bucket,
            item["key"],
            str(temporary),
            Config=TransferConfig(
                multipart_threshold=64 * 1024 * 1024,
                multipart_chunksize=64 * 1024 * 1024,
                max_concurrency=4,
                use_threads=True,
                num_download_attempts=1,
            ),
        )

        if not temporary.is_file():
            raise ModelStorageError(
                "Download do R2 MASTER não materializou arquivo.",
                details={"key": item["key"]},
            )

        observed_size = temporary.stat().st_size
        if observed_size != int(item["size_bytes"]):
            raise ModelStorageError(
                "Download do R2 MASTER terminou com tamanho divergente.",
                details={
                    "key": item["key"],
                    "expected_size": int(item["size_bytes"]),
                    "observed_size": observed_size,
                },
            )

        observed_sha = sha256_file(temporary)
        if observed_sha != item["sha256"]:
            raise ModelStorageError(
                "Download do R2 MASTER terminou com SHA-256 divergente.",
                details={"key": item["key"]},
            )

        os.replace(temporary, destination)
        return destination, True
    finally:
        temporary.unlink(missing_ok=True)


def materialize_canonical_model_registry(
    settings: Settings,
    *,
    client=None,
    specs: Iterable[dict[str, Any]] = CANONICAL_MODEL_OBJECTS,
) -> dict[str, Any]:
    normalized = _validate_specs(specs)

    if settings.model_registry_r2_bucket_name != CANONICAL_MODEL_REGISTRY_BUCKET:
        raise ModelStorageError(
            "Bucket do Model Registry diverge do MASTER canônico.",
            details={
                "expected": CANONICAL_MODEL_REGISTRY_BUCKET,
                "configured": settings.model_registry_r2_bucket_name,
            },
        )
    if not settings.model_registry_r2_configured:
        raise ModelStorageError("R2 MASTER do Model Registry não está configurado.")

    model_root = settings.model_root.resolve(strict=False)
    runtime_root = settings.runtime_root.resolve(strict=False)
    if _inside(model_root, runtime_root) or _inside(runtime_root, model_root):
        raise ModelStorageError("MODEL_ROOT e RUNTIME_ROOT precisam ser raízes separadas.")

    if os.name != "nt":
        tmp_root = Path("/tmp").resolve(strict=False)
        if not _inside(model_root, tmp_root) or not _inside(runtime_root, tmp_root):
            raise ModelStorageError(
                "MODEL_SOURCE_MODE=r2_registry exige MODEL_ROOT e RUNTIME_ROOT efêmeros sob /tmp."
            )

    marker_path = settings.model_root / MODEL_REGISTRY_MARKER_NAME

    if _fast_local_bundle_ready(settings.model_root, marker_path, normalized):
        return {
            "version": CANONICAL_MODEL_BUNDLE_VERSION,
            "bucket": CANONICAL_MODEL_REGISTRY_BUCKET,
            "object_count": len(normalized),
            "bundle_bytes": sum(int(item["size_bytes"]) for item in normalized),
            "downloaded_count": 0,
            "reused_count": len(normalized),
            "marker_path": str(marker_path),
        }

    disk = _assert_bootstrap_disk_capacity(
        settings.model_root,
        normalized,
        reserve_gb=settings.ephemeral_min_free_gb,
    )
    s3 = client or model_registry_r2_client(settings)

    downloaded = 0
    reused = 0

    for item in normalized:
        _head_and_validate(s3, CANONICAL_MODEL_REGISTRY_BUCKET, item)
        _, changed = _download_one(
            s3,
            bucket=CANONICAL_MODEL_REGISTRY_BUCKET,
            model_root=settings.model_root,
            item=item,
        )
        if changed:
            downloaded += 1
        else:
            reused += 1

    marker = _marker_payload(normalized)
    temporary_marker = marker_path.with_name(
        f".{marker_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary_marker.write_text(
            json.dumps(marker, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_marker, marker_path)
    finally:
        temporary_marker.unlink(missing_ok=True)

    log_event(
        "canonical_model_registry_materialized",
        model_registry_version=CANONICAL_MODEL_BUNDLE_VERSION,
        model_registry_bucket=CANONICAL_MODEL_REGISTRY_BUCKET,
        object_count=len(normalized),
        downloaded_count=downloaded,
        reused_count=reused,
        bundle_bytes=disk["bundle_bytes"],
        free_bytes_before=disk["free_bytes"],
    )

    return {
        "version": CANONICAL_MODEL_BUNDLE_VERSION,
        "bucket": CANONICAL_MODEL_REGISTRY_BUCKET,
        "object_count": len(normalized),
        "bundle_bytes": disk["bundle_bytes"],
        "downloaded_count": downloaded,
        "reused_count": reused,
        "marker_path": str(marker_path),
    }
