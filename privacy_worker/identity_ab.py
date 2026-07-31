from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3

from .lora_namespace import (
    CONVERSION_VERSION,
    convert_diffsynth_peft_lora,
    model_layout_sha256,
    read_conversion_attestation,
    sha256_file,
    write_conversion_attestation,
)

from .config import Settings
from .errors import ContractError, DownloadError, LoraCompatibilityError, LoraNotAppliedError

CONTRACT_VERSION = "privacy-identity-neutral-ab-v1"
WORKFLOW_ID = "wan-2.1-v2v-identity-ab-v1"
NEUTRAL_BUCKET = "privacy-media"
NEUTRAL_KEY = "qa-assets/neutral-motion-01.mp4"


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _sha(value: Any) -> str:
    value = _text(value).lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ContractError("Checksum SHA-256 inválido no contrato A/B.")
    return value


def _private_ref(value: Any) -> dict[str, str]:
    item = value if isinstance(value, dict) else {}
    bucket, key = _text(item.get("bucket")), _text(item.get("key"))
    if not bucket or not key or key.startswith("/") or "://" in bucket or "://" in key:
        raise ContractError("Referência privada inválida no contrato A/B.")
    return {"bucket": bucket, "key": key, "sha256": _sha(item.get("sha256"))}


def _kyc_reference_ref(value: Any, actor_profile_id: str) -> dict[str, str]:
    item = value if isinstance(value, dict) else {}
    ref = _private_ref(item)
    system_tag = _text(item.get("system_tag")).lower()
    if system_tag != "face_front":
        raise ContractError("A referência KYC do ramo B precisa usar system_tag face_front.")
    if ref["bucket"] != NEUTRAL_BUCKET:
        raise ContractError("A referência KYC do ramo B precisa permanecer no bucket privado aprovado.")
    normalized_key = ref["key"].lower()
    actor_scope = f"/actor-{actor_profile_id.lower()}/"
    if not normalized_key.startswith("vault/actor-mapping/") or actor_scope not in f"/{normalized_key}":
        raise ContractError("A referência KYC do ramo B não pertence ao cofre privado deste ator.")
    if Path(ref["key"]).suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise ContractError("Formato da referência KYC frontal não suportado.")
    return {
        **ref,
        "system_tag": "face_front",
        "asset_id": _text(item.get("asset_id")),
    }


@dataclass(frozen=True)
class IdentityAbRequest:
    request_id: str
    contract_version: str
    engine: str
    task: str
    positive_prompt: str
    negative_prompt: str
    base_video_ref: dict[str, str]
    reference_image_ref: dict[str, str]
    adapter_ref: dict[str, str]
    actor_profile_id: str
    training_run_id: str
    adapter_id: str
    width: int = 832
    height: int = 480
    fps: int = 16
    frames: int = 17
    steps: int = 30
    guidance_scale: float = 5.0
    seed: int = 99
    workflow_id: str = WORKFLOW_ID
    workflow_version: str = "1"
    graph_override: None = None
    metadata: dict[str, Any] = None

    @property
    def is_i2v(self) -> bool: return False
    @property
    def is_v2v(self) -> bool: return True


def parse_identity_ab_request(event: dict[str, Any]) -> IdentityAbRequest:
    payload = event.get("input", event)
    if not isinstance(payload, dict) or payload.get("contract_version") != CONTRACT_VERSION:
        raise ContractError("Contrato A/B neutro incompatível.")
    if payload.get("execution_mode") != "controlled_identity_neutral_ab":
        raise ContractError("Modo de execução A/B inválido.")
    request_id = _text(payload.get("request_id"))
    if not request_id:
        raise ContractError("request_id obrigatório para o teste A/B.")

    ids = [_text(payload.get(name)) for name in ("actor_profile_id", "training_run_id", "adapter_id")]
    if any(not value for value in ids):
        raise ContractError("Escopo do ator, run e adapter é obrigatório.")

    base = _private_ref(payload.get("base_video"))
    if base["bucket"] != NEUTRAL_BUCKET or base["key"] != NEUTRAL_KEY:
        raise ContractError("O A/B aceita somente o vídeo neutro homologado.")
    reference_image = _kyc_reference_ref(payload.get("reference_image"), ids[0])
    adapter = _private_ref(payload.get("adapter"))

    sampling = payload.get("sampling") or {}
    exact = {"seed": 99, "width": 832, "height": 480, "fps": 16, "frames": 17, "steps": 30, "denoise": 1.0, "lora_strength": 0.65}
    mismatched = [key for key, expected in exact.items() if sampling.get(key) != expected]
    if mismatched:
        raise ContractError("Parâmetros A/B divergentes do perfil homologado.", details={"fields": mismatched})

    smoke = payload.get("smoke") or {}
    if smoke.get("enabled") is not True or smoke.get("one_shot") is not True or int(smoke.get("max_jobs") or 0) != 1:
        raise ContractError("O teste A/B precisa ser one-shot.")
    expiry_raw = _text(smoke.get("expires_at"))
    try:
        expiry = datetime.fromisoformat(expiry_raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError("Janela do A/B inválida.") from exc
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    if expiry <= datetime.now(timezone.utc):
        raise ContractError("Janela do A/B expirada.")

    safety = payload.get("safety") or {}
    required = {
        "private_storage_only": True,
        "public_urls_forbidden": True,
        "automatic_retry_allowed": False,
        "one_shot_smoke": True,
        "kyc_reference_required": True,
        "kyc_reference_private_only": True,
        "kyc_reference_branch_b_only": True,
        "kyc_reference_persistence_forbidden": True,
        "product_release_allowed": False,
    }
    if any(safety.get(key) is not expected for key, expected in required.items()):
        raise ContractError("Contrato de segurança A/B incompleto.")

    prompt = payload.get("prompt") or {}
    positive = _text(prompt.get("positive"))
    if not positive:
        raise ContractError("Prompt neutro obrigatório.")

    return IdentityAbRequest(
        request_id=request_id,
        contract_version=CONTRACT_VERSION,
        engine="wan-2.1-v2v",
        task="identity.neutral_ab",
        positive_prompt=positive,
        negative_prompt=_text(prompt.get("negative")),
        base_video_ref=base,
        reference_image_ref=reference_image,
        adapter_ref=adapter,
        actor_profile_id=ids[0],
        training_run_id=ids[1],
        adapter_id=ids[2],
        metadata=payload.get("metadata") or {},
    )

def r2_client(settings: Settings):
    if not settings.r2_configured:
        raise DownloadError("O teste A/B exige R2 privado configurado.")
    return boto3.client("s3", endpoint_url=settings.r2_endpoint_url, aws_access_key_id=settings.r2_access_key_id, aws_secret_access_key=settings.r2_secret_access_key, region_name="auto")


def download_private_ref(client, ref: dict[str, str], destination: Path, max_mb: int) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try: client.download_file(ref["bucket"], ref["key"], str(destination))
    except Exception as exc: raise DownloadError("Falha ao baixar insumo privado do A/B.") from exc
    if destination.stat().st_size <= 0 or destination.stat().st_size > max_mb * 1024 * 1024:
        destination.unlink(missing_ok=True); raise DownloadError("Insumo privado do A/B vazio ou acima do limite.")
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    if digest != ref["sha256"]:
        destination.unlink(missing_ok=True); raise DownloadError("Checksum do insumo privado A/B divergente.")
    return destination


def materialize_lora(
    client, request: IdentityAbRequest, settings: Settings, work_dir: Path
) -> tuple[Path, str, dict[str, Any]]:
    staged = download_private_ref(
        client, request.adapter_ref, work_dir / "identity_adapter_original.safetensors", 4096
    )
    name = (
        f"privacy_identity_{request.adapter_ref['sha256'][:24]}_"
        f"{CONVERSION_VERSION}.safetensors"
    )
    destination = settings.model_root / "loras" / name
    attestation_path = destination.with_suffix(destination.suffix + ".attestation.json")
    model_path = settings.model_root / "diffusion_models" / settings.v2v_model_name
    if not model_path.is_file():
        raise LoraCompatibilityError(
            "Modelo Wan VACE não encontrado para o preflight LoRA.",
            details={"model_path": str(model_path)},
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or attestation_path.exists():
        if not destination.exists() or not attestation_path.exists():
            raise LoraCompatibilityError("Cache traduzido da LoRA está incompleto.")
        cached = read_conversion_attestation(attestation_path)
        if cached.get("source_sha256") != request.adapter_ref["sha256"]:
            raise LoraCompatibilityError("Cache traduzido pertence a outro adapter.")
        if cached.get("translated_sha256") != sha256_file(destination):
            raise LoraCompatibilityError("Checksum do cache traduzido é divergente.")
        if cached.get("model_layout_sha256") != model_layout_sha256(model_path):
            raise LoraCompatibilityError("Cache traduzido foi atestado contra outro layout Wan.")
        staged.unlink(missing_ok=True)
        return destination, name, cached

    converted_temp = work_dir / name
    attestation = convert_diffsynth_peft_lora(
        staged,
        model_path,
        converted_temp,
        source_sha256=request.adapter_ref["sha256"],
    )
    os.replace(converted_temp, destination)
    attestation = {**attestation, "translated_path": str(destination)}
    write_conversion_attestation(attestation_path, attestation)
    staged.unlink(missing_ok=True)
    return destination, name, attestation


def runtime_attestation_path(request: IdentityAbRequest, settings: Settings) -> Path:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in request.request_id)
    return settings.runtime_root / "identity-ab-attestations" / f"{safe[:160]}.json"


def read_runtime_lora_attestation(
    request: IdentityAbRequest,
    settings: Settings,
    *,
    expected_pair_count: int,
) -> dict[str, Any]:
    path = runtime_attestation_path(request, settings)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LoraNotAppliedError(
            "O ComfyUI terminou sem produzir atestação de patches LoRA.",
            details={"attestation_path": str(path)},
        ) from exc
    if payload.get("lora_applied") is not True:
        raise LoraNotAppliedError("A atestação do ComfyUI não confirma LoRA aplicada.", details=payload)
    if int(payload.get("patched_model_key_count") or 0) <= 0:
        raise LoraNotAppliedError("O ComfyUI registrou zero patches LoRA.", details=payload)
    if payload.get("all_expected_loaded") is not True or payload.get("all_expected_patched") is not True:
        raise LoraCompatibilityError("Nem todos os pares LoRA foram aplicados.", details=payload)
    expected = int(expected_pair_count)
    counts = {
        "expected_pair_count": int(payload.get("expected_pair_count") or 0),
        "loaded_patch_count": int(payload.get("loaded_patch_count") or 0),
        "patched_model_key_count": int(payload.get("patched_model_key_count") or 0),
    }
    if expected <= 0 or any(value != expected for value in counts.values()):
        raise LoraNotAppliedError(
            "A atestação runtime não comprova todos os pares LoRA esperados.",
            details={"conversion_pair_count": expected, **counts},
        )
    return payload


def _safe_lock_component(value: str) -> str:
    normalized = "".join(
        char if char.isalnum() or char in "._-" else "_"
        for char in str(value or "").strip()
    )
    if not normalized:
        raise ContractError("Escopo inválido para a reserva A/B.")
    return normalized[:160]


def one_shot_lock_path(request: IdentityAbRequest, settings: Settings) -> Path:
    root = settings.runtime_root / "identity-ab-locks"
    scope = (
        root
        / _safe_lock_component(request.actor_profile_id)
        / _safe_lock_component(request.training_run_id)
        / _safe_lock_component(request.adapter_id)
    )
    request_digest = hashlib.sha256(request.request_id.encode("utf-8")).hexdigest()
    return scope / f"{request_digest}.json"


def reserve_one_shot(request: IdentityAbRequest, settings: Settings) -> Path:
    path = one_shot_lock_path(request, settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "lock_version": 2,
        "request_id": request.request_id,
        "actor_profile_id": request.actor_profile_id,
        "training_run_id": request.training_run_id,
        "adapter_id": request.adapter_id,
        "contract_version": request.contract_version,
        "status": "reserved",
        "automatic_retry": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    except FileExistsError as exc:
        raise ContractError(
            "Este request_id A/B já foi reservado; repetição bloqueada."
        ) from exc
    return path


def update_lock(path: Path, status: str, **extra: Any) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**payload, "status": status, **extra}, indent=2), encoding="utf-8")
