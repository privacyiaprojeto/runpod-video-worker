from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import ContractError, DownloadError
from .identity_ab import (
    download_private_ref,
    materialize_lora,
    r2_client,
    read_runtime_lora_attestation,
    reserve_one_shot,
    runtime_attestation_path,
    update_lock,
)
from .config import Settings

CONTRACT_VERSION = "privacy-identity-motion-abc-v1"
WORKFLOW_ID = "wan-2.1-v2v-identity-motion-abc-v1"
VALIDATION_PROFILE = "video_softedge_abc_v1"
CONTROL_REPRESENTATION = "softedge_ffmpeg_edgedetect_v1"
NEUTRAL_BUCKET = "privacy-media"
NEUTRAL_KEY = "qa-assets/neutral-motion-01.mp4"
WIDTH = 832
HEIGHT = 480
FPS = 16
FRAMES = 17
STEPS = 30
DENOISE = 0.85
LORA_STRENGTH = 0.65


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _sha(value: Any) -> str:
    value = _text(value).lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ContractError("Checksum SHA-256 inválido no contrato motion A/B/C.")
    return value


def _private_ref(value: Any) -> dict[str, str]:
    item = value if isinstance(value, dict) else {}
    bucket, key = _text(item.get("bucket")), _text(item.get("key"))
    if not bucket or not key or key.startswith("/") or "://" in bucket or "://" in key:
        raise ContractError("Referência privada inválida no contrato motion A/B/C.")
    return {"bucket": bucket, "key": key, "sha256": _sha(item.get("sha256"))}


def _kyc_reference_ref(value: Any, actor_profile_id: str) -> dict[str, str]:
    item = value if isinstance(value, dict) else {}
    ref = _private_ref(item)
    system_tag = _text(item.get("system_tag")).lower()
    if system_tag != "face_front":
        raise ContractError("A referência KYC dos ramos B/C precisa usar system_tag face_front.")
    if ref["bucket"] != NEUTRAL_BUCKET:
        raise ContractError("A referência KYC dos ramos B/C precisa permanecer no bucket privado aprovado.")
    normalized_key = ref["key"].lower()
    actor_scope = f"/actor-{actor_profile_id.lower()}/"
    if not normalized_key.startswith("vault/actor-mapping/") or actor_scope not in f"/{normalized_key}":
        raise ContractError("A referência KYC dos ramos B/C não pertence ao cofre privado deste ator.")
    if Path(ref["key"]).suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise ContractError("Formato da referência KYC frontal não suportado.")
    asset_id = _text(item.get("asset_id"))
    if not asset_id:
        raise ContractError("asset_id da referência KYC frontal é obrigatório.")
    return {**ref, "system_tag": "face_front", "asset_id": asset_id}


@dataclass(frozen=True)
class IdentityMotionAbcRequest:
    request_id: str
    contract_version: str
    engine: str
    task: str
    positive_prompt: str
    positive_prompt_b: str
    trigger_token: str
    negative_prompt: str
    base_video_ref: dict[str, str]
    reference_image_ref: dict[str, str]
    adapter_ref: dict[str, str]
    actor_profile_id: str
    training_run_id: str
    adapter_id: str
    width: int = WIDTH
    height: int = HEIGHT
    fps: int = FPS
    frames: int = FRAMES
    steps: int = STEPS
    guidance_scale: float = 5.0
    seed: int = 99
    branch_a_denoise: float = DENOISE
    branch_b_denoise: float = DENOISE
    branch_c_denoise: float = DENOISE
    workflow_id: str = WORKFLOW_ID
    workflow_version: str = "1"
    graph_override: None = None
    metadata: dict[str, Any] | None = None

    @property
    def is_i2v(self) -> bool:
        return False

    @property
    def is_v2v(self) -> bool:
        return True


def parse_identity_motion_abc_request(event: dict[str, Any]) -> IdentityMotionAbcRequest:
    payload = event.get("input", event)
    if not isinstance(payload, dict) or payload.get("contract_version") != CONTRACT_VERSION:
        raise ContractError("Contrato motion A/B/C incompatível.")
    if payload.get("execution_mode") != "controlled_identity_motion_abc":
        raise ContractError("Modo de execução motion A/B/C inválido.")

    request_id = _text(payload.get("request_id"))
    if not request_id:
        raise ContractError("request_id obrigatório para o teste motion A/B/C.")

    ids = [_text(payload.get(name)) for name in ("actor_profile_id", "training_run_id", "adapter_id")]
    if any(not value for value in ids):
        raise ContractError("Escopo do ator, run e adapter é obrigatório.")

    base = _private_ref(payload.get("base_video"))
    if base["bucket"] != NEUTRAL_BUCKET or base["key"] != NEUTRAL_KEY:
        raise ContractError("O motion A/B/C aceita somente o vídeo neutro homologado.")

    reference_image = _kyc_reference_ref(payload.get("reference_image"), ids[0])
    adapter = _private_ref(payload.get("adapter"))

    control = payload.get("control") or {}
    required_control = {
        "representation": CONTROL_REPRESENTATION,
        "derive_from_base_video": True,
        "raw_rgb_control_allowed": False,
        "same_control_all_branches": True,
    }
    if any(control.get(key) != expected for key, expected in required_control.items()):
        raise ContractError("Contrato de controle estrutural A/B/C incompleto ou inseguro.")

    sampling = payload.get("sampling") or {}
    exact = {
        "seed": 99,
        "width": WIDTH,
        "height": HEIGHT,
        "fps": FPS,
        "frames": FRAMES,
        "steps": STEPS,
        "denoise": DENOISE,
        "branch_b_denoise": DENOISE,
        "branch_c_denoise": DENOISE,
        "lora_strength": LORA_STRENGTH,
    }
    mismatched = [key for key, expected in exact.items() if sampling.get(key) != expected]
    if mismatched:
        raise ContractError(
            "Parâmetros motion A/B/C divergentes do perfil homologado.",
            details={"fields": mismatched},
        )

    smoke = payload.get("smoke") or {}
    if smoke.get("enabled") is not True or smoke.get("one_shot") is not True or int(smoke.get("max_jobs") or 0) != 1:
        raise ContractError("O teste motion A/B/C precisa ser one-shot.")
    expiry_raw = _text(smoke.get("expires_at"))
    try:
        expiry = datetime.fromisoformat(expiry_raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError("Janela do motion A/B/C inválida.") from exc
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    if expiry <= datetime.now(timezone.utc):
        raise ContractError("Janela do motion A/B/C expirada.")

    safety = payload.get("safety") or {}
    required_safety = {
        "private_storage_only": True,
        "public_urls_forbidden": True,
        "automatic_retry_allowed": False,
        "one_shot_smoke": True,
        "kyc_reference_required": True,
        "kyc_reference_private_only": True,
        "kyc_reference_baseline_forbidden": True,
        "kyc_reference_identity_branches_only": True,
        "kyc_reference_persistence_forbidden": True,
        "product_release_allowed": False,
    }
    if any(safety.get(key) is not expected for key, expected in required_safety.items()):
        raise ContractError("Contrato de segurança motion A/B/C incompleto.")

    identity = payload.get("identity") or {}
    trigger_token = _text(identity.get("trigger_token"))
    if not re.fullmatch(r"prv_actor_[a-z0-9_]+", trigger_token):
        raise ContractError("Trigger token da identidade ausente ou inválido.")
    if _text(identity.get("reference_asset_id")) != reference_image["asset_id"]:
        raise ContractError("A KYC explícita não corresponde ao asset_id identitário do contrato.")
    if _sha(identity.get("reference_sha256")) != reference_image["sha256"]:
        raise ContractError("A KYC explícita não corresponde ao checksum identitário do contrato.")

    prompt = payload.get("prompt") or {}
    positive = _text(prompt.get("positive"))
    positive_b = _text(prompt.get("positive_identity"))
    if not positive:
        raise ContractError("Prompt-base do ramo A obrigatório.")
    if not positive_b or not positive_b.startswith(trigger_token):
        raise ContractError("O prompt dos ramos B/C precisa iniciar com o trigger token exato da identidade.")
    if trigger_token in positive:
        raise ContractError("O trigger token não pode vazar para o ramo A.")

    return IdentityMotionAbcRequest(
        request_id=request_id,
        contract_version=CONTRACT_VERSION,
        engine="wan-2.1-v2v",
        task="identity.motion_abc",
        positive_prompt=positive,
        positive_prompt_b=positive_b,
        trigger_token=trigger_token,
        negative_prompt=_text(prompt.get("negative")),
        base_video_ref=base,
        reference_image_ref=reference_image,
        adapter_ref=adapter,
        actor_profile_id=ids[0],
        training_run_id=ids[1],
        adapter_id=ids[2],
        metadata=payload.get("metadata") or {},
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], *, error_message: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        stderr = _text(getattr(exc, "stderr", ""))[-500:]
        message = f"{error_message} ({command[0]})"
        if stderr:
            message += f": {stderr}"
        raise DownloadError(message) from exc


def assert_softedge_runtime_available() -> None:
    result = _run(
        ["ffmpeg", "-hide_banner", "-filters"],
        error_message="FFmpeg indisponível para derivar o controle soft-edge.",
    )
    if "edgedetect" not in result.stdout:
        raise DownloadError("A imagem do worker não possui o filtro FFmpeg edgedetect homologado.")


def derive_softedge_control(source: Path, destination: Path) -> dict[str, Any]:
    assert_softedge_runtime_available()
    destination.parent.mkdir(parents=True, exist_ok=True)
    vf = (
        f"fps={FPS},scale={WIDTH}:{HEIGHT}:flags=lanczos,"
        "format=gray,"
        "edgedetect=low=0.0784314:high=0.196078:mode=wires,"
        "format=yuv420p"
    )
    _run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source),
            "-vf", vf,
            "-frames:v", str(FRAMES),
            "-an",
            str(destination),
        ],
        error_message="Falha ao derivar o controle estrutural soft-edge.",
    )
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise DownloadError("O controle soft-edge derivado ficou vazio.")

    probe = _run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=width,height,nb_read_frames",
            "-of", "json",
            str(destination),
        ],
        error_message="Falha ao validar o controle soft-edge derivado.",
    )
    try:
        stream = (json.loads(probe.stdout).get("streams") or [])[0]
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        frames = int(stream.get("nb_read_frames") or 0)
    except (ValueError, TypeError, IndexError, json.JSONDecodeError) as exc:
        raise DownloadError("Metadados do controle soft-edge derivado são inválidos.") from exc
    if (width, height, frames) != (WIDTH, HEIGHT, FRAMES):
        raise DownloadError(
            f"Controle soft-edge derivado fora do contrato: {width}x{height}, frames={frames}."
        )
    return {
        "representation": CONTROL_REPRESENTATION,
        "source_sha256": sha256_file(source),
        "derived_sha256": sha256_file(destination),
        "width": width,
        "height": height,
        "frames": frames,
        "fps": FPS,
        "raw_rgb_control_used": False,
        "appearance_reduced_structural_control_used": True,
        "ffmpeg_filter": "format=gray,edgedetect(mode=wires),format=yuv420p",
    }


__all__ = [
    "CONTRACT_VERSION",
    "WORKFLOW_ID",
    "VALIDATION_PROFILE",
    "CONTROL_REPRESENTATION",
    "IdentityMotionAbcRequest",
    "parse_identity_motion_abc_request",
    "derive_softedge_control",
    "download_private_ref",
    "materialize_lora",
    "r2_client",
    "read_runtime_lora_attestation",
    "reserve_one_shot",
    "runtime_attestation_path",
    "update_lock",
]
