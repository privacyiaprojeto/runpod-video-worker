from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .config import Settings
from .errors import ContractError


_SAFE_UPDATE_FIELDS = {
    "workflow_id",
    "lora_conversion_version",
    "lora_pair_count",
    "control_representation",
    "derived_control_sha256",
    "asset_count",
    "patched_model_key_count",
    "ab_ssim_all",
    "error_code",
    "automatic_retry",
}
_VALID_STATUSES = {"reserved", "running", "completed", "failed"}


def _safe_component(value: str, *, name: str) -> str:
    normalized = "".join(
        char if char.isalnum() or char in "._-" else "_"
        for char in str(value or "").strip()
    )
    if not normalized or normalized in {".", ".."}:
        raise ContractError(f"Componente {name} inválido para o lock global.")
    return normalized[:160]


def _safe_prefix(prefix: str) -> str:
    prefix = str(prefix or "").strip().strip("/")
    if not prefix or "://" in prefix or "\\" in prefix:
        raise ContractError("IDENTITY_ONE_SHOT_LOCK_PREFIX inválido.")
    parts = prefix.split("/")
    if any(not part or part in {".", ".."} for part in parts):
        raise ContractError("IDENTITY_ONE_SHOT_LOCK_PREFIX inválido.")
    if any(not re.fullmatch(r"[A-Za-z0-9._-]+", part) for part in parts):
        raise ContractError("IDENTITY_ONE_SHOT_LOCK_PREFIX contém caracteres inseguros.")
    return "/".join(parts)


def r2_one_shot_lock_key(request: Any, settings: Settings) -> str:
    import hashlib

    request_digest = hashlib.sha256(request.request_id.encode("utf-8")).hexdigest()
    return "/".join(
        (
            _safe_prefix(settings.identity_one_shot_lock_prefix),
            "identity-one-shot",
            _safe_component(request.contract_version, name="contract_version"),
            _safe_component(request.actor_profile_id, name="actor_profile_id"),
            _safe_component(request.training_run_id, name="training_run_id"),
            _safe_component(request.adapter_id, name="adapter_id"),
            f"{request_digest}.json",
        )
    )


def _private_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _is_precondition_failed(error: Exception) -> bool:
    response = getattr(error, "response", None)
    if not isinstance(response, dict):
        return False
    error_data = response.get("Error") if isinstance(response.get("Error"), dict) else {}
    metadata = (
        response.get("ResponseMetadata")
        if isinstance(response.get("ResponseMetadata"), dict)
        else {}
    )
    return str(error_data.get("Code") or "") in {"PreconditionFailed", "412"} or str(
        metadata.get("HTTPStatusCode") or ""
    ) == "412"


@dataclass
class R2OneShotLock:
    client: Any
    bucket: str
    key: str
    payload: dict[str, Any]

    def update(self, status: str, **extra: Any) -> None:
        if status not in _VALID_STATUSES or status == "reserved":
            raise ContractError("Status inválido para atualização do lock global.")
        safe_extra = {key: value for key, value in extra.items() if key in _SAFE_UPDATE_FIELDS}
        updated = {
            **self.payload,
            "status": status,
            **safe_extra,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=self.key,
                Body=_private_json(updated),
                ContentType="application/json",
                CacheControl="private, no-store",
                Metadata={"private": "true", "one-shot-lock": "true"},
            )
        except Exception as exc:
            raise ContractError(
                "Falha ao atualizar o status do lock global; a reserva permanece ativa.",
                details={"lock_backend": "r2", "status": status},
            ) from exc
        self.payload = updated


def reserve_r2_one_shot(
    client: Any,
    request: Any,
    settings: Settings,
    payload: dict[str, Any],
) -> R2OneShotLock:
    key = r2_one_shot_lock_key(request, settings)
    try:
        client.put_object(
            Bucket=settings.r2_bucket_name,
            Key=key,
            Body=_private_json(payload),
            ContentType="application/json",
            CacheControl="private, no-store",
            Metadata={"private": "true", "one-shot-lock": "true"},
            IfNoneMatch="*",
        )
    except Exception as exc:
        if _is_precondition_failed(exc):
            raise ContractError(
                "Este request_id A/B já foi reservado; repetição bloqueada."
            ) from exc
        raise ContractError(
            "Falha ao reservar o lock global no R2; execução bloqueada.",
            details={"lock_backend": "r2"},
        ) from exc
    return R2OneShotLock(
        client=client,
        bucket=settings.r2_bucket_name,
        key=key,
        payload=dict(payload),
    )
