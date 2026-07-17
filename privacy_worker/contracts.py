from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import ContractError

CONTRACT_VERSION = "privacy-production-spec-v1"
SUPPORTED_ENGINES = {"wan-2.1-i2v", "wan-2.1-v2v"}
EXPECTED_TASK_BY_ENGINE = {"wan-2.1-i2v": "video.i2v", "wan-2.1-v2v": "video.v2v"}
COMFYUI_ADAPTER_VERSION = "comfyui-graph-contract-v1"
IMAGE_TAG_PRIORITY = (
    "face_front",
    "face_profile",
    "nsfw_closeup_front",
    "nsfw_closeup_back",
    "body_front",
    "body_back",
    "video_expression",
    "video_walk",
)


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _number(value: Any, default: int | float, *, minimum: float, maximum: float):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    parsed = min(max(parsed, minimum), maximum)
    if isinstance(default, int):
        return int(parsed)
    return parsed


def _multiple_of_16(value: Any, default: int) -> int:
    parsed = _number(value, default, minimum=256, maximum=2048)
    return max(256, (int(parsed) // 16) * 16)


def _wan_frame_count(value: Any, default: int) -> int:
    parsed = _number(value, default, minimum=9, maximum=1201)
    # Wan latents operate on 4n+1 frame counts.
    return max(9, ((int(parsed) - 1) // 4) * 4 + 1)


def _unwrap_payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("input", event)
    if not isinstance(payload, dict):
        raise ContractError("O input do job precisa ser um objeto JSON.")
    nested = payload.get("production_spec") or payload.get("productionSpec")
    if isinstance(nested, dict):
        return nested
    return payload


def _approved_image_candidates(spec: dict[str, Any]) -> list[dict[str, Any]]:
    identity = spec.get("identity") or {}
    actors = identity.get("actors") or []
    candidates: list[dict[str, Any]] = []
    for actor in actors if isinstance(actors, list) else []:
        references = actor.get("references") or []
        primary = _text(actor.get("primary_reference_url"))
        for item in references if isinstance(references, list) else []:
            media_type = _text(item.get("media_type")).lower()
            content_type = _text(item.get("content_type")).lower()
            url = _text(item.get("signed_url"))
            if not url or (media_type and media_type != "image"):
                continue
            if content_type and not content_type.startswith("image/"):
                continue
            candidates.append(
                {
                    "url": url,
                    "system_tag": _text(item.get("system_tag")),
                    "slot_index": actor.get("slot_index", 1),
                    "actor_profile_id": actor.get("actor_profile_id"),
                    "is_primary": bool(primary and url == primary),
                }
            )

    priority = {tag: index for index, tag in enumerate(IMAGE_TAG_PRIORITY)}
    candidates.sort(
        key=lambda item: (
            int(item.get("slot_index") or 1),
            0 if item.get("is_primary") else 1,
            priority.get(item.get("system_tag"), 999),
        )
    )
    seen: set[str] = set()
    unique = []
    for candidate in candidates:
        if candidate["url"] in seen:
            continue
        seen.add(candidate["url"])
        unique.append(candidate)
    return unique


@dataclass(frozen=True)
class ProductionRequest:
    request_id: str
    contract_version: str
    engine: str
    task: str
    positive_prompt: str
    negative_prompt: str
    source_image_url: str
    base_video_url: str | None
    width: int
    height: int
    fps: int
    frames: int
    steps: int
    guidance_scale: float
    seed: int
    workflow_id: str
    workflow_version: str
    graph_override: dict[str, Any] | None
    metadata: dict[str, Any]

    @property
    def is_i2v(self) -> bool:
        return self.engine == "wan-2.1-i2v"

    @property
    def is_v2v(self) -> bool:
        return self.engine == "wan-2.1-v2v"


def parse_production_request(event: dict[str, Any]) -> ProductionRequest:
    spec = _unwrap_payload(event)
    contract_version = _text(spec.get("contract_version"))
    if contract_version != CONTRACT_VERSION:
        raise ContractError(
            f"contract_version inválido: esperado {CONTRACT_VERSION}.",
            details={"received": contract_version or None},
        )

    engine = _text(spec.get("engine")).lower()
    if engine not in SUPPORTED_ENGINES:
        raise ContractError(
            "Engine de vídeo não suportado.",
            details={"engine": engine, "supported": sorted(SUPPORTED_ENGINES)},
        )

    safety = spec.get("safety") or {}
    required_flags = (
        "licensed_or_consented_assets_only",
        "require_approved_identity_references",
        "private_output_only",
        "public_url_forbidden",
        "qa_required",
    )
    missing_flags = [name for name in required_flags if safety.get(name) is not True]
    if missing_flags:
        raise ContractError(
            "Contrato recusado por política de segurança.",
            details={"required_true_flags": missing_flags},
        )

    prompt = spec.get("prompt") or {}
    positive_prompt = _text(prompt.get("positive"))
    if not positive_prompt:
        raise ContractError("Prompt positivo obrigatório.")
    negative_prompt = _text(prompt.get("negative"))

    task = _text(spec.get("task")).lower()
    expected_task = EXPECTED_TASK_BY_ENGINE[engine]
    if task != expected_task:
        raise ContractError(
            "Task incompatível com o engine selecionado.",
            details={"engine": engine, "received": task or None, "expected": expected_task},
        )

    candidates = _approved_image_candidates(spec)
    if not candidates:
        raise ContractError("Nenhuma imagem biométrica aprovada foi fornecida ao worker.")

    # Identity must always originate from the approved KYC reference_media list.
    conditioning = spec.get("conditioning") or {}
    requested_source = _text(conditioning.get("source_image_url"))
    approved_urls = {item["url"] for item in candidates}
    source_image_url = requested_source if requested_source in approved_urls else candidates[0]["url"]

    base_video_url = _text(conditioning.get("base_video_url")) or None
    if engine == "wan-2.1-v2v" and not base_video_url:
        raise ContractError("base_video_url é obrigatório para wan-2.1-v2v.")

    output = spec.get("output") or {}
    sampling = spec.get("sampling") or {}
    comfyui = spec.get("comfyui") or {}
    adapter = _text(comfyui.get("adapter"))
    if adapter and adapter != COMFYUI_ADAPTER_VERSION:
        raise ContractError(
            "Adapter ComfyUI incompatível.",
            details={"received": adapter, "expected": COMFYUI_ADAPTER_VERSION},
        )
    workflow_id = _text(comfyui.get("workflow_id")) or (
        "wan-2.1-i2v-v1" if engine == "wan-2.1-i2v" else "wan-2.1-v2v-v1"
    )
    workflow_version = _text(comfyui.get("workflow_version")) or "1"
    graph_override = comfyui.get("graph") if isinstance(comfyui.get("graph"), dict) else None

    request_id = _text(spec.get("request_id") or event.get("id"))
    if not request_id:
        raise ContractError("request_id obrigatório para rastreabilidade.")

    return ProductionRequest(
        request_id=request_id,
        contract_version=contract_version,
        engine=engine,
        task=task,
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        source_image_url=source_image_url,
        base_video_url=base_video_url,
        width=_multiple_of_16(output.get("width"), 832),
        height=_multiple_of_16(output.get("height"), 480),
        fps=_number(output.get("fps"), 16, minimum=1, maximum=60),
        frames=_wan_frame_count(output.get("frames"), 49 if engine == "wan-2.1-i2v" else 81),
        steps=_number(sampling.get("steps"), 30, minimum=1, maximum=150),
        guidance_scale=_number(sampling.get("guidance_scale"), 5.0, minimum=0, maximum=30),
        seed=_number(sampling.get("seed"), 1, minimum=0, maximum=2**63 - 1),
        workflow_id=workflow_id,
        workflow_version=workflow_version,
        graph_override=graph_override,
        metadata=spec.get("metadata") if isinstance(spec.get("metadata"), dict) else {},
    )
