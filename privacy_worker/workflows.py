from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Settings
from .contracts import ProductionRequest
from .errors import WorkflowError

WORKFLOW_SCHEMA_VERSION = "privacy-comfyui-workflow-v1"


@dataclass(frozen=True)
class PreparedWorkflow:
    workflow_id: str
    workflow_version: str
    prompt: dict[str, Any]
    output_nodes: tuple[str, ...]


def _set_single_node_input(
    prompt: dict[str, Any], binding: dict[str, Any], value: Any, logical_name: str
) -> None:
    node_id = str(binding.get("node_id") or binding.get("node") or "")
    input_name = str(binding.get("input") or "")
    required = binding.get("required", True)
    if not node_id or not input_name:
        if required:
            raise WorkflowError(f"Binding inválido para {logical_name}.")
        return
    node = prompt.get(node_id)
    if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict):
        if required:
            raise WorkflowError(f"Nó {node_id} do binding {logical_name} não existe no workflow.")
        return
    node["inputs"][input_name] = value


def _set_node_input(
    prompt: dict[str, Any], binding: dict[str, Any] | list[dict[str, Any]], value: Any, logical_name: str
) -> None:
    targets = binding if isinstance(binding, list) else [binding]
    if not targets:
        raise WorkflowError(f"Binding vazio para {logical_name}.")
    for target in targets:
        if not isinstance(target, dict):
            raise WorkflowError(f"Binding inválido para {logical_name}.")
        _set_single_node_input(prompt, target, value, logical_name)


def _load_envelope(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkflowError(f"Não foi possível carregar o workflow {path.name}.") from error
    if payload.get("schema_version") != WORKFLOW_SCHEMA_VERSION:
        raise WorkflowError(f"Workflow {path.name} usa schema_version incompatível.")
    if not isinstance(payload.get("prompt"), dict) or not payload["prompt"]:
        raise WorkflowError(f"Workflow {path.name} não contém um prompt API do ComfyUI.")
    if not isinstance(payload.get("bindings"), dict):
        raise WorkflowError(f"Workflow {path.name} não contém bindings.")
    return payload


def _override_envelope(local: dict[str, Any], graph_override: dict[str, Any] | None) -> dict[str, Any]:
    if not graph_override:
        return local
    override = copy.deepcopy(graph_override)
    if "prompt" in override:
        prompt = override.get("prompt")
        if not isinstance(prompt, dict) or not prompt:
            raise WorkflowError("comfyui.graph.prompt precisa ser um workflow em API format.")
        return {
            **local,
            **{key: value for key, value in override.items() if key != "prompt"},
            "prompt": prompt,
            "bindings": override.get("bindings") or local["bindings"],
            "output_nodes": override.get("output_nodes") or local.get("output_nodes", []),
        }

    # A pure ComfyUI API prompt can be supplied directly; packaged bindings remain authoritative.
    if all(isinstance(value, dict) and "class_type" in value for value in override.values()):
        return {**local, "prompt": override}
    raise WorkflowError("comfyui.graph precisa ser um prompt API puro ou um envelope com prompt/bindings.")


def prepare_workflow(
    *,
    request: ProductionRequest,
    source_image_filename: str,
    base_video_filename: str | None,
    output_prefix: str,
    settings: Settings,
) -> PreparedWorkflow:
    path = settings.workflow_root / f"{request.workflow_id}.json"
    if not path.exists():
        raise WorkflowError(f"Workflow versionado não encontrado: {request.workflow_id}.")
    envelope = _override_envelope(_load_envelope(path), request.graph_override)

    engine = str(envelope.get("engine") or "")
    if engine and engine != request.engine:
        raise WorkflowError(
            f"Workflow {request.workflow_id} pertence ao engine {engine}, não {request.engine}."
        )
    declared_version = str(envelope.get("workflow_version") or "")
    if declared_version and declared_version != request.workflow_version:
        raise WorkflowError(
            f"Versão do workflow incompatível: solicitado {request.workflow_version}, disponível {declared_version}."
        )

    prompt = copy.deepcopy(envelope["prompt"])
    values = {
        "positive_prompt": request.positive_prompt,
        "negative_prompt": request.negative_prompt,
        "source_image": source_image_filename,
        "base_video": base_video_filename,
        "width": request.width,
        "height": request.height,
        "frames": request.frames,
        "fps": request.fps,
        "steps": request.steps,
        "cfg": request.guidance_scale,
        "seed": request.seed,
        "filename_prefix": output_prefix,
        "i2v_model_name": settings.i2v_model_name,
        "v2v_model_name": settings.v2v_model_name,
        "text_encoder_name": settings.text_encoder_name,
        "vae_name": settings.vae_name,
        "clip_vision_name": settings.clip_vision_name,
    }

    bindings = envelope["bindings"]
    for logical_name, binding in bindings.items():
        if logical_name not in values:
            continue
        value = values[logical_name]
        if value is None and binding.get("required", True):
            raise WorkflowError(f"Valor obrigatório ausente para binding {logical_name}.")
        if value is None:
            continue
        _set_node_input(prompt, binding, value, logical_name)

    output_nodes = tuple(str(item) for item in envelope.get("output_nodes") or ())
    if not output_nodes:
        raise WorkflowError("O workflow não declara output_nodes.")

    return PreparedWorkflow(
        workflow_id=request.workflow_id,
        workflow_version=request.workflow_version,
        prompt=prompt,
        output_nodes=output_nodes,
    )
