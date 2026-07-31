from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


from .errors import LoraCompatibilityError

CONVERSION_VERSION = "privacy-diffsynth-peft-to-comfy-wan-v1"
SOURCE_FORMAT = "diffsynth_peft_wan_dit"
TARGET_FORMAT = "comfyui_generic_wan_lora_down_up"

# This is deliberately narrow. Unknown namespaces are rejected instead of guessed.
_SOURCE_KEY = re.compile(
    r"^(?:(?:base_model\.model|pipe\.dit|dit)\.)?"
    r"(?P<module>blocks\.(?P<block>\d+)\."
    r"(?P<target>cross_attn\.(?:q|k|v|o)|ffn\.(?:0|2)))\."
    r"lora_(?P<side>[AB])(?:\.default)?\.weight$"
)
_ALLOWED_TARGETS = {
    "cross_attn.q",
    "cross_attn.k",
    "cross_attn.v",
    "cross_attn.o",
    "ffn.0",
    "ffn.2",
}


@dataclass(frozen=True)
class TensorHeader:
    shape: tuple[int, ...]
    dtype: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _header(path: Path) -> tuple[dict[str, TensorHeader], dict[str, str]]:
    # D3.6H10-HF2 — lazy safetensors/PyTorch runtime imports
    # Merely importing the contract must not require GPU/runtime packages.
    # safetensors is loaded only when an adapter/model header is actually read.
    try:
        from safetensors import safe_open
    except ModuleNotFoundError as exc:
        raise LoraCompatibilityError(
            "Runtime safetensors indisponível para inspecionar a LoRA Wan.",
            details={"missing_dependency": getattr(exc, "name", None) or "safetensors"},
        ) from exc

    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            headers = {
                key: TensorHeader(
                    shape=tuple(int(item) for item in handle.get_slice(key).get_shape()),
                    dtype=str(handle.get_slice(key).get_dtype()),
                )
                for key in handle.keys()
            }
            metadata = dict(handle.metadata() or {})
    except Exception as exc:  # safetensors raises several concrete error types
        raise LoraCompatibilityError(
            "Não foi possível ler o header safetensors.",
            details={"path": str(path), "reason": str(exc)},
        ) from exc
    if not headers:
        raise LoraCompatibilityError("O safetensors não contém tensores.")
    return headers, metadata


def model_layout_sha256(path: Path) -> str:
    headers, _ = _header(path)
    canonical = [
        {"key": key, "shape": list(header.shape), "dtype": header.dtype}
        for key, header in sorted(headers.items())
    ]
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _base_weight_candidates(module: str) -> tuple[str, ...]:
    block, remainder = module.split(".", 2)[1:]
    compiled_module = f"blocks.{block}._orig_mod.{remainder}"
    return (
        f"{module}.weight",
        f"{compiled_module}.weight",
        f"diffusion_model.{module}.weight",
        f"diffusion_model.{compiled_module}.weight",
        f"model.diffusion_model.{module}.weight",
        f"model.diffusion_model.{compiled_module}.weight",
        f"model.{module}.weight",
        f"model.{compiled_module}.weight",
    )


def _normalize_model_target(key: str) -> str | None:
    normalized = key
    for prefix in ("model.diffusion_model.", "diffusion_model.", "model."):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    normalized = normalized.replace("._orig_mod.", ".")
    if not normalized.endswith(".weight"):
        return None
    module = normalized[:-len(".weight")]
    match = re.fullmatch(
        r"blocks\.(?P<block>\d+)\.(?P<target>cross_attn\.(?:q|k|v|o)|ffn\.(?:0|2))",
        module,
    )
    if not match:
        return None
    return module


def _resolve_base_weight(model_headers: dict[str, TensorHeader], module: str) -> tuple[str, TensorHeader]:
    for key in _base_weight_candidates(module):
        header = model_headers.get(key)
        if header is not None:
            return key, header
    raise LoraCompatibilityError(
        "O alvo LoRA não existe no modelo Wan implantado.",
        details={"module": module, "candidates": list(_base_weight_candidates(module))},
    )


def inspect_diffsynth_peft_lora(adapter_path: Path, model_path: Path) -> dict[str, Any]:
    adapter_headers, source_metadata = _header(adapter_path)
    model_headers, _ = _header(model_path)

    pairs: dict[str, dict[str, tuple[str, TensorHeader]]] = {}
    rejected: list[str] = []
    source_styles: set[str] = set()

    for key, header in adapter_headers.items():
        match = _SOURCE_KEY.fullmatch(key)
        if not match:
            rejected.append(key)
            continue
        module = match.group("module")
        side = match.group("side")
        target = match.group("target")
        if target not in _ALLOWED_TARGETS:
            rejected.append(key)
            continue
        source_styles.add("peft_default" if ".default.weight" in key else "peft_plain")
        pair = pairs.setdefault(module, {})
        if side in pair:
            raise LoraCompatibilityError(
                "O adapter contém lado LoRA duplicado.",
                details={"module": module, "side": side},
            )
        pair[side] = (key, header)

    if rejected:
        raise LoraCompatibilityError(
            "O adapter contém chaves fora do contrato DiffSynth/PEFT aprovado.",
            details={"rejected_count": len(rejected), "rejected_sample": rejected[:12]},
        )
    if not pairs:
        raise LoraCompatibilityError("Nenhum par LoRA DiffSynth/PEFT foi encontrado.")

    expected_model_modules = {
        module
        for key in model_headers
        if (module := _normalize_model_target(key)) is not None
    }
    if not expected_model_modules:
        raise LoraCompatibilityError(
            "O modelo Wan não expõe os alvos DiT esperados para a LoRA.",
            details={"allowed_targets": sorted(_ALLOWED_TARGETS)},
        )
    adapter_modules = set(pairs)
    missing_modules = sorted(expected_model_modules - adapter_modules)
    unexpected_modules = sorted(adapter_modules - expected_model_modules)
    if missing_modules or unexpected_modules:
        raise LoraCompatibilityError(
            "A cobertura do adapter não corresponde aos alvos DiT do modelo Wan.",
            details={
                "expected_pair_count": len(expected_model_modules),
                "adapter_pair_count": len(adapter_modules),
                "missing_sample": missing_modules[:12],
                "unexpected_sample": unexpected_modules[:12],
            },
        )

    pair_reports: list[dict[str, Any]] = []
    ranks: set[int] = set()
    blocks: set[int] = set()
    targets: set[str] = set()

    for module in sorted(pairs):
        pair = pairs[module]
        if set(pair) != {"A", "B"}:
            raise LoraCompatibilityError(
                "O adapter contém par LoRA incompleto.",
                details={"module": module, "sides": sorted(pair)},
            )
        a_key, a_header = pair["A"]
        b_key, b_header = pair["B"]
        if len(a_header.shape) != 2 or len(b_header.shape) != 2:
            raise LoraCompatibilityError(
                "Somente matrizes LoRA lineares 2D são aceitas.",
                details={"module": module, "a_shape": a_header.shape, "b_shape": b_header.shape},
            )
        rank = a_header.shape[0]
        if rank <= 0 or b_header.shape[1] != rank:
            raise LoraCompatibilityError(
                "As matrizes A/B possuem ranks incompatíveis.",
                details={"module": module, "a_shape": a_header.shape, "b_shape": b_header.shape},
            )
        model_key, model_header = _resolve_base_weight(model_headers, module)
        expected = (b_header.shape[0], a_header.shape[1])
        if tuple(model_header.shape) != expected:
            raise LoraCompatibilityError(
                "A forma do par LoRA não corresponde ao peso Wan alvo.",
                details={
                    "module": module,
                    "model_key": model_key,
                    "model_shape": model_header.shape,
                    "a_shape": a_header.shape,
                    "b_shape": b_header.shape,
                    "expected_model_shape": expected,
                },
            )
        block = int(module.split(".")[1])
        target = ".".join(module.split(".")[2:])
        blocks.add(block)
        targets.add(target)
        ranks.add(rank)
        pair_reports.append(
            {
                "module": module,
                "source_a": a_key,
                "source_b": b_key,
                "target_prefix": f"diffusion_model.{module}",
                "model_key": model_key,
                "rank": rank,
                "a_shape": list(a_header.shape),
                "b_shape": list(b_header.shape),
                "model_shape": list(model_header.shape),
            }
        )

    return {
        "schema_version": "privacy-lora-namespace-attestation-v1",
        "conversion_version": CONVERSION_VERSION,
        "source_format": SOURCE_FORMAT,
        "target_format": TARGET_FORMAT,
        "source_style": sorted(source_styles),
        "source_tensor_count": len(adapter_headers),
        "pair_count": len(pair_reports),
        "model_keys_matched": len(pair_reports),
        "expected_model_pair_count": len(expected_model_modules),
        "full_target_coverage": True,
        "all_pairs_complete": True,
        "all_shapes_compatible": True,
        "rank_values": sorted(ranks),
        "blocks": sorted(blocks),
        "target_modules": sorted(targets),
        "pair_sample": pair_reports[:12],
        "source_metadata": source_metadata,
        "model_layout_sha256": hashlib.sha256(
            json.dumps(
                [
                    {"key": key, "shape": list(header.shape), "dtype": header.dtype}
                    for key, header in sorted(model_headers.items())
                ],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "_pairs": pairs,
    }


def convert_diffsynth_peft_lora(
    adapter_path: Path,
    model_path: Path,
    destination: Path,
    *,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    # D3.6H10-HF2 — lazy safetensors/PyTorch runtime imports
    # Tensor conversion is the only place that requires PyTorch and
    # safetensors.torch. Contract parsing and lock tests stay dependency-light.
    # D3.6H10-HF3 — bind safe_open inside conversion runtime
    # convert_diffsynth_peft_lora uses safe_open directly after inspection.
    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ModuleNotFoundError as exc:
        raise LoraCompatibilityError(
            "Runtime PyTorch/safetensors.torch indisponível para materializar a LoRA Wan.",
            details={"missing_dependency": getattr(exc, "name", None) or "torch"},
        ) from exc

    inspection = inspect_diffsynth_peft_lora(adapter_path, model_path)
    pairs = inspection.pop("_pairs")
    tensors: dict[str, Any] = {}

    try:
        with safe_open(adapter_path, framework="pt", device="cpu") as handle:
            for module in sorted(pairs):
                a_source = pairs[module]["A"][0]
                b_source = pairs[module]["B"][0]
                prefix = f"diffusion_model.{module}"
                a_tensor = handle.get_tensor(a_source).contiguous()
                b_tensor = handle.get_tensor(b_source).contiguous()
                rank = int(a_tensor.shape[0])
                # PEFT A is the down projection; PEFT B is the up projection.
                # ComfyUI v0.27.0 reliably recognizes the generic down/up suffixes.
                tensors[f"{prefix}.lora_down.weight"] = a_tensor
                tensors[f"{prefix}.lora_up.weight"] = b_tensor
                # The training contract fixes alpha == rank, making the intrinsic scale 1.0.
                tensors[f"{prefix}.alpha"] = torch.tensor(float(rank), dtype=torch.float32)
    except LoraCompatibilityError:
        raise
    except Exception as exc:
        raise LoraCompatibilityError(
            "Falha ao materializar os tensores LoRA traduzidos.",
            details={"reason": str(exc)},
        ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    temp.unlink(missing_ok=True)
    metadata = {
        "privacy_conversion_version": CONVERSION_VERSION,
        "privacy_source_format": SOURCE_FORMAT,
        "privacy_target_format": TARGET_FORMAT,
        "privacy_source_sha256": source_sha256 or sha256_file(adapter_path),
        "privacy_pair_count": str(inspection["pair_count"]),
        "privacy_alpha_policy": "rank_equals_alpha",
    }
    try:
        save_file(tensors, str(temp), metadata=metadata)
        temp.replace(destination)
    finally:
        temp.unlink(missing_ok=True)

    translated_sha256 = sha256_file(destination)
    result = {
        **inspection,
        "source_sha256": source_sha256 or sha256_file(adapter_path),
        "translated_sha256": translated_sha256,
        "translated_tensor_count": len(tensors),
        "translated_path": str(destination),
        "translated_name": destination.name,
        "alpha_policy": "rank_equals_alpha",
        "ready_for_comfyui_attested_loader": True,
    }
    return result


def write_conversion_attestation(path: Path, payload: dict[str, Any]) -> None:
    serializable = {key: value for key, value in payload.items() if not key.startswith("_")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8")


def read_conversion_attestation(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LoraCompatibilityError(
            "Atestação de tradução da LoRA ausente ou inválida.",
            details={"path": str(path)},
        ) from exc
    if payload.get("conversion_version") != CONVERSION_VERSION:
        raise LoraCompatibilityError("Versão da atestação de tradução incompatível.")
    if payload.get("pair_count", 0) <= 0 or payload.get("all_shapes_compatible") is not True:
        raise LoraCompatibilityError("Atestação de tradução não comprova pares LoRA válidos.")
    return payload
