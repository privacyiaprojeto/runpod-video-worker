from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import comfy.lora
import comfy.lora_convert
import comfy.utils
import folder_paths

NODE_VERSION = "privacy-attested-lora-loader-v1"


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._")
    if not normalized:
        raise RuntimeError("LORA_ATTESTATION_INVALID_NAME: attestation_name vazio")
    return normalized[:160]


def _expected_prefixes(lora: dict[str, Any]) -> set[str]:
    down = {
        key[: -len(".lora_down.weight")]
        for key in lora
        if key.endswith(".lora_down.weight")
    }
    up = {
        key[: -len(".lora_up.weight")]
        for key in lora
        if key.endswith(".lora_up.weight")
    }
    if not down or down != up:
        raise RuntimeError(
            f"LORA_KEY_FORMAT_MISMATCH: pares down/up inválidos; down={len(down)} up={len(up)}"
        )
    return down


def _key_map_with_compiled_aliases(model) -> dict[str, Any]:
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    for actual_key in model.model.state_dict().keys():
        if not actual_key.startswith("diffusion_model.") or not actual_key.endswith(".weight"):
            continue
        normalized_key = actual_key.replace("._orig_mod.", ".")
        if normalized_key == actual_key:
            continue
        normalized_prefix = normalized_key[:-len(".weight")]
        key_map.setdefault(normalized_prefix, actual_key)
    return key_map


class PrivacyAttestedLoraLoaderModelOnly:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
                "attestation_name": ("STRING", {"default": "identity-ab"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora_attested"
    CATEGORY = "Privacy IA/Identity QA"
    DESCRIPTION = "Loads a translated Wan LoRA and fails before sampling unless real model patches are attached."

    def load_lora_attested(self, model, lora_name: str, strength_model: float, attestation_name: str):
        if strength_model == 0:
            raise RuntimeError("LORA_NOT_APPLIED: strength_model não pode ser zero")
        lora_path = Path(folder_paths.get_full_path_or_raise("loras", lora_name))
        lora, metadata = comfy.utils.load_torch_file(
            str(lora_path), safe_load=True, return_metadata=True
        )
        prefixes = _expected_prefixes(lora)
        key_map = _key_map_with_compiled_aliases(model)
        missing_prefixes = sorted(prefix for prefix in prefixes if prefix not in key_map)
        if missing_prefixes:
            raise RuntimeError(
                "LORA_KEY_FORMAT_MISMATCH: prefixes sem destino no WAN21_Vace: "
                + ", ".join(missing_prefixes[:8])
            )

        converted = comfy.lora_convert.convert_lora(lora)
        loaded = comfy.lora.load_lora(converted, key_map, log_missing=False)
        expected_model_keys = {key_map[prefix] for prefix in prefixes}
        loaded_model_keys = set(loaded)
        missing_loaded = sorted(expected_model_keys - loaded_model_keys)
        if not loaded or missing_loaded:
            raise RuntimeError(
                f"LORA_KEY_FORMAT_MISMATCH: expected={len(expected_model_keys)} "
                f"loaded={len(loaded_model_keys)} missing={missing_loaded[:8]}"
            )

        patched_model = model.clone()
        patched = set(patched_model.add_patches(loaded, strength_model))
        missing_patched = sorted(expected_model_keys - patched)
        if not patched or missing_patched:
            raise RuntimeError(
                f"LORA_NOT_APPLIED: expected={len(expected_model_keys)} "
                f"patched={len(patched)} missing={missing_patched[:8]}"
            )

        runtime_root = Path(os.getenv("RUNTIME_ROOT", "/runpod-volume/privacy-wan-runtime"))
        output_path = runtime_root / "identity-ab-attestations" / f"{_safe_name(attestation_name)}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "privacy-comfyui-lora-patch-attestation-v1",
            "node_version": NODE_VERSION,
            "lora_name": lora_name,
            "strength_model": float(strength_model),
            "expected_pair_count": len(prefixes),
            "loaded_patch_count": len(loaded_model_keys),
            "patched_model_key_count": len(patched),
            "all_expected_loaded": not missing_loaded,
            "all_expected_patched": not missing_patched,
            "lora_applied": True,
            "patched_model_key_sample": sorted(patched)[:16],
            "metadata": metadata or {},
        }
        temp = output_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp.replace(output_path)
        return (patched_model,)


NODE_CLASS_MAPPINGS = {
    "PrivacyAttestedLoraLoaderModelOnly": PrivacyAttestedLoraLoaderModelOnly,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PrivacyAttestedLoraLoaderModelOnly": "Privacy IA — Attested Wan LoRA Loader",
}
