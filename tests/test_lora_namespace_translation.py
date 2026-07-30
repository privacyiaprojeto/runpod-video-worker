from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from privacy_worker.errors import LoraCompatibilityError
from privacy_worker.lora_namespace import convert_diffsynth_peft_lora, inspect_diffsynth_peft_lora


def _write_model(path: Path):
    save_file(
        {
            "blocks.0.cross_attn.q.weight": torch.zeros((8, 6)),
            "blocks.0.ffn.0.weight": torch.zeros((10, 6)),
        },
        str(path),
    )


def _write_adapter(path: Path, *, bad_shape=False):
    save_file(
        {
            "blocks.0.cross_attn.q.lora_A.default.weight": torch.ones((2, 6)),
            "blocks.0.cross_attn.q.lora_B.default.weight": torch.ones((7 if bad_shape else 8, 2)),
            "blocks.0.ffn.0.lora_A.default.weight": torch.ones((2, 6)),
            "blocks.0.ffn.0.lora_B.default.weight": torch.ones((10, 2)),
        },
        str(path),
    )


def test_diffsynth_peft_namespace_is_translated_and_shape_attested(tmp_path):
    model = tmp_path / "wan.safetensors"
    adapter = tmp_path / "adapter.safetensors"
    output = tmp_path / "translated.safetensors"
    _write_model(model)
    _write_adapter(adapter)

    preflight = inspect_diffsynth_peft_lora(adapter, model)
    assert preflight["pair_count"] == 2
    assert preflight["all_shapes_compatible"] is True
    assert preflight["target_modules"] == ["cross_attn.q", "ffn.0"]

    report = convert_diffsynth_peft_lora(adapter, model, output, source_sha256="a" * 64)
    assert report["ready_for_comfyui_attested_loader"] is True
    assert report["translated_tensor_count"] == 6
    with safe_open(output, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        assert "diffusion_model.blocks.0.cross_attn.q.lora_down.weight" in keys
        assert "diffusion_model.blocks.0.cross_attn.q.lora_up.weight" in keys
        assert "diffusion_model.blocks.0.cross_attn.q.alpha" in keys
        assert not any(".default.weight" in key for key in keys)
        assert not any(".lora_A.weight" in key or ".lora_B.weight" in key for key in keys)
        assert handle.get_tensor("diffusion_model.blocks.0.cross_attn.q.alpha").item() == 2.0


def test_shape_mismatch_fails_closed(tmp_path):
    model = tmp_path / "wan.safetensors"
    adapter = tmp_path / "adapter.safetensors"
    _write_model(model)
    _write_adapter(adapter, bad_shape=True)
    with pytest.raises(LoraCompatibilityError, match="não corresponde"):
        inspect_diffsynth_peft_lora(adapter, model)


def test_missing_target_pair_fails_full_coverage(tmp_path):
    model = tmp_path / "wan.safetensors"
    adapter = tmp_path / "adapter.safetensors"
    save_file({
        "blocks.0.cross_attn.q.weight": torch.zeros((8, 6)),
        "blocks.0.ffn.0.weight": torch.zeros((10, 6)),
    }, str(model))
    save_file({
        "blocks.0.cross_attn.q.lora_A.default.weight": torch.ones((2, 6)),
        "blocks.0.cross_attn.q.lora_B.default.weight": torch.ones((8, 2)),
    }, str(adapter))
    with pytest.raises(LoraCompatibilityError, match="cobertura"):
        inspect_diffsynth_peft_lora(adapter, model)


def test_compiled_orig_mod_model_header_is_supported(tmp_path):
    model = tmp_path / "wan.safetensors"
    adapter = tmp_path / "adapter.safetensors"
    output = tmp_path / "translated.safetensors"
    save_file({
        "diffusion_model.blocks.0._orig_mod.cross_attn.q.weight": torch.zeros((8, 6)),
    }, str(model))
    save_file({
        "blocks.0.cross_attn.q.lora_A.default.weight": torch.ones((2, 6)),
        "blocks.0.cross_attn.q.lora_B.default.weight": torch.ones((8, 2)),
    }, str(adapter))
    report = convert_diffsynth_peft_lora(adapter, model, output)
    assert report["pair_count"] == 1
    assert report["full_target_coverage"] is True
