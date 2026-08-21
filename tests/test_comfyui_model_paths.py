import json
from dataclasses import replace
from pathlib import Path

import pytest

from privacy_worker.comfyui import ComfyUIClient, ComfyUIProcessManager
from privacy_worker.config import Settings
from privacy_worker.errors import OutputError


class FakeProcess:
    stdout = []
    returncode = None

    def poll(self):
        return None


def test_comfyui_receives_dynamic_extra_model_paths(monkeypatch, tmp_path):
    comfyui_root = tmp_path / "ComfyUI"
    comfyui_root.mkdir()
    (comfyui_root / "main.py").write_text("# test\n", encoding="utf-8")
    settings = replace(
        Settings(),
        comfyui_root=comfyui_root,
        runtime_root=tmp_path / "runtime",
        model_root=tmp_path / "models-overlay",
        comfyui_start_timeout_seconds=1,
    )
    settings.ensure_runtime_dirs()
    manager = ComfyUIProcessManager(settings)
    readiness = iter((False, False, True))
    monkeypatch.setattr(manager, "_is_ready", lambda: next(readiness))
    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        return FakeProcess()

    monkeypatch.setattr("privacy_worker.comfyui.subprocess.Popen", fake_popen)
    manager.ensure_started("request-test")

    command = captured["command"]
    option = command.index("--extra-model-paths-config")
    config_path = Path(command[option + 1])
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["privacy_wan_models"]["base_path"] == str(settings.model_root)


def test_cached_mode_removes_only_controlled_local_comfy_output(tmp_path):
    settings = replace(
        Settings(),
        model_source_mode="cached_model",
        runtime_root=tmp_path / "runtime",
        model_root=tmp_path / "models-overlay",
    )
    settings.ensure_runtime_dirs()
    client = ComfyUIClient(settings)
    local_output = settings.output_dir / "private-result.mp4"
    local_output.write_bytes(b"video")
    client._cleanup_ephemeral_output(
        {"filename": local_output.name, "subfolder": "", "type": "output"},
        "request-cleanup",
    )
    assert not local_output.exists()


def test_legacy_mode_does_not_change_existing_output_cleanup(tmp_path):
    settings = replace(
        Settings(),
        model_source_mode="network_volume",
        runtime_root=tmp_path / "runtime",
    )
    settings.ensure_runtime_dirs()
    client = ComfyUIClient(settings)
    local_output = settings.output_dir / "legacy-result.mp4"
    local_output.write_bytes(b"video")
    client._cleanup_ephemeral_output(
        {"filename": local_output.name, "subfolder": "", "type": "output"},
        "request-legacy",
    )
    assert local_output.is_file()


def test_cached_cleanup_rejects_path_outside_controlled_roots(tmp_path):
    settings = replace(
        Settings(),
        model_source_mode="cached_model",
        runtime_root=tmp_path / "runtime",
    )
    settings.ensure_runtime_dirs()
    outside = tmp_path / "must-remain.mp4"
    outside.write_bytes(b"video")
    with pytest.raises(OutputError, match="fora da raiz efêmera"):
        ComfyUIClient(settings)._cleanup_ephemeral_output(
            {"filename": outside.name, "subfolder": "..", "type": "output"},
            "request-traversal",
        )
    assert outside.is_file()
