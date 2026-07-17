import importlib
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "i2v.json"


class FakeProcessManager:
    def ensure_started(self, request_id: str) -> None:
        assert request_id == "req-i2v-001"

    def interrupt(self, request_id: str) -> None:
        raise AssertionError("interrupt should not be called in successful mock")


class FakeClient:
    def queue_prompt(self, prompt, request_id: str) -> str:
        assert prompt["54"]["inputs"]["image"].startswith("privacy_req-i2v-001_identity_")
        return "prompt-001"

    def wait_for_history(self, prompt_id: str, request_id: str):
        assert prompt_id == "prompt-001"
        return {"outputs": {"28": {"gifs": [{"filename": "mock.mp4", "type": "output"}]}}}

    def download_output(self, *, record, output_nodes, destination: Path, request_id: str):
        assert output_nodes == ("28",)
        destination.write_bytes(b"mock-mp4")
        return destination


def test_handler_consumes_canonical_contract_without_gpu(monkeypatch, tmp_path):
    fake_runpod = types.ModuleType("runpod")
    fake_runpod.serverless = types.SimpleNamespace(start=lambda payload: payload)
    monkeypatch.setitem(sys.modules, "runpod", fake_runpod)
    monkeypatch.setenv("APP_ROOT", str(ROOT))
    monkeypatch.setenv("WORKFLOW_ROOT", str(ROOT / "workflows"))
    monkeypatch.setenv("RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("COMFYUI_ROOT", str(tmp_path / "ComfyUI"))
    monkeypatch.setenv("SKIP_MODEL_VALIDATION", "true")

    for name in ["handler", "privacy_worker.config"]:
        sys.modules.pop(name, None)
    module = importlib.import_module("handler")

    def fake_download_media(*, destination_dir: Path, stem: str, fallback_extension: str, **kwargs):
        path = destination_dir / f"{stem}{fallback_extension}"
        path.write_bytes(b"mock-input")
        return path

    monkeypatch.setattr(module, "download_media", fake_download_media)
    monkeypatch.setattr(module, "_process_manager", FakeProcessManager())
    monkeypatch.setattr(module, "_client", FakeClient())
    monkeypatch.setattr(
        module,
        "publish_output",
        lambda path, settings, request_id: {
            "video_base64": "bW9jaw==",
            "mime_type": "video/mp4",
            "extension": "mp4",
            "size_bytes": path.stat().st_size,
            "private_output_only": True,
            "qa_required": True,
        },
    )

    event = json.loads(FIXTURE.read_text(encoding="utf-8"))
    response = module.handler(event)
    assert response["contract_version"] == "privacy-production-spec-v1"
    assert response["engine"] == "wan-2.1-i2v"
    assert response["workflow_id"] == "wan-2.1-i2v-v1"
    assert response["private_output_only"] is True
    assert response["qa_required"] is True
