import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_h9a_worker_has_converter_attested_loader_and_visual_guard():
    handler = (ROOT / "handler.py").read_text(encoding="utf-8")
    identity = (ROOT / "privacy_worker" / "identity_ab.py").read_text(encoding="utf-8")
    namespace = (ROOT / "privacy_worker" / "lora_namespace.py").read_text(encoding="utf-8")
    visual = (ROOT / "privacy_worker" / "visual_ab.py").read_text(encoding="utf-8")
    custom = (ROOT / "custom_nodes" / "privacy_lora_attestation" / "nodes.py").read_text(encoding="utf-8")
    docker = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    workflow = json.loads((ROOT / "workflows" / "wan-2.1-v2v-identity-ab-v1.json").read_text())

    assert "(?:\\.default)?\\.weight" in namespace
    assert "diffusion_model.{module}" in namespace
    assert "lora_down.weight" in namespace and "lora_up.weight" in namespace
    assert "all_shapes_compatible" in namespace
    assert "model_layout_sha256" in namespace
    assert "PrivacyAttestedLoraLoaderModelOnly" in custom
    assert "patched_model_key_count" in custom
    assert "LORA_NOT_APPLIED" in custom
    assert workflow["prompt"]["13"]["class_type"] == "PrivacyAttestedLoraLoaderModelOnly"
    assert "lora_attestation_name" in workflow["bindings"]
    assert "compare_ab_videos" in handler
    assert "AB_OUTPUTS_IDENTICAL" in (ROOT / "privacy_worker" / "errors.py").read_text()
    assert "ffmpeg" in visual and "ssim" in visual
    assert "publish_private_named_output" in handler
    assert handler.index("compare_ab_videos") < handler.index("publish_private_named_output", handler.index("def _handle_identity_ab"))
    assert "custom_nodes/privacy_lora_attestation" in docker
    assert "automatic_retry=False" in handler
    assert "identity_neutral_ab_completed" in handler
    assert "lora_attestation" in handler
    assert "read_runtime_lora_attestation" in identity
