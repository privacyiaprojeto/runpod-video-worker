import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from privacy_worker.config import Settings
from privacy_worker.identity_ab import CONTRACT_VERSION, parse_identity_ab_request
from privacy_worker.workflows import prepare_workflow

ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 64


def event():
    return {
        "input": {
            "contract_version": CONTRACT_VERSION,
            "execution_mode": "controlled_identity_neutral_ab",
            "request_id": "ab-001",
            "actor_profile_id": "actor",
            "training_run_id": "run",
            "adapter_id": "adapter",
            "base_video": {"bucket": "privacy-media", "key": "qa-assets/neutral-motion-01.mp4", "sha256": SHA},
            "reference_image": {
                "bucket": "privacy-media",
                "key": "vault/actor-mapping/2026/07/30/actor-actor/case-test/face-front.jpg",
                "sha256": SHA,
                "system_tag": "face_front",
                "asset_id": "face-front-asset",
            },
            "adapter": {"bucket": "privacy-media", "key": "identity/adapter.safetensors", "sha256": SHA},
            "sampling": {"seed": 99, "width": 832, "height": 480, "fps": 16, "frames": 17, "steps": 30, "denoise": 1.0, "lora_strength": 0.65},
            "prompt": {"positive": "adult person walking in a neutral studio", "negative": "artifacts"},
            "smoke": {"enabled": True, "one_shot": True, "max_jobs": 1, "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()},
            "safety": {
                "private_storage_only": True,
                "public_urls_forbidden": True,
                "automatic_retry_allowed": False,
                "one_shot_smoke": True,
                "kyc_reference_required": True,
                "kyc_reference_private_only": True,
                "kyc_reference_branch_b_only": True,
                "kyc_reference_persistence_forbidden": True,
                "product_release_allowed": False,
            },
        }
    }


def test_neutral_ab_contract_and_graph_are_exact():
    """branch B must use private face_front KYC while branch A remains neutral."""
    request = parse_identity_ab_request(event())
    settings = replace(Settings(), workflow_root=ROOT / "workflows")
    prepared = prepare_workflow(
        request=request,
        source_image_filename="face-front.jpg",
        base_video_filename="neutral.mp4",
        output_prefix="privacy/ab",
        settings=settings,
        lora_filename="identity.safetensors",
        lora_attestation_name="ab-001",
    )
    p = prepared.prompt
    assert request.reference_image_ref["system_tag"] == "face_front"
    assert p["6"]["inputs"]["video"] == "neutral.mp4"
    assert p["7"]["inputs"]["image"] == ["6", 0]
    assert p["8"]["inputs"]["control_video"] == ["6", 0]
    assert p["8"]["inputs"]["reference_image"] == ["7", 0]
    assert p["18"]["inputs"]["image"] == "face-front.jpg"
    assert p["19"]["inputs"]["control_video"] == ["6", 0]
    assert p["19"]["inputs"]["reference_image"] == ["18", 0]
    assert p["9"]["inputs"]["seed"] == p["14"]["inputs"]["seed"] == 99
    assert p["9"]["inputs"]["denoise"] == p["14"]["inputs"]["denoise"] == 1.0
    assert p["9"]["inputs"]["model"] == ["1", 0]
    assert p["13"]["class_type"] == "PrivacyAttestedLoraLoaderModelOnly"
    assert p["13"]["inputs"]["attestation_name"] == "ab-001"
    assert p["13"]["inputs"]["strength_model"] == 0.65
    assert p["14"]["inputs"]["model"] == ["13", 0]
    assert p["14"]["inputs"]["latent_image"] == ["19", 2]
    assert prepared.output_nodes == ("12", "17")

def test_one_shot_lock_is_scoped_by_request_id(tmp_path):
    from privacy_worker.identity_ab import reserve_one_shot

    first_event = event()
    first_event["input"]["request_id"] = "ab-request-001"
    second_event = event()
    second_event["input"]["request_id"] = "ab-request-002"

    first = parse_identity_ab_request(first_event)
    second = parse_identity_ab_request(second_event)
    settings = replace(Settings(), runtime_root=tmp_path / "runtime")

    first_path = reserve_one_shot(first, settings)
    second_path = reserve_one_shot(second, settings)

    assert first_path != second_path
    assert first_path.exists()
    assert second_path.exists()
    assert first_path.parent == second_path.parent

    first_payload = json.loads(first_path.read_text(encoding="utf-8"))
    second_payload = json.loads(second_path.read_text(encoding="utf-8"))
    assert first_payload["lock_version"] == 2
    assert second_payload["lock_version"] == 2
    assert first_payload["request_id"] == "ab-request-001"
    assert second_payload["request_id"] == "ab-request-002"
    assert first_payload["automatic_retry"] is False


def test_same_request_id_is_still_fail_closed(tmp_path):
    import pytest
    from privacy_worker.errors import ContractError
    from privacy_worker.identity_ab import reserve_one_shot

    payload = event()
    payload["input"]["request_id"] = "ab-request-duplicate"
    request = parse_identity_ab_request(payload)
    settings = replace(Settings(), runtime_root=tmp_path / "runtime")

    reserve_one_shot(request, settings)
    with pytest.raises(ContractError, match="request_id A/B já foi reservado"):
        reserve_one_shot(request, settings)


def test_legacy_adapter_scoped_lock_does_not_block_authorized_new_request(tmp_path):
    from privacy_worker.identity_ab import reserve_one_shot

    payload = event()
    payload["input"]["request_id"] = "ab-request-after-legacy-lock"
    request = parse_identity_ab_request(payload)
    settings = replace(Settings(), runtime_root=tmp_path / "runtime")

    legacy_root = settings.runtime_root / "identity-ab-locks"
    legacy_root.mkdir(parents=True, exist_ok=True)
    legacy_path = legacy_root / (
        f"{request.actor_profile_id}_{request.training_run_id}_{request.adapter_id}.json"
    )
    legacy_path.write_text(
        json.dumps({"request_id": "historical", "status": "completed"}),
        encoding="utf-8",
    )

    new_path = reserve_one_shot(request, settings)

    assert legacy_path.exists()
    assert new_path.exists()
    assert new_path != legacy_path
