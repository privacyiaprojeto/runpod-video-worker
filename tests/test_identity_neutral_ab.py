import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from privacy_worker.config import Settings
from privacy_worker.errors import ContractError
from privacy_worker.identity_ab import CONTRACT_VERSION, parse_identity_ab_request
from privacy_worker.workflows import prepare_workflow

ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 64
TRIGGER = "prv_actor_767f0277_v1"


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
            "identity": {
                "trigger_token": TRIGGER,
                "reference_asset_id": "face-front-asset",
                "reference_sha256": SHA,
            },
            "adapter": {"bucket": "privacy-media", "key": "identity/adapter.safetensors", "sha256": SHA},
            "sampling": {
                "seed": 99, "width": 832, "height": 480, "fps": 16,
                "frames": 17, "steps": 30, "denoise": 0.85,
                "branch_b_denoise": 0.85, "lora_strength": 0.65,
            },
            "prompt": {
                "positive": "adult man walking naturally in a neutral studio, full body visible",
                "positive_b": f"{TRIGGER}, adult man walking naturally in a neutral studio, full body visible",
                "negative": "identity mismatch, artifacts",
            },
            "smoke": {
                "enabled": True, "one_shot": True, "max_jobs": 1,
                "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            },
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


def test_h12_contract_and_graph_are_exact():
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
    assert request.trigger_token == TRIGGER
    assert request.positive_prompt_b.startswith(TRIGGER)
    assert request.branch_a_denoise == 0.85
    assert request.branch_b_denoise == 0.85
    assert p["4"]["inputs"]["text"] == event()["input"]["prompt"]["positive"]
    assert p["21"]["inputs"]["text"].startswith(TRIGGER)
    assert p["8"]["inputs"]["control_video"] == ["6", 0]
    assert p["19"]["inputs"]["control_video"] == ["6", 0]
    assert p["19"]["inputs"]["reference_image"] == ["18", 0]
    assert p["19"]["inputs"]["positive"] == ["21", 0]
    assert "20" not in p
    assert all(node.get("class_type") != "PrivacyMotionOnlyStructure" for node in p.values())
    assert p["9"]["inputs"]["denoise"] == 0.85
    assert p["14"]["inputs"]["denoise"] == 0.85
    for sampler_key in ("seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"):
        assert p["9"]["inputs"][sampler_key] == p["14"]["inputs"][sampler_key]
    assert p["13"]["inputs"]["strength_model"] == 0.65
    assert p["14"]["inputs"]["model"] == ["13", 0]
    assert prepared.output_nodes == ("12", "17")


def test_denoise_mismatch_is_fail_closed():
    mismatched = event()
    mismatched["input"]["sampling"]["denoise"] = 1.0
    with pytest.raises(ContractError, match="Parâmetros A/B divergentes"):
        parse_identity_ab_request(mismatched)

def test_trigger_token_is_fail_closed_and_branch_b_only():
    missing = event()
    missing["input"]["prompt"]["positive_b"] = "adult man"
    with pytest.raises(ContractError, match="trigger token exato"):
        parse_identity_ab_request(missing)
    leaked = event()
    leaked["input"]["prompt"]["positive"] = f"{TRIGGER}, baseline"
    with pytest.raises(ContractError, match="exclusivo do ramo B"):
        parse_identity_ab_request(leaked)


def test_explicit_kyc_asset_and_sha_must_match():
    bad_asset = event()
    bad_asset["input"]["identity"]["reference_asset_id"] = "another"
    with pytest.raises(ContractError, match="asset_id"):
        parse_identity_ab_request(bad_asset)
    bad_sha = event()
    bad_sha["input"]["identity"]["reference_sha256"] = "b" * 64
    with pytest.raises(ContractError, match="checksum"):
        parse_identity_ab_request(bad_sha)


def test_branch_b_must_use_private_face_front_kyc():
    """branch B must use private face_front KYC."""
    request = parse_identity_ab_request(event())
    assert request.reference_image_ref["system_tag"] == "face_front"
    assert request.reference_image_ref["bucket"] == "privacy-media"

    wrong_tag = event()
    wrong_tag["input"]["reference_image"]["system_tag"] = "profile"
    with pytest.raises(ContractError, match="face_front"):
        parse_identity_ab_request(wrong_tag)

    public_bucket = event()
    public_bucket["input"]["reference_image"]["bucket"] = "public-media"
    with pytest.raises(ContractError, match="bucket privado aprovado"):
        parse_identity_ab_request(public_bucket)


def test_one_shot_lock_is_scoped_by_request_id(tmp_path):
    from privacy_worker.identity_ab import reserve_one_shot
    first_event = event(); first_event["input"]["request_id"] = "ab-request-001"
    second_event = event(); second_event["input"]["request_id"] = "ab-request-002"
    first = parse_identity_ab_request(first_event)
    second = parse_identity_ab_request(second_event)
    settings = replace(Settings(), runtime_root=tmp_path / "runtime")
    first_path = reserve_one_shot(first, settings)
    second_path = reserve_one_shot(second, settings)
    assert first_path != second_path
    assert first_path.parent == second_path.parent
    assert json.loads(first_path.read_text(encoding="utf-8"))["request_id"] == "ab-request-001"


def test_same_request_id_is_still_fail_closed(tmp_path):
    from privacy_worker.identity_ab import reserve_one_shot
    payload = event(); payload["input"]["request_id"] = "ab-request-duplicate"
    request = parse_identity_ab_request(payload)
    settings = replace(Settings(), runtime_root=tmp_path / "runtime")
    reserve_one_shot(request, settings)
    with pytest.raises(ContractError, match="request_id A/B já foi reservado"):
        reserve_one_shot(request, settings)


def test_legacy_adapter_scoped_lock_does_not_block_authorized_new_request(tmp_path):
    from privacy_worker.identity_ab import reserve_one_shot
    payload = event(); payload["input"]["request_id"] = "ab-request-after-legacy-lock"
    request = parse_identity_ab_request(payload)
    settings = replace(Settings(), runtime_root=tmp_path / "runtime")
    legacy_root = settings.runtime_root / "identity-ab-locks"; legacy_root.mkdir(parents=True, exist_ok=True)
    legacy_path = legacy_root / f"{request.actor_profile_id}_{request.training_run_id}_{request.adapter_id}.json"
    legacy_path.write_text(json.dumps({"request_id": "historical", "status": "completed"}), encoding="utf-8")
    new_path = reserve_one_shot(request, settings)
    assert legacy_path.exists() and new_path.exists() and new_path != legacy_path
