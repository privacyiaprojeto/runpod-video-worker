import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from privacy_worker.config import Settings
from privacy_worker.errors import ContractError
from privacy_worker.identity_motion_abc import (
    CONTRACT_VERSION,
    CONTROL_REPRESENTATION,
    parse_identity_motion_abc_request,
)
from privacy_worker.workflows import prepare_workflow

ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 64
TRIGGER = "prv_actor_767f0277_v1"


def event():
    return {
        "input": {
            "contract_version": CONTRACT_VERSION,
            "execution_mode": "controlled_identity_motion_abc",
            "request_id": "motion-abc-001",
            "actor_profile_id": "actor",
            "training_run_id": "run",
            "adapter_id": "adapter",
            "base_video": {
                "bucket": "privacy-media",
                "key": "qa-assets/neutral-motion-01.mp4",
                "sha256": SHA,
            },
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
            "adapter": {
                "bucket": "privacy-media",
                "key": "identity/adapter.safetensors",
                "sha256": SHA,
            },
            "control": {
                "representation": CONTROL_REPRESENTATION,
                "derive_from_base_video": True,
                "raw_rgb_control_allowed": False,
                "same_control_all_branches": True,
            },
            "sampling": {
                "seed": 99,
                "width": 832,
                "height": 480,
                "fps": 16,
                "frames": 17,
                "steps": 30,
                "denoise": 0.85,
                "branch_b_denoise": 0.85,
                "branch_c_denoise": 0.85,
                "lora_strength": 0.65,
            },
            "prompt": {
                "positive": "adult man walking naturally in a neutral studio, full body visible",
                "positive_identity": f"{TRIGGER}, adult man walking naturally in a neutral studio, full body visible",
                "negative": "identity mismatch, artifacts",
            },
            "smoke": {
                "enabled": True,
                "one_shot": True,
                "max_jobs": 1,
                "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            },
            "safety": {
                "private_storage_only": True,
                "public_urls_forbidden": True,
                "automatic_retry_allowed": False,
                "one_shot_smoke": True,
                "kyc_reference_required": True,
                "kyc_reference_private_only": True,
                "kyc_reference_baseline_forbidden": True,
                "kyc_reference_identity_branches_only": True,
                "kyc_reference_persistence_forbidden": True,
                "product_release_allowed": False,
            },
        }
    }


def test_motion_abc_contract_and_graph_isolate_lora():
    request = parse_identity_motion_abc_request(event())
    settings = replace(Settings(), workflow_root=ROOT / "workflows")
    prepared = prepare_workflow(
        request=request,
        source_image_filename="face-front.jpg",
        base_video_filename="neutral-softedge.mp4",
        output_prefix="privacy/motion-abc",
        settings=settings,
        lora_filename="identity.safetensors",
        lora_attestation_name="motion-abc-001",
    )
    p = prepared.prompt

    assert request.trigger_token == TRIGGER
    assert request.positive_prompt_b.startswith(TRIGGER)
    assert p["8"]["inputs"]["control_video"] == ["6", 0]
    assert p["19"]["inputs"]["control_video"] == ["6", 0]
    assert p["19"]["inputs"]["reference_image"] == ["18", 0]

    # A: no KYC, no trigger, no LoRA.
    assert p["9"]["inputs"]["model"] == ["1", 0]
    assert p["9"]["inputs"]["positive"] == ["8", 0]

    # B: KYC + trigger, but still base model.
    assert p["14"]["inputs"]["model"] == ["1", 0]
    assert p["14"]["inputs"]["positive"] == ["19", 0]

    # C: exact same B conditioning, only model changes to attested LoRA.
    assert p["23"]["inputs"]["model"] == ["13", 0]
    assert p["23"]["inputs"]["positive"] == ["19", 0]

    for sampler_key in ("seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"):
        assert p["9"]["inputs"][sampler_key] == p["14"]["inputs"][sampler_key]
        assert p["14"]["inputs"][sampler_key] == p["23"]["inputs"][sampler_key]

    assert p["13"]["inputs"]["strength_model"] == 0.65
    assert prepared.output_nodes == ("12", "17", "26")


def test_motion_abc_rejects_raw_rgb_control():
    payload = event()
    payload["input"]["control"]["raw_rgb_control_allowed"] = True
    with pytest.raises(ContractError, match="controle estrutural"):
        parse_identity_motion_abc_request(payload)


def test_motion_abc_rejects_parameter_drift():
    payload = event()
    payload["input"]["sampling"]["branch_c_denoise"] = 1.0
    with pytest.raises(ContractError, match="Parâmetros motion A/B/C divergentes"):
        parse_identity_motion_abc_request(payload)


def test_motion_abc_trigger_is_identity_branches_only():
    payload = event()
    payload["input"]["prompt"]["positive"] = f"{TRIGGER}, baseline"
    with pytest.raises(ContractError, match="não pode vazar"):
        parse_identity_motion_abc_request(payload)


def test_workflow_declares_softedge_abc_revision():
    envelope = json.loads((ROOT / "workflows" / "wan-2.1-v2v-identity-motion-abc-v1.json").read_text(encoding="utf-8"))
    assert envelope["revision"] == "M4-identity-motion-abc-softedge-v1"
    assert envelope["methodology_hotfix"] == "M4-HF-softedge-abc-lora-isolation-v1"
    assert envelope["output_nodes"] == ["12", "17", "26"]
