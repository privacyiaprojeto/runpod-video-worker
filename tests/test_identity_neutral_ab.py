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
    return {"input": {"contract_version": CONTRACT_VERSION, "execution_mode":"controlled_identity_neutral_ab", "request_id":"ab-001", "actor_profile_id":"actor", "training_run_id":"run", "adapter_id":"adapter", "base_video":{"bucket":"privacy-media","key":"qa-assets/neutral-motion-01.mp4","sha256":SHA}, "adapter":{"bucket":"privacy-media","key":"identity/adapter.safetensors","sha256":SHA}, "sampling":{"seed":99,"width":832,"height":480,"fps":16,"frames":17,"steps":30,"denoise":1.0,"lora_strength":0.65}, "prompt":{"positive":"adult person walking in a neutral studio","negative":"artifacts"}, "smoke":{"enabled":True,"one_shot":True,"max_jobs":1,"expires_at":(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()}, "safety":{"private_storage_only":True,"public_urls_forbidden":True,"automatic_retry_allowed":False,"one_shot_smoke":True,"kyc_source_forbidden":True,"product_release_allowed":False}}}


def test_neutral_ab_contract_and_graph_are_exact():
    request=parse_identity_ab_request(event())
    settings=replace(Settings(), workflow_root=ROOT/"workflows")
    prepared=prepare_workflow(request=request, source_image_filename=None, base_video_filename="neutral.mp4", output_prefix="privacy/ab", settings=settings, lora_filename="identity.safetensors")
    p=prepared.prompt
    assert p["6"]["inputs"]["video"] == "neutral.mp4"
    assert p["7"]["inputs"]["image"] == ["6",0]
    assert p["9"]["inputs"]["seed"] == p["14"]["inputs"]["seed"] == 99
    assert p["9"]["inputs"]["denoise"] == p["14"]["inputs"]["denoise"] == 1.0
    assert p["9"]["inputs"]["model"] == ["1",0]
    assert p["13"]["class_type"] == "LoraLoaderModelOnly"
    assert p["13"]["inputs"]["strength_model"] == 0.65
    assert p["14"]["inputs"]["model"] == ["13",0]
    assert prepared.output_nodes == ("12","17")
    serialized=json.dumps(p).lower()
    assert "kyc" not in serialized
