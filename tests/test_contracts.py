import copy
import json
from pathlib import Path

import pytest

from privacy_worker.contracts import parse_production_request
from privacy_worker.errors import ContractError

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_i2v_contract_uses_approved_reference_and_normalizes_wan_dimensions():
    request = parse_production_request(load("i2v.json"))
    assert request.engine == "wan-2.1-i2v"
    assert request.task == "video.i2v"
    assert request.source_image_url.endswith("kyc-front.png")
    assert request.base_video_url is None
    assert request.width == 832
    assert request.height == 480
    assert request.frames == 49
    assert request.workflow_id == "wan-2.1-i2v-v1"


def test_v2v_contract_requires_base_video_and_ignores_unapproved_source_override():
    request = parse_production_request(load("v2v.json"))
    assert request.engine == "wan-2.1-v2v"
    assert request.base_video_url.endswith("base-scene.mp4")
    assert request.source_image_url.endswith("kyc-close.png")
    assert "unapproved" not in request.source_image_url
    assert request.frames == 81


def test_contract_rejects_missing_safety_guards():
    payload = load("i2v.json")
    payload["input"]["safety"]["qa_required"] = False
    with pytest.raises(ContractError):
        parse_production_request(payload)


def test_contract_rejects_engine_task_mismatch():
    payload = load("i2v.json")
    payload["input"]["task"] = "video.v2v"
    with pytest.raises(ContractError):
        parse_production_request(payload)


def test_contract_rejects_identity_without_approved_reference_media():
    payload = load("i2v.json")
    payload["input"]["identity"]["actors"] = []
    with pytest.raises(ContractError):
        parse_production_request(payload)
