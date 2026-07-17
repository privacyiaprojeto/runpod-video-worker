import json
from dataclasses import replace
from pathlib import Path

from privacy_worker.config import Settings
from privacy_worker.contracts import parse_production_request
from privacy_worker.workflows import prepare_workflow

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def load(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def settings():
    return replace(Settings(), workflow_root=ROOT / "workflows")


def test_i2v_workflow_injects_contract_values():
    request = parse_production_request(load("i2v.json"))
    prepared = prepare_workflow(
        request=request,
        source_image_filename="identity.png",
        base_video_filename=None,
        output_prefix="privacy/test/i2v",
        settings=settings(),
    )
    prompt = prepared.prompt
    assert prompt["6"]["inputs"]["text"] == request.positive_prompt
    assert prompt["7"]["inputs"]["text"] == request.negative_prompt
    assert prompt["54"]["inputs"]["image"] == "identity.png"
    assert prompt["51"]["inputs"]["width"] == 832
    assert prompt["51"]["inputs"]["length"] == 49
    assert prompt["28"]["inputs"]["frame_rate"] == 16
    assert prepared.output_nodes == ("28",)


def test_v2v_workflow_injects_base_video_into_all_bound_nodes():
    request = parse_production_request(load("v2v.json"))
    prepared = prepare_workflow(
        request=request,
        source_image_filename="identity.png",
        base_video_filename="base.mp4",
        output_prefix="privacy/test/v2v",
        settings=settings(),
    )
    prompt = prepared.prompt
    assert prompt["6"]["inputs"]["video"] == "base.mp4"
    assert prompt["7"]["inputs"]["image"] == "identity.png"
    assert prompt["6"]["inputs"]["custom_width"] == 832
    assert prompt["8"]["inputs"]["width"] == 832
    assert prompt["6"]["inputs"]["frame_load_cap"] == 81
    assert prompt["8"]["inputs"]["length"] == 81
    assert prompt["12"]["inputs"]["frame_rate"] == 16
    assert prepared.output_nodes == ("12",)
