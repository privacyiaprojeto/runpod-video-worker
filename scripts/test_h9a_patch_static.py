#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def main() -> int:
    handler = read("handler.py")
    namespace = read("privacy_worker/lora_namespace.py")
    visual = read("privacy_worker/visual_ab.py")
    identity = read("privacy_worker/identity_ab.py")
    errors = read("privacy_worker/errors.py")
    custom = read("custom_nodes/privacy_lora_attestation/nodes.py")
    docker = read("Dockerfile")
    requirements = read("requirements.txt")
    requirements_dev = read("requirements-dev.txt")
    ci_workflow = read(".github/workflows/build.yml")
    workflow = json.loads(read("workflows/wan-2.1-v2v-identity-ab-v1.json"))

    checks = {
        "converter exists": (ROOT / "privacy_worker/lora_namespace.py").is_file(),
        "preflight exists": (ROOT / "scripts/preflight_identity_lora_namespace.py").is_file(),
        "visual guard exists": (ROOT / "privacy_worker/visual_ab.py").is_file(),
        "custom node exists": (ROOT / "custom_nodes/privacy_lora_attestation/nodes.py").is_file(),
        "DiffSynth PEFT A accepted": "lora_(?P<side>[AB])" in namespace,
        "PEFT default suffix accepted": "(?:\\.default)?\\.weight" in namespace,
        "approved target scope exact": "cross_attn.q" in namespace and "ffn.2" in namespace,
        "model header shapes checked": "all_shapes_compatible" in namespace,
        "full model coverage checked": "full_target_coverage" in namespace,
        "model layout cache pinned": "model_layout_sha256" in namespace and "model_layout_sha256" in identity,
        "compiled model alias supported": "_orig_mod" in namespace and "_orig_mod" in custom,
        "PEFT A translated to down": "lora_down.weight" in namespace,
        "PEFT B translated to up": "lora_up.weight" in namespace,
        "translated alpha emitted": 'f"{prefix}.alpha"' in namespace,
        "translated checksum emitted": "translated_sha256" in namespace,
        "generic Comfy Wan prefix emitted": "diffusion_model.{module}" in namespace,
        "workflow uses attested loader": workflow["prompt"]["13"]["class_type"] == "PrivacyAttestedLoraLoaderModelOnly",
        "generic loader removed from identity node": workflow["prompt"]["13"]["class_type"] != "LoraLoaderModelOnly",
        "runtime attestation bound": "lora_attestation_name" in workflow["bindings"],
        "loader builds actual model key map": "model_lora_keys_unet" in custom,
        "loader converts Comfy format": "convert_lora" in custom,
        "loader resolves actual patches": "load_lora" in custom,
        "loader attaches patches": "add_patches" in custom,
        "zero patches fail fast": "LORA_NOT_APPLIED" in custom,
        "namespace mismatch fail fast": "LORA_KEY_FORMAT_MISMATCH" in custom,
        "runtime patch count emitted": "patched_model_key_count" in custom,
        "runtime count equals converted pairs": "expected_pair_count=int(conversion_attestation" in handler,
        "handler reads patch attestation": "read_runtime_lora_attestation" in handler,
        "handler performs A/B diff": "compare_ab_videos" in handler,
        "visual guard uses decoded SSIM": "[0:v][1:v]ssim" in visual,
        "identical outputs have stable code": "AB_OUTPUTS_IDENTICAL" in errors,
        "visual guard runs before R2 upload": handler.index("compare_ab_videos") < handler.index("publish_private_named_output", handler.index("def _handle_identity_ab")),
        "false success blocked by attestation": handler.index("read_runtime_lora_attestation") < handler.index("identity_neutral_ab_completed"),
        "QA remains not auto approved": '"approval_allowed": False' in handler,
        "automatic retry remains disabled": "automatic_retry=False" in handler,
        "custom node copied into image": "custom_nodes/privacy_lora_attestation" in docker,
        "custom node compiled in image": "/opt/ComfyUI/custom_nodes/privacy_lora_attestation" in docker,
        "safetensors runtime dependency declared": "safetensors>=" in requirements,
        "CI NumPy dependency pinned for safetensors tests": "numpy==2.1.3" in requirements_dev,
        "CI installs pinned CPU torch before pytest": (
            "Install CPU PyTorch for contract tests" in ci_workflow
            and "https://download.pytorch.org/whl/cpu" in ci_workflow
            and "torch==2.8.0" in ci_workflow
            and ci_workflow.index("Install CPU PyTorch for contract tests")
            < ci_workflow.index("Run static contract tests")
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    report = {
        "status": "D3_6H9A_LORA_NAMESPACE_FAILFAST_TEST_READY" if not failed else "D3_6H9A_LORA_NAMESPACE_FAILFAST_TEST_BLOCKED",
        "checksPassed": sum(bool(value) for value in checks.values()),
        "checksTotal": len(checks),
        "checks": [{"name": name, "passed": bool(value)} for name, value in checks.items()],
        "blockers": failed,
        "safety": {
            "staticValidationOnly": True,
            "networkCalled": False,
            "runPodCalled": False,
            "gpuStarted": False,
            "jobSubmitted": False,
            "endpointUpdated": False,
            "volumeUpdated": False,
            "r2MutationExecuted": False,
        },
    }
    print(json.dumps(report, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
