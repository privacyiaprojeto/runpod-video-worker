from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
identity = (ROOT / "privacy_worker" / "identity_ab.py").read_text(encoding="utf-8")
lora_namespace = (ROOT / "privacy_worker" / "lora_namespace.py").read_text(encoding="utf-8")
handler = (ROOT / "handler.py").read_text(encoding="utf-8")
workflow = json.loads((ROOT / "workflows" / "wan-2.1-v2v-identity-ab-v1.json").read_text(encoding="utf-8"))
test = (ROOT / "tests" / "test_identity_neutral_ab.py").read_text(encoding="utf-8")

p = workflow["prompt"]
b = workflow["bindings"]
top_level = lora_namespace.split("def _header", 1)[0]
checks = {
    "contract_has_reference_image": "reference_image_ref: dict[str, str]" in identity,
    "contract_requires_face_front": 'system_tag != "face_front"' in identity,
    "contract_actor_scopes_kyc": 'actor_scope = f"/actor-{actor_profile_id.lower()}/"' in identity,
    "contract_private_kyc_only": 'kyc_reference_private_only' in identity,
    "contract_branch_b_only": 'kyc_reference_branch_b_only' in identity,
    "legacy_kyc_forbidden_removed": '"kyc_source_forbidden": True' not in identity,
    "handler_downloads_kyc": 'request.reference_image_ref' in handler and 'role="kyc_face_front"' in handler,
    "handler_passes_source_image": 'source_image_filename=reference_input.name' in handler,
    "handler_keeps_neutral_video": 'base_video_filename=base_input.name' in handler,
    "handler_reports_qa_only": '"approval_allowed": False' in handler,
    "branch_a_uses_neutral_first_frame": p["8"]["inputs"]["reference_image"] == ["7", 0],
    "branch_b_has_dedicated_kyc_loader": p["18"]["class_type"] == "LoadImage",
    "branch_b_uses_same_control_video": p["19"]["inputs"]["control_video"] == ["6", 0],
    "branch_b_uses_kyc_reference": p["19"]["inputs"]["reference_image"] == ["18", 0],
    "branch_b_sampler_uses_dedicated_conditioning": p["14"]["inputs"]["latent_image"] == ["19", 2],
    "branch_b_trim_uses_dedicated_conditioning": p["15"]["inputs"]["trim_amount"] == ["19", 3],
    "source_image_binding_exists": b["source_image"] == {"node_id": "18", "input": "image"},
    "lora_strength_stays_065": p["13"]["inputs"]["strength_model"] == 0.65,
    "tests_cover_face_front": 'branch B must use private face_front KYC' in test,
    "automatic_retry_stays_disabled": '"automatic_retry": False' in identity,
    "torch_not_imported_eagerly": "import torch" not in top_level,
    "safetensors_not_imported_eagerly": "from safetensors" not in top_level,
    "safe_open_loaded_inside_header_only": "Runtime safetensors indisponível para inspecionar a LoRA Wan." in lora_namespace,
    "torch_loaded_inside_conversion_only": "Runtime PyTorch/safetensors.torch indisponível para materializar a LoRA Wan." in lora_namespace,
    "runtime_import_marker_present": "D3.6H10-HF2 — lazy safetensors/PyTorch runtime imports" in lora_namespace,
    "runtime_missing_dependency_fails_closed": "missing_dependency" in lora_namespace,
}

_real_import = builtins.__import__
def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch" or name.startswith("torch.") or name == "safetensors" or name.startswith("safetensors."):
        root = name.split(".", 1)[0]
        raise ModuleNotFoundError(f"{root} blocked by D3.6H10-HF2 lightweight import test", name=root)
    return _real_import(name, globals, locals, fromlist, level)

lightweight_error = None
try:
    builtins.__import__ = _guarded_import
    sys.path.insert(0, str(ROOT))
    from privacy_worker.identity_ab import CONTRACT_VERSION as imported_contract_version
    checks["identity_contract_imports_without_runtime_dependencies"] = imported_contract_version == "privacy-identity-neutral-ab-v1"
except Exception as exc:
    lightweight_error = f"{type(exc).__name__}: {exc}"
    checks["identity_contract_imports_without_runtime_dependencies"] = False
finally:
    builtins.__import__ = _real_import
    if sys.path and sys.path[0] == str(ROOT):
        sys.path.pop(0)

failed = [name for name, ok in checks.items() if not ok]
print(json.dumps({
    "status": "D3_6H10_HF2_KYC_REFERENCE_LAZY_RUNTIME_READY" if not failed else "D3_6H10_HF2_KYC_REFERENCE_LAZY_RUNTIME_BLOCKED",
    "checksPassed": len(checks) - len(failed),
    "checksTotal": len(checks),
    "checks": checks,
    "blockers": failed,
    "lightweightImportError": lightweight_error,
    "safety": {
        "staticValidationOnly": True,
        "networkCalled": False,
        "runPodCalled": False,
        "gpuStarted": False,
        "r2MutationExecuted": False,
        "databaseMutationExecuted": False,
        "frontendChanged": False,
        "automaticRetryCreated": False,
        "productReleased": False,
    },
}, indent=2, ensure_ascii=False))
raise SystemExit(1 if failed else 0)
