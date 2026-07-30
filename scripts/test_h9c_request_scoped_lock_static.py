from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ROOT / "privacy_worker" / "identity_ab.py"
TESTS = ROOT / "tests" / "test_identity_neutral_ab.py"

identity = IDENTITY.read_text(encoding="utf-8")
tests = TESTS.read_text(encoding="utf-8")

checks = {
    "request scoped path helper exists": "def one_shot_lock_path(" in identity,
    "request digest used": 'hashlib.sha256(request.request_id.encode("utf-8")).hexdigest()' in identity,
    "actor scope retained": "/ _safe_lock_component(request.actor_profile_id)" in identity,
    "run scope retained": "/ _safe_lock_component(request.training_run_id)" in identity,
    "adapter scope retained": "/ _safe_lock_component(request.adapter_id)" in identity,
    "lock version two emitted": '"lock_version": 2' in identity,
    "request id stored": '"request_id": request.request_id' in identity,
    "automatic retry remains false": '"automatic_retry": False' in identity,
    "legacy flat path removed from implementation": 'f"{request.actor_profile_id}_{request.training_run_id}_{request.adapter_id}.json"' not in identity,
    "duplicate request stays fail closed": "Este request_id A/B já foi reservado" in identity,
    "different requests test exists": "test_one_shot_lock_is_scoped_by_request_id" in tests,
    "duplicate request test exists": "test_same_request_id_is_still_fail_closed" in tests,
    "legacy lock compatibility test exists": "test_legacy_adapter_scoped_lock_does_not_block_authorized_new_request" in tests,
}

blockers = [name for name, passed in checks.items() if not passed]
print(json.dumps({
    "status": (
        "D3_6H9C_REQUEST_SCOPED_ONE_SHOT_LOCK_READY"
        if not blockers
        else "D3_6H9C_REQUEST_SCOPED_ONE_SHOT_LOCK_BLOCKED"
    ),
    "checksPassed": sum(1 for value in checks.values() if value),
    "checksTotal": len(checks),
    "checks": checks,
    "blockers": blockers,
    "safety": {
        "staticValidationOnly": True,
        "networkCalled": False,
        "runPodCalled": False,
        "gpuStarted": False,
        "r2MutationExecuted": False,
        "volumeMutationExecuted": False,
    },
}, indent=2, ensure_ascii=False))

if blockers:
    raise SystemExit(1)
