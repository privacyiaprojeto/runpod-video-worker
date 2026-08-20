from pathlib import Path
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
workflow_path = ROOT / "workflows" / "wan-2.1-v2v-identity-motion-abc-v1.json"
handler = (ROOT / "handler.py").read_text(encoding="utf-8")
contract = (ROOT / "privacy_worker" / "identity_motion_abc.py").read_text(encoding="utf-8")
validator = (ROOT / "scripts" / "validate_workflows.py").read_text(encoding="utf-8")
workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
p = workflow["prompt"]

docker_path = ROOT / "Dockerfile"
if docker_path.is_file():
    docker = docker_path.read_text(encoding="utf-8")
    softedge_gate_available = "edgedetect" in docker
else:
    probe = subprocess.run(
        ["ffmpeg", "-hide_banner", "-filters"],
        capture_output=True,
        text=True,
        check=False,
    )
    softedge_gate_available = (
        probe.returncode == 0
        and "edgedetect" in f"{probe.stdout}\n{probe.stderr}"
    )

checks = {
    "contract_version": 'privacy-identity-motion-abc-v1' in contract,
    "worker_routes_contract": "IDENTITY_MOTION_ABC_CONTRACT_VERSION" in handler,
    "softedge_runtime": "derive_softedge_control" in handler and "edgedetect" in contract,
    "softedge_gate_available": softedge_gate_available,
    "official_validator_includes_motion_abc": '"wan-2.1-v2v-identity-motion-abc-v1.json": "wan-2.1-v2v"' in validator,
    "three_outputs": workflow["output_nodes"] == ["12", "17", "26"],
    "same_control_a_b": p["8"]["inputs"]["control_video"] == p["19"]["inputs"]["control_video"] == ["6", 0],
    "b_without_lora": p["14"]["inputs"]["model"] == ["1", 0],
    "c_with_lora": p["23"]["inputs"]["model"] == ["13", 0],
    "same_identity_conditioning_b_c": p["14"]["inputs"]["positive"] == p["23"]["inputs"]["positive"] == ["19", 0],
    "same_seed": p["9"]["inputs"]["seed"] == p["14"]["inputs"]["seed"] == p["23"]["inputs"]["seed"] == 99,
    "same_denoise": p["9"]["inputs"]["denoise"] == p["14"]["inputs"]["denoise"] == p["23"]["inputs"]["denoise"] == 0.85,
    "lora_strength": p["13"]["inputs"]["strength_model"] == 0.65,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit("IDENTITY_MOTION_ABC_STATIC_FAILED: " + ",".join(failed))
print("IDENTITY_MOTION_ABC_STATIC_READY", len(checks), "/", len(checks))
