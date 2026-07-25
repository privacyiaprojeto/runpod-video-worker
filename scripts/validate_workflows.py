#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = ROOT / "workflows"
EXPECTED = {
    "wan-2.1-i2v-v1.json": "wan-2.1-i2v",
    "wan-2.1-v2v-v1.json": "wan-2.1-v2v",
    "wan-2.1-v2v-identity-ab-v1.json": "wan-2.1-v2v",
}


def main() -> int:
    for filename, engine in EXPECTED.items():
        path = WORKFLOW_ROOT / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == "privacy-comfyui-workflow-v1"
        assert payload["engine"] == engine
        assert payload["workflow_version"] == "1"
        assert isinstance(payload["prompt"], dict) and payload["prompt"]
        assert isinstance(payload["bindings"], dict) and payload["bindings"]
        assert isinstance(payload["output_nodes"], list) and payload["output_nodes"]
        for node_id, node in payload["prompt"].items():
            assert isinstance(node_id, str)
            assert isinstance(node, dict) and node.get("class_type")
            assert isinstance(node.get("inputs"), dict)
        for output_node in payload["output_nodes"]:
            assert str(output_node) in payload["prompt"]
    print(json.dumps({"status": "WAN_WORKFLOWS_VALID", "workflows": sorted(EXPECTED)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
