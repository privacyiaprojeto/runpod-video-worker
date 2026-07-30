#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from privacy_worker.lora_namespace import inspect_diffsynth_peft_lora, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only DiffSynth PEFT -> ComfyUI Wan LoRA preflight")
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    args = parser.parse_args()
    report = inspect_diffsynth_peft_lora(args.adapter, args.model)
    report.pop("_pairs", None)
    report["adapter_sha256"] = sha256_file(args.adapter)
    report["adapter_path"] = str(args.adapter)
    report["model_path"] = str(args.model)
    report["status"] = "D3_6H9A_LORA_NAMESPACE_PREFLIGHT_READY"
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
