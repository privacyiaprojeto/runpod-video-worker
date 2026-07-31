from __future__ import annotations
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "privacy_worker" / "lora_namespace.py"
source = path.read_text(encoding="utf-8")
tree = ast.parse(source)


def fn(name: str):
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"função ausente: {name}")


def imports(node):
    found = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Import):
            found.update(alias.name for alias in child.names)
        elif isinstance(child, ast.ImportFrom):
            found.update(f"{child.module}.{alias.name}" for alias in child.names)
    return found


convert = fn("convert_diffsynth_peft_lora")
header = fn("_header")
convert_imports = imports(convert)
header_imports = imports(header)
top_imports = set()
for node in tree.body:
    if isinstance(node, ast.Import):
        top_imports.update(alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom):
        top_imports.update(f"{node.module}.{alias.name}" for alias in node.names)

safe_open_calls = sum(
    1
    for node in ast.walk(convert)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "safe_open"
)
checks = {
    "module_compiles": True,
    "no_eager_torch": not any(x == "torch" or x.startswith("torch.") for x in top_imports),
    "no_eager_safetensors": not any(x == "safetensors" or x.startswith("safetensors.") for x in top_imports),
    "header_binds_safe_open_locally": "safetensors.safe_open" in header_imports,
    "conversion_binds_torch_locally": "torch" in convert_imports,
    "conversion_binds_safe_open_locally": "safetensors.safe_open" in convert_imports,
    "conversion_binds_save_file_locally": "safetensors.torch.save_file" in convert_imports,
    "conversion_uses_safe_open": safe_open_calls >= 1,
    "hf3_marker_present": "D3.6H10-HF3 — bind safe_open inside conversion runtime" in source,
}
failed = [name for name, ok in checks.items() if not ok]
print(json.dumps({
    "status": "D3_6H10_HF3_SAFE_OPEN_CONVERSION_BINDING_READY" if not failed else "D3_6H10_HF3_SAFE_OPEN_CONVERSION_BINDING_BLOCKED",
    "checksPassed": len(checks) - len(failed),
    "checksTotal": len(checks),
    "checks": checks,
    "blockers": failed,
    "safety": {
        "staticValidationOnly": True,
        "networkCalled": False,
        "runPodCalled": False,
        "gpuStarted": False,
        "r2MutationExecuted": False,
        "databaseMutationExecuted": False,
    },
}, indent=2, ensure_ascii=False))
raise SystemExit(1 if failed else 0)
