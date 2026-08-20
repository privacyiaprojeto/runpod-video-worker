from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

match = re.search(
    r"(?m)^runpod==([0-9]+)\.([0-9]+)\.([0-9]+)\s*$",
    requirements,
)

assert match, "runpod SDK precisa permanecer com versão exata pinada"

version = tuple(int(part) for part in match.groups())

assert version >= (1, 10, 1), (
    "Network Volume endpoints exigem runpod>=1.10.1 "
    "para evitar o bug conhecido de job tracking."
)

print(
    "RUNPOD_SDK_NETWORK_VOLUME_GATE_READY",
    ".".join(map(str, version)),
)
