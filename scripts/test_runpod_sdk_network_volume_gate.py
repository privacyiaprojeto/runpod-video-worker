from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")


def exact_version(package: str) -> tuple[int, int, int]:
    match = re.search(
        rf"(?m)^{re.escape(package)}==([0-9]+)\.([0-9]+)\.([0-9]+)\s*$",
        requirements,
    )

    assert match, f"{package} precisa permanecer com versão exata pinada"

    return tuple(int(part) for part in match.groups())


runpod_version = exact_version("runpod")
boto3_version = exact_version("boto3")
requests_version = exact_version("requests")

assert runpod_version >= (1, 10, 1), (
    "Network Volume endpoints exigem runpod>=1.10.1 "
    "para evitar o bug conhecido de job tracking."
)

assert boto3_version >= (1, 43, 40), (
    "runpod>=1.10.1 exige boto3>=1.43.40."
)

assert requests_version >= (2, 34, 2), (
    "runpod>=1.10.1 exige requests>=2.34.2."
)

print(
    "RUNPOD_SDK_DEPENDENCY_GATE_READY",
    "runpod=" + ".".join(map(str, runpod_version)),
    "boto3=" + ".".join(map(str, boto3_version)),
    "requests=" + ".".join(map(str, requests_version)),
)
