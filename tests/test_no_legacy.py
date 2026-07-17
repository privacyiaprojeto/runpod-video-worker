from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FILES = [
    ROOT / "handler.py",
    ROOT / "Dockerfile",
    ROOT / "requirements.txt",
    *sorted((ROOT / "privacy_worker").glob("*.py")),
]
FORBIDDEN = (
    "insightface",
    "onnxruntime",
    "inswapper_128",
    "faceanalysis",
    "faceswap_zero_shot",
    "opencv-python-headless",
)


def test_runtime_has_no_legacy_faceswap_dependencies_or_contracts():
    merged = "\n".join(path.read_text(encoding="utf-8").lower() for path in RUNTIME_FILES)
    for token in FORBIDDEN:
        assert token not in merged, f"legacy token still present: {token}"
