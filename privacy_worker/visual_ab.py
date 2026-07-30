from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from .errors import ABOutputsIdenticalError, OutputError

# Identity replacement must produce a perceptible frame delta. This threshold only rejects
# near-identical decoded videos; it does not approve identity quality.
MAX_IDENTICAL_SSIM = 0.9995


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=False, capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OutputError("Falha ao executar o guardião visual A/B.") from exc


def _probe(path: Path) -> dict[str, Any]:
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ]
    )
    if result.returncode != 0:
        raise OutputError("ffprobe não conseguiu validar um vídeo A/B.", details={"stderr": result.stderr[-1000:]})
    try:
        streams = json.loads(result.stdout).get("streams") or []
        stream = streams[0]
    except (json.JSONDecodeError, IndexError, TypeError) as exc:
        raise OutputError("Metadados de vídeo A/B inválidos.") from exc
    return {
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "avg_frame_rate": str(stream.get("avg_frame_rate") or ""),
        "nb_frames": int(stream.get("nb_frames") or 0),
        "duration": float(stream.get("duration") or 0.0),
    }


def compare_ab_videos(a_path: Path, b_path: Path) -> dict[str, Any]:
    a_probe, b_probe = _probe(a_path), _probe(b_path)
    for field in ("width", "height", "avg_frame_rate", "nb_frames"):
        if a_probe[field] != b_probe[field]:
            raise OutputError(
                "As saídas A/B não são estruturalmente comparáveis.",
                details={"field": field, "a": a_probe[field], "b": b_probe[field]},
            )

    result = _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(a_path),
            "-i",
            str(b_path),
            "-lavfi",
            "[0:v][1:v]ssim",
            "-f",
            "null",
            "-",
        ]
    )
    if result.returncode != 0:
        raise OutputError("FFmpeg não conseguiu comparar as saídas A/B.", details={"stderr": result.stderr[-2000:]})
    match = re.search(r"All:(?P<all>[0-9.]+)", result.stderr)
    if not match:
        raise OutputError("FFmpeg não retornou a métrica SSIM A/B esperada.")
    ssim_all = float(match.group("all"))
    report = {
        "schema_version": "privacy-identity-ab-visual-guard-v1",
        "metric": "ffmpeg_ssim_all",
        "ssim_all": ssim_all,
        "identical_threshold": MAX_IDENTICAL_SSIM,
        "a_probe": a_probe,
        "b_probe": b_probe,
        "visually_identical": ssim_all >= MAX_IDENTICAL_SSIM,
        "identity_quality_approved": False,
    }
    if report["visually_identical"]:
        raise ABOutputsIdenticalError(
            "Os vídeos A e B são visualmente idênticos ou quase idênticos.",
            details=report,
        )
    return report
