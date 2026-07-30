import shutil
import subprocess
from pathlib import Path

import pytest

from privacy_worker.errors import ABOutputsIdenticalError
from privacy_worker.visual_ab import compare_ab_videos

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def _video(path: Path, color: str):
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"color=c={color}:s=64x64:r=16:d=1.0625",
            "-frames:v", "17", "-pix_fmt", "yuv420p", "-y", str(path),
        ],
        check=True,
    )


def test_identical_decoded_outputs_fail_closed(tmp_path):
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    _video(a, "red")
    shutil.copy2(a, b)
    with pytest.raises(ABOutputsIdenticalError):
        compare_ab_videos(a, b)


def test_perceptibly_different_outputs_pass_guard(tmp_path):
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    _video(a, "red")
    _video(b, "blue")
    report = compare_ab_videos(a, b)
    assert report["visually_identical"] is False
    assert report["ssim_all"] < report["identical_threshold"]
    assert report["identity_quality_approved"] is False
