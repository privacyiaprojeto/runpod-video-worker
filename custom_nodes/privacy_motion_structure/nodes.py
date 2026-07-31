from __future__ import annotations

import torch
import torch.nn.functional as F

NODE_VERSION = "privacy-motion-only-structure-v1"


def _normalize_per_frame(value: torch.Tensor) -> torch.Tensor:
    peak = value.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return (value / peak).clamp(0.0, 1.0)


class PrivacyMotionOnlyStructure:
    """Remove RGB appearance and keep coarse temporal/edge structure for VACE control."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "spatial_downscale": (
                    "INT",
                    {"default": 4, "min": 2, "max": 8, "step": 1},
                ),
                "edge_threshold": (
                    "FLOAT",
                    {"default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01},
                ),
                "motion_threshold": (
                    "FLOAT",
                    {"default": 0.06, "min": 0.0, "max": 0.5, "step": 0.01},
                ),
                "motion_dilation": (
                    "INT",
                    {"default": 11, "min": 3, "max": 31, "step": 2},
                ),
                "structure_fill": (
                    "FLOAT",
                    {"default": 0.22, "min": 0.0, "max": 0.5, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("motion_structure",)
    FUNCTION = "extract"
    CATEGORY = "Privacy IA/Identity QA"
    DESCRIPTION = (
        "Converts a video IMAGE batch to low-detail grayscale motion/edge structure. "
        "RGB color, skin texture, clothing texture and background appearance are not passed through."
    )

    def extract(
        self,
        images: torch.Tensor,
        spatial_downscale: int,
        edge_threshold: float,
        motion_threshold: float,
        motion_dilation: int,
        structure_fill: float,
    ):
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise RuntimeError("MOTION_STRUCTURE_INVALID_INPUT: expected IMAGE batch [B,H,W,C]")
        if images.shape[0] < 2:
            raise RuntimeError("MOTION_STRUCTURE_TOO_SHORT: at least two frames are required")
        if images.shape[-1] < 3:
            raise RuntimeError("MOTION_STRUCTURE_INVALID_CHANNELS: RGB input is required")

        source = images[..., :3].float().clamp(0.0, 1.0).permute(0, 3, 1, 2)
        height, width = int(source.shape[-2]), int(source.shape[-1])
        low_height = max(32, height // int(spatial_downscale))
        low_width = max(32, width // int(spatial_downscale))

        gray = (
            source[:, 0:1] * 0.299
            + source[:, 1:2] * 0.587
            + source[:, 2:3] * 0.114
        )
        gray = F.interpolate(
            gray,
            size=(low_height, low_width),
            mode="bilinear",
            align_corners=False,
        )
        gray = F.avg_pool2d(gray, kernel_size=5, stride=1, padding=2)

        kernel_x = gray.new_tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
        ).unsqueeze(0)
        kernel_y = gray.new_tensor(
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]
        ).unsqueeze(0)
        grad_x = F.conv2d(gray, kernel_x, padding=1)
        grad_y = F.conv2d(gray, kernel_y, padding=1)
        edges = _normalize_per_frame(torch.sqrt(grad_x.square() + grad_y.square() + 1e-8))
        edges = ((edges - float(edge_threshold)) / max(1e-6, 1.0 - float(edge_threshold))).clamp(0.0, 1.0)

        temporal = torch.zeros_like(gray)
        temporal[1:] = (gray[1:] - gray[:-1]).abs()
        temporal[0] = temporal[1]
        temporal = F.avg_pool2d(temporal, kernel_size=7, stride=1, padding=3)
        temporal = _normalize_per_frame(temporal)
        temporal = (
            (temporal - float(motion_threshold))
            / max(1e-6, 1.0 - float(motion_threshold))
        ).clamp(0.0, 1.0)

        dilation = max(3, int(motion_dilation))
        if dilation % 2 == 0:
            dilation += 1
        motion_region = F.max_pool2d(
            temporal,
            kernel_size=dilation,
            stride=1,
            padding=dilation // 2,
        ).clamp(0.0, 1.0)

        structure = (edges * motion_region + float(structure_fill) * motion_region).clamp(0.0, 1.0)
        structure = F.interpolate(
            structure,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        structure = structure.repeat(1, 3, 1, 1).permute(0, 2, 3, 1)
        return (structure.to(dtype=images.dtype, device=images.device),)


NODE_CLASS_MAPPINGS = {
    "PrivacyMotionOnlyStructure": PrivacyMotionOnlyStructure,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PrivacyMotionOnlyStructure": "Privacy IA — Motion-Only Structural Control",
}
