from __future__ import annotations

from pathlib import Path

from .config import Settings
from .contracts import ProductionRequest
from .errors import WorkflowError


def _required_paths(request: ProductionRequest, settings: Settings) -> tuple[Path, ...]:
    model_name = settings.i2v_model_name if request.is_i2v else settings.v2v_model_name
    required = [
        settings.model_root / "diffusion_models" / model_name,
        settings.model_root / "text_encoders" / settings.text_encoder_name,
        settings.model_root / "vae" / settings.vae_name,
    ]
    if request.is_i2v:
        required.append(settings.model_root / "clip_vision" / settings.clip_vision_name)
    return tuple(required)


def validate_required_models(request: ProductionRequest, settings: Settings) -> None:
    if settings.skip_model_validation:
        return
    missing = [str(path) for path in _required_paths(request, settings) if not path.is_file()]
    if missing:
        raise WorkflowError(
            "Modelos Wan necessários não estão disponíveis no storage configurado.",
            details={"missing_models": missing, "engine": request.engine},
        )
