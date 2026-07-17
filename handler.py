from __future__ import annotations

import atexit
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

import runpod

from privacy_worker.comfyui import ComfyUIClient, ComfyUIProcessManager
from privacy_worker.config import settings
from privacy_worker.contracts import parse_production_request
from privacy_worker.downloader import download_media
from privacy_worker.errors import WorkerError
from privacy_worker.models import validate_required_models
from privacy_worker.output import publish_output
from privacy_worker.telemetry import log_event, now_ms
from privacy_worker.workflows import prepare_workflow

settings.ensure_runtime_dirs()
_process_manager = ComfyUIProcessManager(settings)
_client = ComfyUIClient(settings)
atexit.register(_process_manager.shutdown)


def _copy_to_comfy_input(source: Path, *, request_id: str, role: str) -> Path:
    filename = f"privacy_{request_id}_{role}_{uuid.uuid4().hex[:8]}{source.suffix.lower()}"
    destination = settings.input_dir / filename
    shutil.copy2(source, destination)
    return destination


def handler(event: dict[str, Any]) -> dict[str, Any]:
    started_at = now_ms()
    request = parse_production_request(event)
    log_event(
        "wan_job_received",
        request_id=request.request_id,
        engine=request.engine,
        task=request.task,
        workflow_id=request.workflow_id,
        workflow_version=request.workflow_version,
        width=request.width,
        height=request.height,
        fps=request.fps,
        frames=request.frames,
    )

    comfy_inputs: list[Path] = []
    try:
        validate_required_models(request, settings)
        with tempfile.TemporaryDirectory(dir=str(settings.temp_dir), prefix=f"wan_{request.request_id}_") as temp:
            work_dir = Path(temp)
            source_download = download_media(
                url=request.source_image_url,
                destination_dir=work_dir,
                stem="identity_reference",
                max_mb=settings.max_image_download_mb,
                fallback_extension=".png",
                settings=settings,
                request_id=request.request_id,
                expected_content_prefix="image/",
            )
            source_input = _copy_to_comfy_input(
                source_download, request_id=request.request_id, role="identity"
            )
            comfy_inputs.append(source_input)

            base_input = None
            if request.base_video_url:
                base_download = download_media(
                    url=request.base_video_url,
                    destination_dir=work_dir,
                    stem="base_video",
                    max_mb=settings.max_video_download_mb,
                    fallback_extension=".mp4",
                    settings=settings,
                    request_id=request.request_id,
                    expected_content_prefix="video/",
                )
                base_input = _copy_to_comfy_input(
                    base_download, request_id=request.request_id, role="base"
                )
                comfy_inputs.append(base_input)

            output_prefix = f"privacy/{request.engine}/{request.request_id}/{uuid.uuid4().hex[:8]}"
            workflow = prepare_workflow(
                request=request,
                source_image_filename=source_input.name,
                base_video_filename=base_input.name if base_input else None,
                output_prefix=output_prefix,
                settings=settings,
            )
            log_event(
                "workflow_prepared",
                request_id=request.request_id,
                workflow_id=workflow.workflow_id,
                workflow_version=workflow.workflow_version,
                output_nodes=workflow.output_nodes,
            )

            _process_manager.ensure_started(request.request_id)
            prompt_id = _client.queue_prompt(workflow.prompt, request.request_id)
            try:
                history = _client.wait_for_history(prompt_id, request.request_id)
            except Exception:
                _process_manager.interrupt(request.request_id)
                raise

            output_path = _client.download_output(
                record=history,
                output_nodes=workflow.output_nodes,
                destination=work_dir / "wan_output.mp4",
                request_id=request.request_id,
            )
            response = publish_output(output_path, settings, request.request_id)
            response.update(
                {
                    "request_id": request.request_id,
                    "contract_version": request.contract_version,
                    "engine": request.engine,
                    "workflow_id": workflow.workflow_id,
                    "workflow_version": workflow.workflow_version,
                    "elapsed_ms": now_ms() - started_at,
                }
            )
            log_event(
                "wan_job_completed",
                request_id=request.request_id,
                engine=request.engine,
                elapsed_ms=response["elapsed_ms"],
                size_bytes=response["size_bytes"],
                output_mode="private_r2" if response.get("r2_key") else "base64",
            )
            return response
    except WorkerError as error:
        log_event(
            "wan_job_failed",
            request_id=request.request_id,
            level="ERROR",
            error_code=error.code,
            retryable=error.retryable,
            message=str(error),
            details=error.details,
            elapsed_ms=now_ms() - started_at,
        )
        raise RuntimeError(f"{error.code}: {error}") from error
    except Exception as error:
        log_event(
            "wan_job_failed",
            request_id=request.request_id,
            level="ERROR",
            error_code="UNEXPECTED_WORKER_ERROR",
            retryable=False,
            message=str(error),
            elapsed_ms=now_ms() - started_at,
        )
        raise
    finally:
        for path in comfy_inputs:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
