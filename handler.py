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
from privacy_worker.identity_ab import CONTRACT_VERSION as IDENTITY_AB_CONTRACT_VERSION, download_private_ref, materialize_lora, parse_identity_ab_request, r2_client, reserve_one_shot, update_lock
from privacy_worker.downloader import download_media
from privacy_worker.errors import WorkerError
from privacy_worker.models import validate_required_models
from privacy_worker.output import publish_output, publish_private_named_output
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



def _input_payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("input", event)
    return payload if isinstance(payload, dict) else {}


def _handle_identity_ab(event: dict[str, Any]) -> dict[str, Any]:
    started_at = now_ms()
    request = parse_identity_ab_request(event)
    validate_required_models(request, settings)
    lock = reserve_one_shot(request, settings)
    comfy_inputs: list[Path] = []
    try:
        with tempfile.TemporaryDirectory(dir=str(settings.temp_dir), prefix=f"identity_ab_{request.request_id}_") as temp:
            work_dir = Path(temp)
            client = r2_client(settings)
            neutral = download_private_ref(client, request.base_video_ref, work_dir / "neutral-motion-01.mp4", settings.max_video_download_mb)
            base_input = _copy_to_comfy_input(neutral, request_id=request.request_id, role="neutral")
            comfy_inputs.append(base_input)
            _, lora_name = materialize_lora(client, request, settings, work_dir)
            workflow = prepare_workflow(request=request, source_image_filename=None, base_video_filename=base_input.name, output_prefix=f"privacy/identity-ab/{request.request_id}", settings=settings, lora_filename=lora_name)
            update_lock(lock, "running", workflow_id=workflow.workflow_id)
            _process_manager.ensure_started(request.request_id)
            prompt_id = _client.queue_prompt(workflow.prompt, request.request_id)
            try:
                history = _client.wait_for_history(prompt_id, request.request_id)
            except Exception:
                _process_manager.interrupt(request.request_id); raise
            a_path = _client.download_output(record=history, output_nodes=("12",), destination=work_dir / "baseline_without_lora.mp4", request_id=request.request_id, strict_output_nodes=True)
            b_path = _client.download_output(record=history, output_nodes=("17",), destination=work_dir / "candidate_with_lora.mp4", request_id=request.request_id, strict_output_nodes=True)
            assets = []
            for asset_key, label, path in (("baseline_without_lora", "A — vídeo neutro sem LoRA", a_path), ("candidate_with_lora", "B — mesmo vídeo com LoRA DiT 0.65", b_path)):
                uploaded = publish_private_named_output(path, settings, request.request_id, asset_key, request.contract_version)
                assets.append({**uploaded, "asset_key":asset_key, "label":label, "kind":"video", "width":832, "height":480, "num_frames":17, "fps":16, "duration_seconds":17/16, "private_only":True})
            update_lock(lock, "completed", asset_count=2)
            return {"contract_version":request.contract_version,"status":"identity_neutral_ab_completed","qa_kit":{"schema_version":"privacy-identity-neutral-ab-kit-v1","actor_profile_id":request.actor_profile_id,"training_run_id":request.training_run_id,"adapter_id":request.adapter_id,"asset_count":2,"assets":assets,"reviewable":True,"same_seed":99,"same_neutral_source_sha256":request.base_video_ref["sha256"],"lora_strength":0.65,"private_only":True,"approval_allowed":False},"elapsed_ms":now_ms()-started_at}
    except Exception as error:
        update_lock(lock, "failed", error_code=type(error).__name__, automatic_retry=False)
        raise
    finally:
        for path in comfy_inputs: path.unlink(missing_ok=True)

def handler(event: dict[str, Any]) -> dict[str, Any]:
    if _input_payload(event).get("contract_version") == IDENTITY_AB_CONTRACT_VERSION:
        return _handle_identity_ab(event)
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
