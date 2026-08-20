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
from privacy_worker.identity_ab import (
    CONTRACT_VERSION as IDENTITY_AB_CONTRACT_VERSION,
    download_private_ref,
    materialize_lora,
    parse_identity_ab_request,
    r2_client,
    read_runtime_lora_attestation,
    reserve_one_shot,
    runtime_attestation_path,
    update_lock,
)
from privacy_worker.identity_motion_abc import (
    CONTRACT_VERSION as IDENTITY_MOTION_ABC_CONTRACT_VERSION,
    derive_softedge_control,
    download_private_ref as download_motion_private_ref,
    materialize_lora as materialize_motion_lora,
    parse_identity_motion_abc_request,
    r2_client as motion_r2_client,
    read_runtime_lora_attestation as read_motion_runtime_lora_attestation,
    reserve_one_shot as reserve_motion_one_shot,
    runtime_attestation_path as motion_runtime_attestation_path,
    update_lock as update_motion_lock,
)
from privacy_worker.downloader import download_media
from privacy_worker.errors import ComfyUIError, DownloadError, LoraCompatibilityError, LoraNotAppliedError, WorkerError
from privacy_worker.models import validate_required_models
from privacy_worker.output import publish_output, publish_private_named_output
from privacy_worker.telemetry import log_event, now_ms
from privacy_worker.visual_ab import compare_ab_videos
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


def _identity_comfy_failure(error: ComfyUIError) -> WorkerError | None:
    serialized = str(error) + " " + str(getattr(error, "details", {}))
    if "LORA_KEY_FORMAT_MISMATCH" in serialized:
        return LoraCompatibilityError(
            "O ComfyUI recusou o namespace ou os alvos da LoRA.",
            details={"comfyui": getattr(error, "details", {})},
        )
    if "LORA_NOT_APPLIED" in serialized:
        return LoraNotAppliedError(
            "O ComfyUI não anexou patches LoRA ao modelo Wan.",
            details={"comfyui": getattr(error, "details", {})},
        )
    return None


def _handle_identity_ab(event: dict[str, Any]) -> dict[str, Any]:
    started_at = now_ms()
    request = parse_identity_ab_request(event)
    validate_required_models(request, settings)
    lock = reserve_one_shot(request, settings)
    comfy_inputs: list[Path] = []
    attestation_path = runtime_attestation_path(request, settings)
    attestation_path.unlink(missing_ok=True)
    try:
        with tempfile.TemporaryDirectory(
            dir=str(settings.temp_dir), prefix=f"identity_ab_{request.request_id}_"
        ) as temp:
            work_dir = Path(temp)
            client = r2_client(settings)
            neutral = download_private_ref(
                client,
                request.base_video_ref,
                work_dir / "neutral-motion-01.mp4",
                settings.max_video_download_mb,
            )
            base_input = _copy_to_comfy_input(
                neutral, request_id=request.request_id, role="neutral"
            )
            comfy_inputs.append(base_input)

            reference_suffix = Path(request.reference_image_ref["key"]).suffix.lower()
            reference = download_private_ref(
                client,
                request.reference_image_ref,
                work_dir / f"kyc-face-front{reference_suffix}",
                settings.max_image_download_mb,
            )
            reference_input = _copy_to_comfy_input(
                reference, request_id=request.request_id, role="kyc_face_front"
            )
            comfy_inputs.append(reference_input)

            _, lora_name, conversion_attestation = materialize_lora(
                client, request, settings, work_dir
            )
            workflow = prepare_workflow(
                request=request,
                source_image_filename=reference_input.name,
                base_video_filename=base_input.name,
                output_prefix=f"privacy/identity-ab/{request.request_id}",
                settings=settings,
                lora_filename=lora_name,
                lora_attestation_name=request.request_id,
            )
            update_lock(
                lock,
                "running",
                workflow_id=workflow.workflow_id,
                lora_conversion_version=conversion_attestation.get("conversion_version"),
                lora_pair_count=conversion_attestation.get("pair_count"),
            )
            _process_manager.ensure_started(request.request_id)
            prompt_id = _client.queue_prompt(workflow.prompt, request.request_id)
            try:
                history = _client.wait_for_history(prompt_id, request.request_id)
            except ComfyUIError as error:
                _process_manager.interrupt(request.request_id)
                classified = _identity_comfy_failure(error)
                if classified is not None:
                    raise classified from error
                raise
            runtime_attestation = read_runtime_lora_attestation(
                request,
                settings,
                expected_pair_count=int(conversion_attestation["pair_count"]),
            )
            a_path = _client.download_output(
                record=history,
                output_nodes=("12",),
                destination=work_dir / "baseline_without_lora.mp4",
                request_id=request.request_id,
                strict_output_nodes=True,
            )
            b_path = _client.download_output(
                record=history,
                output_nodes=("17",),
                destination=work_dir / "candidate_with_lora.mp4",
                request_id=request.request_id,
                strict_output_nodes=True,
            )
            visual_guard = compare_ab_videos(a_path, b_path)
            assets = []
            for asset_key, label, path in (
                ("baseline_without_lora", "A — vídeo neutro sem LoRA", a_path),
                ("candidate_with_lora", "B — vídeo RGB + KYC explícita + trigger token + LoRA DiT 0.65", b_path),
            ):
                uploaded = publish_private_named_output(
                    path, settings, request.request_id, asset_key, request.contract_version
                )
                assets.append(
                    {
                        **uploaded,
                        "asset_key": asset_key,
                        "label": label,
                        "kind": "video",
                        "width": 832,
                        "height": 480,
                        "num_frames": 17,
                        "fps": 16,
                        "duration_seconds": 17 / 16,
                        "private_only": True,
                    }
                )
            update_lock(
                lock,
                "completed",
                asset_count=2,
                patched_model_key_count=runtime_attestation["patched_model_key_count"],
                ab_ssim_all=visual_guard["ssim_all"],
            )
            return {
                "contract_version": request.contract_version,
                "status": "identity_neutral_ab_completed",
                "qa_kit": {
                    "schema_version": "privacy-identity-neutral-ab-kit-v2",
                    "actor_profile_id": request.actor_profile_id,
                    "training_run_id": request.training_run_id,
                    "adapter_id": request.adapter_id,
                    "asset_count": 2,
                    "assets": assets,
                    "reviewable": True,
                    "same_seed": 99,
                    "same_neutral_source_sha256": request.base_video_ref["sha256"],
                    "branch_a_reference": "neutral_first_frame",
                    "branch_b_reference_system_tag": request.reference_image_ref["system_tag"],
                    "branch_b_reference_asset_id": request.reference_image_ref["asset_id"],
                    "branch_b_reference_sha256": request.reference_image_ref["sha256"],
                    "branch_b_trigger_token": request.trigger_token,
                    "branch_b_control_mode": "raw_rgb_v2v_denoise_085",
                    "branch_b_control_source_sha256": request.base_video_ref["sha256"],
                    "branch_b_raw_rgb_control": True,
                    "branch_a_k_sampler_denoise": request.branch_a_denoise,
                    "branch_b_k_sampler_denoise": request.branch_b_denoise,
                    "ab_denoise_paired": request.branch_a_denoise == request.branch_b_denoise,
                    "workflow_revision": "D3.6H12-trigger-token-raw-rgb-v2v-denoise-085-v1",
                    "lora_strength": 0.65,
                    "provenance": {
                        "trigger_token": request.trigger_token,
                        "reference_asset_id": request.reference_image_ref["asset_id"],
                        "reference_sha256": request.reference_image_ref["sha256"],
                        "control_mode": "raw_rgb_v2v_denoise_085",
                        "control_source_sha256": request.base_video_ref["sha256"],
                        "branch_a_denoise": request.branch_a_denoise,
                        "branch_b_denoise": request.branch_b_denoise,
                        "denoise_paired": request.branch_a_denoise == request.branch_b_denoise,
                        "methodology_hotfix": "D3.6H12-HF2-paired-denoise-085-v1",
                        "lora_strength": 0.65,
                        "workflow_revision": "D3.6H12-trigger-token-raw-rgb-v2v-denoise-085-v1",
                    },
                    "private_only": True,
                    "approval_allowed": False,
                    "lora_attestation": {
                        "source_format": conversion_attestation["source_format"],
                        "target_format": conversion_attestation["target_format"],
                        "conversion_version": conversion_attestation["conversion_version"],
                        "source_sha256": conversion_attestation["source_sha256"],
                        "translated_sha256": conversion_attestation["translated_sha256"],
                        "tensor_pairs": conversion_attestation["pair_count"],
                        "model_keys_matched": conversion_attestation["model_keys_matched"],
                        "loaded_patch_count": runtime_attestation["loaded_patch_count"],
                        "patched_model_key_count": runtime_attestation["patched_model_key_count"],
                        "lora_applied": runtime_attestation["lora_applied"],
                    },
                    "visual_guard": visual_guard,
                },
                "elapsed_ms": now_ms() - started_at,
            }
    except WorkerError as error:
        update_lock(
            lock,
            "failed",
            error_code=error.code,
            error_details=error.details,
            automatic_retry=False,
        )
        raise RuntimeError(f"{error.code}: {error}") from error
    except Exception as error:
        update_lock(
            lock,
            "failed",
            error_code=type(error).__name__,
            automatic_retry=False,
        )
        raise
    finally:
        for path in comfy_inputs:
            path.unlink(missing_ok=True)
        attestation_path.unlink(missing_ok=True)


def _handle_identity_motion_abc(event: dict[str, Any]) -> dict[str, Any]:
    started_at = now_ms()
    request = parse_identity_motion_abc_request(event)
    validate_required_models(request, settings)
    lock = reserve_motion_one_shot(request, settings)
    comfy_inputs: list[Path] = []
    attestation_path = motion_runtime_attestation_path(request, settings)
    attestation_path.unlink(missing_ok=True)
    try:
        with tempfile.TemporaryDirectory(
            dir=str(settings.temp_dir), prefix=f"identity_motion_abc_{request.request_id}_"
        ) as temp:
            work_dir = Path(temp)
            client = motion_r2_client(settings)

            neutral = download_motion_private_ref(
                client,
                request.base_video_ref,
                work_dir / "neutral-motion-01.mp4",
                settings.max_video_download_mb,
            )
            softedge = work_dir / "neutral-motion-01-softedge.mp4"
            control_provenance = derive_softedge_control(neutral, softedge)
            if control_provenance["source_sha256"] != request.base_video_ref["sha256"]:
                raise DownloadError(
                    "O vídeo neutro usado para derivar o soft-edge não corresponde ao contrato."
                )

            control_input = _copy_to_comfy_input(
                softedge, request_id=request.request_id, role="softedge_control"
            )
            comfy_inputs.append(control_input)

            reference_suffix = Path(request.reference_image_ref["key"]).suffix.lower()
            reference = download_motion_private_ref(
                client,
                request.reference_image_ref,
                work_dir / f"kyc-face-front{reference_suffix}",
                settings.max_image_download_mb,
            )
            reference_input = _copy_to_comfy_input(
                reference, request_id=request.request_id, role="kyc_face_front"
            )
            comfy_inputs.append(reference_input)

            _, lora_name, conversion_attestation = materialize_motion_lora(
                client, request, settings, work_dir
            )

            workflow = prepare_workflow(
                request=request,
                source_image_filename=reference_input.name,
                base_video_filename=control_input.name,
                output_prefix=f"privacy/identity-motion-abc/{request.request_id}",
                settings=settings,
                lora_filename=lora_name,
                lora_attestation_name=request.request_id,
            )

            update_motion_lock(
                lock,
                "running",
                workflow_id=workflow.workflow_id,
                lora_conversion_version=conversion_attestation.get("conversion_version"),
                lora_pair_count=conversion_attestation.get("pair_count"),
                control_representation=control_provenance["representation"],
                derived_control_sha256=control_provenance["derived_sha256"],
            )

            _process_manager.ensure_started(request.request_id)
            prompt_id = _client.queue_prompt(workflow.prompt, request.request_id)
            try:
                history = _client.wait_for_history(prompt_id, request.request_id)
            except ComfyUIError as error:
                _process_manager.interrupt(request.request_id)
                classified = _identity_comfy_failure(error)
                if classified is not None:
                    raise classified from error
                raise

            runtime_attestation = read_motion_runtime_lora_attestation(
                request,
                settings,
                expected_pair_count=int(conversion_attestation["pair_count"]),
            )

            a_path = _client.download_output(
                record=history,
                output_nodes=("12",),
                destination=work_dir / "baseline_without_identity.mp4",
                request_id=request.request_id,
                strict_output_nodes=True,
            )
            b_path = _client.download_output(
                record=history,
                output_nodes=("17",),
                destination=work_dir / "identity_reference_without_lora.mp4",
                request_id=request.request_id,
                strict_output_nodes=True,
            )
            c_path = _client.download_output(
                record=history,
                output_nodes=("26",),
                destination=work_dir / "candidate_with_lora.mp4",
                request_id=request.request_id,
                strict_output_nodes=True,
            )

            assets = []
            for asset_key, label, path in (
                ("baseline_without_identity", "A — soft-edge, sem KYC/trigger/LoRA", a_path),
                ("identity_reference_without_lora", "B — mesma estrutura + KYC/trigger, sem LoRA", b_path),
                ("candidate_with_lora", "C — igual ao B + LoRA DiT 0.65", c_path),
            ):
                uploaded = publish_private_named_output(
                    path, settings, request.request_id, asset_key, request.contract_version
                )
                assets.append(
                    {
                        **uploaded,
                        "asset_key": asset_key,
                        "label": label,
                        "kind": "video",
                        "width": request.width,
                        "height": request.height,
                        "num_frames": request.frames,
                        "fps": request.fps,
                        "duration_seconds": request.frames / request.fps,
                        "private_only": True,
                    }
                )

            provenance = {
                "validation_profile": "video_softedge_abc_v1",
                "control_representation": control_provenance["representation"],
                "source_motion_sha256": request.base_video_ref["sha256"],
                "derived_control_sha256": control_provenance["derived_sha256"],
                "raw_rgb_control_used": False,
                "appearance_reduced_structural_control_used": True,
                "same_control_across_branches": True,
                "same_seed_across_branches": True,
                "same_sampler_across_branches": True,
                "seed": request.seed,
                "denoise": request.branch_a_denoise,
                "branch_a": {
                    "kyc": False,
                    "trigger": False,
                    "lora": False,
                },
                "branch_b": {
                    "kyc": True,
                    "trigger": True,
                    "lora": False,
                    "reference_asset_id": request.reference_image_ref["asset_id"],
                    "reference_sha256": request.reference_image_ref["sha256"],
                    "trigger_token": request.trigger_token,
                },
                "branch_c": {
                    "kyc": True,
                    "trigger": True,
                    "lora": True,
                    "reference_asset_id": request.reference_image_ref["asset_id"],
                    "reference_sha256": request.reference_image_ref["sha256"],
                    "trigger_token": request.trigger_token,
                    "adapter_sha256": request.adapter_ref["sha256"],
                    "lora_strength": 0.65,
                },
                "workflow_revision": "M4-identity-motion-abc-softedge-v1",
                "methodology_hotfix": "M4-HF-softedge-abc-lora-isolation-v1",
            }

            update_motion_lock(
                lock,
                "completed",
                asset_count=3,
                patched_model_key_count=runtime_attestation["patched_model_key_count"],
                derived_control_sha256=control_provenance["derived_sha256"],
            )

            return {
                "contract_version": request.contract_version,
                "status": "identity_motion_abc_completed",
                "qa_kit": {
                    "schema_version": "privacy-identity-motion-abc-kit-v1",
                    "qa_kit_id": request.request_id,
                    "actor_profile_id": request.actor_profile_id,
                    "training_run_id": request.training_run_id,
                    "adapter_id": request.adapter_id,
                    "asset_count": 3,
                    "assets": assets,
                    "reviewable": True,
                    "provenance": provenance,
                    "private_only": True,
                    "approval_allowed": False,
                    "lora_attestation": {
                        "source_sha256": request.adapter_ref["sha256"],
                        "conversion_version": conversion_attestation.get("conversion_version"),
                        "pair_count": conversion_attestation.get("pair_count"),
                        "patched_model_key_count": runtime_attestation["patched_model_key_count"],
                        "all_expected_loaded": runtime_attestation["all_expected_loaded"],
                        "all_expected_patched": runtime_attestation["all_expected_patched"],
                    },
                },
                "elapsed_ms": now_ms() - started_at,
                "safety": {
                    "private_storage_only": True,
                    "public_url_created": False,
                    "automatic_retry": False,
                    "adapter_approved": False,
                    "product_released": False,
                },
            }
    except WorkerError as error:
        update_motion_lock(
            lock,
            "failed",
            error_code=error.code,
            error_details=error.details,
            automatic_retry=False,
        )
        raise RuntimeError(f"{error.code}: {error}") from error
    except Exception as error:
        update_motion_lock(
            lock,
            "failed",
            error_code=type(error).__name__,
            automatic_retry=False,
        )
        raise
    finally:
        for path in comfy_inputs:
            path.unlink(missing_ok=True)
        attestation_path.unlink(missing_ok=True)


def handler(event: dict[str, Any]) -> dict[str, Any]:
    contract_version = _input_payload(event).get("contract_version")
    if contract_version == IDENTITY_MOTION_ABC_CONTRACT_VERSION:
        return _handle_identity_motion_abc(event)
    if contract_version == IDENTITY_AB_CONTRACT_VERSION:
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
