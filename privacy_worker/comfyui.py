from __future__ import annotations

import json
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

from .config import Settings
from .errors import ComfyUIError, OutputError
from .model_storage import write_extra_model_paths_config
from .telemetry import log_event, now_ms


class ComfyUIProcessManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def _is_ready(self) -> bool:
        try:
            response = requests.get(
                f"{self.settings.comfyui_base_url}/system_stats",
                timeout=3,
            )
            return response.ok
        except requests.RequestException:
            return False

    def _pump_logs(self, process: subprocess.Popen) -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            line = line.rstrip()
            if line:
                log_event("comfyui_log", message=line)

    def ensure_started(self, request_id: str) -> None:
        if self._is_ready():
            return
        if not self.settings.comfyui_start_local:
            raise ComfyUIError("ComfyUI não está acessível e COMFYUI_START_LOCAL=false.")

        with self._lock:
            if self._is_ready():
                return
            if self._process and self._process.poll() is None:
                pass
            else:
                main_py = self.settings.comfyui_root / "main.py"
                if not main_py.exists():
                    raise ComfyUIError(f"ComfyUI não encontrado em {self.settings.comfyui_root}.")
                command = [
                    "python",
                    str(main_py),
                    "--listen",
                    self.settings.comfyui_host,
                    "--port",
                    str(self.settings.comfyui_port),
                    "--disable-auto-launch",
                    "--input-directory",
                    str(self.settings.input_dir),
                    "--output-directory",
                    str(self.settings.output_dir),
                    "--temp-directory",
                    str(self.settings.temp_dir),
                ]
                extra_paths = write_extra_model_paths_config(self.settings)
                command.extend(["--extra-model-paths-config", str(extra_paths)])
                log_event("comfyui_starting", request_id=request_id, command=command)
                self._process = subprocess.Popen(
                    command,
                    cwd=str(self.settings.comfyui_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                threading.Thread(target=self._pump_logs, args=(self._process,), daemon=True).start()

        deadline = time.monotonic() + self.settings.comfyui_start_timeout_seconds
        while time.monotonic() < deadline:
            if self._process and self._process.poll() is not None:
                raise ComfyUIError(
                    "ComfyUI encerrou durante a inicialização.",
                    details={"returncode": self._process.returncode},
                )
            if self._is_ready():
                log_event("comfyui_ready", request_id=request_id)
                return
            time.sleep(2)
        raise ComfyUIError("Timeout aguardando a inicialização do ComfyUI.")

    def interrupt(self, request_id: str) -> None:
        try:
            requests.post(f"{self.settings.comfyui_base_url}/interrupt", timeout=5)
            log_event("comfyui_interrupt_requested", request_id=request_id, level="WARN")
        except requests.RequestException:
            pass

    def shutdown(self) -> None:
        process = self._process
        if not process or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        log_event("comfyui_process_stopped")


class ComfyUIClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def queue_prompt(self, prompt: dict[str, Any], request_id: str) -> str:
        payload = {"prompt": prompt, "client_id": f"privacy-{request_id}-{uuid.uuid4().hex[:8]}"}
        try:
            response = requests.post(
                f"{self.settings.comfyui_base_url}/prompt",
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            body = response.json()
        except (requests.RequestException, json.JSONDecodeError) as error:
            raise ComfyUIError("Falha ao enviar workflow para o ComfyUI.") from error
        prompt_id = body.get("prompt_id")
        if not prompt_id:
            raise ComfyUIError(
                "ComfyUI rejeitou o workflow.",
                details={"error": body.get("error"), "node_errors": body.get("node_errors")},
            )
        log_event("comfyui_prompt_queued", request_id=request_id, prompt_id=prompt_id)
        return str(prompt_id)

    def wait_for_history(self, prompt_id: str, request_id: str) -> dict[str, Any]:
        started = now_ms()
        deadline = time.monotonic() + self.settings.comfyui_job_timeout_seconds
        while time.monotonic() < deadline:
            try:
                response = requests.get(
                    f"{self.settings.comfyui_base_url}/history/{prompt_id}",
                    timeout=20,
                )
                response.raise_for_status()
                payload = response.json()
            except (requests.RequestException, json.JSONDecodeError) as error:
                log_event(
                    "comfyui_history_retry",
                    request_id=request_id,
                    level="WARN",
                    prompt_id=prompt_id,
                    error=str(error),
                )
                time.sleep(self.settings.comfyui_poll_interval_seconds)
                continue

            record = payload.get(prompt_id)
            if record:
                status = record.get("status") or {}
                status_str = str(status.get("status_str") or "").lower()
                completed = status.get("completed") is True
                messages = status.get("messages") or []
                if status_str == "error":
                    raise ComfyUIError(
                        "ComfyUI falhou ao executar o workflow.",
                        details={"messages": messages[-5:]},
                    )
                if completed or record.get("outputs"):
                    log_event(
                        "comfyui_prompt_completed",
                        request_id=request_id,
                        prompt_id=prompt_id,
                        elapsed_ms=now_ms() - started,
                    )
                    return record
            time.sleep(self.settings.comfyui_poll_interval_seconds)
        raise ComfyUIError("Timeout aguardando conclusão do workflow ComfyUI.")

    @staticmethod
    def _candidate_entries(record: dict[str, Any], output_nodes: tuple[str, ...], strict_output_nodes: bool = False):
        outputs = record.get("outputs") or {}
        ordered_nodes = list(output_nodes) if strict_output_nodes else list(output_nodes) + [key for key in outputs if key not in output_nodes]
        for node_id in ordered_nodes:
            node_output = outputs.get(node_id) or {}
            for field in ("gifs", "videos", "images"):
                entries = node_output.get(field) or []
                for entry in reversed(entries if isinstance(entries, list) else []):
                    if isinstance(entry, dict) and entry.get("filename"):
                        yield entry

    def _cleanup_ephemeral_output(self, candidate: dict[str, Any], request_id: str) -> None:
        if self.settings.model_source_mode != "cached_model":
            return
        output_type = str(candidate.get("type") or "output").lower()
        roots = {
            "output": self.settings.output_dir,
            "temp": self.settings.temp_dir,
        }
        root = roots.get(output_type)
        if root is None:
            raise OutputError("Tipo de saída ComfyUI inseguro para cleanup efêmero.")
        root = root.resolve(strict=True)
        candidate_path = root / str(candidate.get("subfolder") or "") / str(candidate["filename"])
        if candidate_path.is_symlink():
            raise OutputError("O cleanup efêmero recusou uma saída ComfyUI por symlink.")
        resolved = candidate_path.resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise OutputError("A saída ComfyUI resolve para fora da raiz efêmera controlada.") from exc
        if not candidate_path.exists():
            return
        if not candidate_path.is_file():
            raise OutputError("A saída ComfyUI local não é um arquivo regular.")
        try:
            candidate_path.unlink()
        except OSError as exc:
            raise OutputError("Falha ao remover a saída local efêmera do ComfyUI.") from exc
        log_event(
            "comfyui_ephemeral_output_removed",
            request_id=request_id,
            filename=candidate.get("filename"),
        )

    def download_output(
        self,
        *,
        record: dict[str, Any],
        output_nodes: tuple[str, ...],
        destination: Path,
        request_id: str,
        strict_output_nodes: bool = False,
    ) -> Path:
        candidate = next(self._candidate_entries(record, output_nodes, strict_output_nodes), None)
        if not candidate:
            raise OutputError("O workflow terminou sem arquivo de vídeo reconhecível.")
        query = urlencode(
            {
                "filename": candidate["filename"],
                "subfolder": candidate.get("subfolder") or "",
                "type": candidate.get("type") or "output",
            }
        )
        max_bytes = self.settings.max_output_mb * 1024 * 1024
        downloaded = 0
        try:
            with requests.get(
                f"{self.settings.comfyui_base_url}/view?{query}",
                stream=True,
                timeout=(15, 600),
            ) as response:
                response.raise_for_status()
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        downloaded += len(chunk)
                        if downloaded > max_bytes:
                            raise OutputError(
                                f"Saída do ComfyUI excede MAX_OUTPUT_MB={self.settings.max_output_mb}."
                            )
                        handle.write(chunk)
        except OutputError:
            raise
        except requests.RequestException as error:
            raise OutputError("Falha ao recuperar o vídeo produzido pelo ComfyUI.") from error
        if not destination.exists() or destination.stat().st_size <= 0:
            raise OutputError("O ComfyUI retornou um arquivo de saída vazio.")
        self._cleanup_ephemeral_output(candidate, request_id)
        log_event(
            "comfyui_output_downloaded",
            request_id=request_id,
            filename=candidate.get("filename"),
            size_bytes=destination.stat().st_size,
        )
        return destination
