from __future__ import annotations

import hmac
import json
import os
import queue
import shutil
import signal
import threading
import traceback
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from privacy_worker.config import settings
from privacy_worker.model_storage import prepare_model_storage


POD_CONTRACT_VERSION = "privacy-pod-runtime-v1"
POD_SUPPORTED_MODEL_SOURCE_MODES = {"network_volume", "cached_model", "r2_registry"}

_storage_bootstrap_lock = threading.Lock()
_storage_bootstrap_status = "not_started"
_storage_bootstrap_error: str | None = None


def storage_bootstrap_snapshot() -> tuple[str, str | None]:
    with _storage_bootstrap_lock:
        return _storage_bootstrap_status, _storage_bootstrap_error


def _storage_bootstrap_worker() -> None:
    global _storage_bootstrap_status, _storage_bootstrap_error

    try:
        prepare_model_storage(settings)
    except Exception as exc:
        traceback.print_exc()
        with _storage_bootstrap_lock:
            _storage_bootstrap_status = "failed"
            _storage_bootstrap_error = f"{type(exc).__name__}: {str(exc)[:500]}"
    else:
        with _storage_bootstrap_lock:
            _storage_bootstrap_status = "ready"
            _storage_bootstrap_error = None


def start_storage_bootstrap() -> None:
    global _storage_bootstrap_status

    with _storage_bootstrap_lock:
        if _storage_bootstrap_status in {"starting", "ready"}:
            return
        _storage_bootstrap_status = "starting"

    threading.Thread(
        target=_storage_bootstrap_worker,
        name="privacy-pod-model-storage-bootstrap",
        daemon=True,
    ).start()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def env_int(
    name: str,
    default: int,
    maximum: int | None = None,
) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default

    value = max(value, 1)

    if maximum is not None:
        value = min(value, maximum)

    return value


@dataclass(frozen=True)
class PodServerConfig:
    host: str
    port: int
    api_token: str
    max_body_bytes: int
    require_cuda: bool
    require_tmp_runtime: bool

    @classmethod
    def from_env(cls) -> "PodServerConfig":
        return cls(
            host=(
                os.getenv("POD_HTTP_HOST", "0.0.0.0").strip()
                or "0.0.0.0"
            ),
            port=env_int(
                "POD_HTTP_PORT",
                8000,
                65535,
            ),
            api_token=os.getenv(
                "POD_API_TOKEN",
                "",
            ).strip(),
            max_body_bytes=(
                env_int(
                    "POD_MAX_BODY_MB",
                    8,
                    64,
                )
                * 1024
                * 1024
            ),
            require_cuda=env_bool(
                "POD_REQUIRE_CUDA",
                True,
            ),
            require_tmp_runtime=env_bool(
                "POD_REQUIRE_TMP_RUNTIME",
                True,
            ),
        )

    def validate_startup(self) -> None:
        if len(self.api_token) < 32:
            raise RuntimeError(
                "POD_API_TOKEN ausente ou curto; "
                "servidor POD bloqueado."
            )


@dataclass
class JobRecord:
    job_id: str
    request_id: str | None
    status: str = "queued"
    created_at: str = field(default_factory=utc_now)
    started_at: str | None = None
    completed_at: str | None = None
    result: dict[str, Any] | None = None
    error_type: str | None = None
    error_message: str | None = None

    def public(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "job_id": self.job_id,
            "request_id": self.request_id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

        if self.status == "completed":
            payload["result"] = self.result

        elif self.status == "failed":
            payload["error"] = {
                "type": self.error_type,
                "message": self.error_message,
            }

        return payload


class PodBusyError(RuntimeError):
    pass


class PodJobRunner:
    def __init__(
        self,
        handler_loader: Callable[
            [],
            Callable[
                [dict[str, Any]],
                dict[str, Any],
            ],
        ]
        | None = None,
    ):
        self._handler_loader = (
            handler_loader
            or self._default_handler_loader
        )

        self._jobs: dict[str, JobRecord] = {}
        self._events: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

        self._queue: queue.Queue[str] = queue.Queue(
            maxsize=1
        )

        self._thread = threading.Thread(
            target=self._worker_loop,
            name="privacy-pod-job-runner",
            daemon=True,
        )

        self._thread.start()

    @staticmethod
    def _default_handler_loader():
        # Importar handler como módulo NÃO executa
        # runpod.serverless.start(), pois aquele trecho
        # fica protegido por __name__ == "__main__".
        from handler import handler

        return handler

    @staticmethod
    def request_id(
        event: dict[str, Any],
    ) -> str | None:
        payload = event.get(
            "input",
            event,
        )

        if not isinstance(payload, dict):
            return None

        value = str(
            payload.get("request_id")
            or ""
        ).strip()

        return value[:160] or None

    def submit(
        self,
        event: dict[str, Any],
    ) -> JobRecord:
        if not isinstance(event, dict):
            raise ValueError(
                "Envelope precisa ser objeto JSON."
            )

        with self._lock:
            busy = any(
                record.status in {
                    "queued",
                    "running",
                }
                for record in self._jobs.values()
            )

            if busy:
                raise PodBusyError(
                    "POD aceita somente um job em voo."
                )

            job_id = str(uuid.uuid4())

            record = JobRecord(
                job_id=job_id,
                request_id=self.request_id(event),
            )

            self._jobs[job_id] = record
            self._events[job_id] = event

            self._queue.put_nowait(job_id)

            return record

    def get(
        self,
        job_id: str,
    ) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()

            try:
                with self._lock:
                    record = self._jobs[job_id]
                    event = self._events.pop(job_id)

                    record.status = "running"
                    record.started_at = utc_now()

                try:
                    result = self._handler_loader()(event)

                    if not isinstance(result, dict):
                        raise RuntimeError(
                            "Handler POD retornou "
                            "resultado não JSON."
                        )

                except Exception as exc:
                    traceback.print_exc()

                    with self._lock:
                        record = self._jobs[job_id]

                        record.status = "failed"
                        record.error_type = (
                            type(exc).__name__[:120]
                        )
                        record.error_message = (
                            str(exc)[:800]
                        )
                        record.completed_at = utc_now()

                else:
                    with self._lock:
                        record = self._jobs[job_id]

                        record.status = "completed"
                        record.result = result
                        record.completed_at = utc_now()

            finally:
                self._queue.task_done()


def inside(
    path: Path,
    root: Path,
) -> bool:
    try:
        path.relative_to(root)
        return True

    except ValueError:
        return False


def runtime_is_tmp() -> bool:
    try:
        runtime = settings.runtime_root.resolve(
            strict=False
        )

    except OSError:
        return False

    tmp_root = Path("/tmp")

    return (
        runtime == tmp_root
        or inside(runtime, tmp_root)
    )


def roots_are_separate() -> bool:
    try:
        model_root = settings.model_root.resolve(
            strict=False
        )

        runtime_root = settings.runtime_root.resolve(
            strict=False
        )

    except OSError:
        return False

    return (
        not inside(model_root, runtime_root)
        and not inside(runtime_root, model_root)
    )


def runtime_disk_ready() -> bool:
    try:
        settings.runtime_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        free_bytes = int(
            shutil.disk_usage(
                settings.runtime_root
            ).free
        )

    except OSError:
        return False

    required_bytes = int(
        settings.ephemeral_min_free_gb
        * 1024**3
    )

    return free_bytes >= required_bytes


def required_model_paths() -> tuple[Path, ...]:
    return (
        settings.model_root
        / "diffusion_models"
        / settings.i2v_model_name,

        settings.model_root
        / "diffusion_models"
        / settings.v2v_model_name,

        settings.model_root
        / "text_encoders"
        / settings.text_encoder_name,

        settings.model_root
        / "vae"
        / settings.vae_name,

        settings.model_root
        / "clip_vision"
        / settings.clip_vision_name,
    )


def readiness_report(
    config: PodServerConfig,
) -> dict[str, Any]:
    blockers: list[str] = []

    if settings.model_source_mode not in POD_SUPPORTED_MODEL_SOURCE_MODES:
        blockers.append(
            "MODEL_SOURCE_MODE_NOT_POD_SUPPORTED"
        )

    if (
        settings.identity_one_shot_lock_backend
        != "r2"
    ):
        blockers.append(
            "IDENTITY_ONE_SHOT_LOCK_NOT_R2"
        )

    if not settings.r2_configured:
        blockers.append(
            "PRIVATE_R2_NOT_CONFIGURED"
        )

    if (
        config.require_tmp_runtime
        and not runtime_is_tmp()
    ):
        blockers.append(
            "RUNTIME_ROOT_NOT_EPHEMERAL_TMP"
        )

    if not roots_are_separate():
        blockers.append(
            "MODEL_ROOT_RUNTIME_ROOT_OVERLAP"
        )

    if not (
        settings.comfyui_root
        / "main.py"
    ).is_file():
        blockers.append(
            "COMFYUI_RUNTIME_MISSING"
        )

    bootstrap_status, bootstrap_error = storage_bootstrap_snapshot()

    if bootstrap_status == "not_started":
        start_storage_bootstrap()
        bootstrap_status, bootstrap_error = storage_bootstrap_snapshot()

    if bootstrap_status == "starting":
        blockers.append(
            "MODEL_STORAGE_BOOTSTRAP_IN_PROGRESS"
        )
    elif bootstrap_status == "failed":
        if bootstrap_error:
            print(
                "[pod-readiness] storage blocker: "
                f"{bootstrap_error}",
                flush=True,
            )
        blockers.append(
            "MODEL_STORAGE_NOT_READY"
        )

    if bootstrap_status == "ready":
        if not runtime_disk_ready():
            blockers.append(
                "EPHEMERAL_RUNTIME_DISK_NOT_READY"
            )

        if any(
            not path.is_file()
            for path in required_model_paths()
        ):
            blockers.append(
                "REQUIRED_MODELS_NOT_READY"
            )

    if config.require_cuda:
        try:
            import torch

            if (
                not torch.cuda.is_available()
                or torch.cuda.device_count() < 1
            ):
                blockers.append(
                    "CUDA_NOT_READY"
                )

        except Exception as exc:
            print(
                "[pod-readiness] CUDA blocker: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

            blockers.append(
                "CUDA_NOT_READY"
            )

    blockers = list(
        dict.fromkeys(blockers)
    )

    return {
        "contract_version":
            POD_CONTRACT_VERSION,
        "status":
            "ready"
            if not blockers
            else "blocked",
        "ready":
            not blockers,
        "blockers":
            blockers,
        "model_storage_bootstrap":
            bootstrap_status,
    }


class PrivacyPodHTTPServer(
    ThreadingHTTPServer
):
    daemon_threads = True

    def __init__(
        self,
        server_address,
        handler_class,
        *,
        config: PodServerConfig,
        runner: PodJobRunner,
    ):
        super().__init__(
            server_address,
            handler_class,
        )

        self.config = config
        self.runner = runner


class PrivacyPodRequestHandler(
    BaseHTTPRequestHandler
):
    server_version = "PrivacyIA-POD/1"

    def log_message(
        self,
        fmt: str,
        *args: Any,
    ) -> None:
        print(
            "[pod-http] "
            f"{self.address_string()} "
            f"{fmt % args}",
            flush=True,
        )

    def send_json(
        self,
        status: int,
        payload: dict[str, Any],
    ) -> None:
        body = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

        self.send_response(status)

        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )

        self.send_header(
            "Content-Length",
            str(len(body)),
        )

        self.send_header(
            "Cache-Control",
            "no-store",
        )

        self.send_header(
            "X-Content-Type-Options",
            "nosniff",
        )

        self.end_headers()
        self.wfile.write(body)

    def authorized(self) -> bool:
        expected = (
            "Bearer "
            + self.server.config.api_token
        )

        supplied = self.headers.get(
            "Authorization",
            "",
        )

        return hmac.compare_digest(
            supplied,
            expected,
        )

    def require_auth(self) -> bool:
        if self.authorized():
            return True

        self.send_json(
            HTTPStatus.UNAUTHORIZED,
            {
                "error": "unauthorized",
            },
        )

        return False

    def read_event(
        self,
    ) -> dict[str, Any]:
        raw_length = self.headers.get(
            "Content-Length"
        )

        if raw_length is None:
            raise ValueError(
                "Content-Length obrigatório."
            )

        try:
            length = int(raw_length)

        except ValueError as exc:
            raise ValueError(
                "Content-Length inválido."
            ) from exc

        if (
            length <= 0
            or length
            > self.server.config.max_body_bytes
        ):
            raise ValueError(
                "Payload vazio ou acima do limite."
            )

        try:
            payload = json.loads(
                self.rfile.read(length).decode(
                    "utf-8"
                )
            )

        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise ValueError(
                "JSON inválido."
            ) from exc

        if not isinstance(payload, dict):
            raise ValueError(
                "Envelope precisa ser objeto JSON."
            )

        return payload

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self.send_json(
                HTTPStatus.OK,
                {
                    "contract_version":
                        POD_CONTRACT_VERSION,
                    "status": "ok",
                },
            )

            return

        if self.path == "/readyz":
            report = readiness_report(
                self.server.config
            )

            status = (
                HTTPStatus.OK
                if report["ready"]
                else HTTPStatus.SERVICE_UNAVAILABLE
            )

            self.send_json(
                status,
                report,
            )

            return

        prefix = "/v1/jobs/"

        if self.path.startswith(prefix):
            if not self.require_auth():
                return

            job_id = self.path[
                len(prefix):
            ].strip("/")

            try:
                uuid.UUID(job_id)

            except (
                ValueError,
                AttributeError,
            ):
                self.send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": "invalid_job_id",
                    },
                )

                return

            record = self.server.runner.get(
                job_id
            )

            if record is None:
                self.send_json(
                    HTTPStatus.NOT_FOUND,
                    {
                        "error": "job_not_found",
                    },
                )

                return

            self.send_json(
                HTTPStatus.OK,
                record.public(),
            )

            return

        self.send_json(
            HTTPStatus.NOT_FOUND,
            {
                "error": "not_found",
            },
        )

    def do_POST(self) -> None:
        if self.path != "/v1/jobs":
            self.send_json(
                HTTPStatus.NOT_FOUND,
                {
                    "error": "not_found",
                },
            )

            return

        if not self.require_auth():
            return

        report = readiness_report(
            self.server.config
        )

        if not report["ready"]:
            self.send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                report,
            )

            return

        try:
            event = self.read_event()

            record = self.server.runner.submit(
                event
            )

        except PodBusyError:
            self.send_json(
                HTTPStatus.CONFLICT,
                {
                    "error":
                        "pod_busy_single_flight",
                },
            )

            return

        except ValueError as exc:
            self.send_json(
                HTTPStatus.BAD_REQUEST,
                {
                    "error":
                        "invalid_request",
                    "message":
                        str(exc)[:300],
                },
            )

            return

        self.send_json(
            HTTPStatus.ACCEPTED,
            {
                "contract_version":
                    POD_CONTRACT_VERSION,
                "job_id":
                    record.job_id,
                "request_id":
                    record.request_id,
                "status":
                    record.status,
            },
        )


def main() -> None:
    config = PodServerConfig.from_env()
    config.validate_startup()
    start_storage_bootstrap()

    server = PrivacyPodHTTPServer(
        (
            config.host,
            config.port,
        ),
        PrivacyPodRequestHandler,
        config=config,
        runner=PodJobRunner(),
    )

    def shutdown(
        _signum,
        _frame,
    ) -> None:
        threading.Thread(
            target=server.shutdown,
            daemon=True,
        ).start()

    signal.signal(
        signal.SIGTERM,
        shutdown,
    )

    signal.signal(
        signal.SIGINT,
        shutdown,
    )

    print(
        json.dumps(
            {
                "event":
                    "privacy_pod_http_started",
                "contract_version":
                    POD_CONTRACT_VERSION,
                "host":
                    config.host,
                "port":
                    config.port,
                "single_flight":
                    True,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )

    try:
        server.serve_forever(
            poll_interval=0.5
        )

    finally:
        server.server_close()


if __name__ == "__main__":
    main()