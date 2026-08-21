class WorkerError(Exception):
    """Base exception with a stable machine-readable code."""

    code = "WORKER_ERROR"
    retryable = False

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


class ContractError(WorkerError):
    code = "INVALID_PRODUCTION_CONTRACT"


class DownloadError(WorkerError):
    code = "MEDIA_DOWNLOAD_FAILED"
    retryable = True


class WorkflowError(WorkerError):
    code = "WORKFLOW_ERROR"


class ModelStorageError(WorkerError):
    code = "MODEL_STORAGE_CONFIGURATION_ERROR"


class EphemeralDiskError(WorkerError):
    code = "EPHEMERAL_DISK_NOT_READY"


class ComfyUIError(WorkerError):
    code = "COMFYUI_RUNTIME_ERROR"
    retryable = True


class OutputError(WorkerError):
    code = "OUTPUT_ERROR"


class LoraCompatibilityError(WorkerError):
    code = "LORA_KEY_FORMAT_MISMATCH"


class LoraNotAppliedError(WorkerError):
    code = "LORA_NOT_APPLIED"


class ABOutputsIdenticalError(WorkerError):
    code = "AB_OUTPUTS_IDENTICAL"
