from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import urlsplit


SENSITIVE_KEYS = {
    "signed_url",
    "primary_reference_url",
    "base_video_url",
    "source_image_url",
    "url",
    "authorization",
    "token",
    "secret",
    "r2_access_key_id",
    "r2_secret_access_key",
    "hf_token",
    "kyc_url",
    "reference_image_url",
}


def now_ms() -> int:
    return int(time.time() * 1000)


def _redact(value: Any, key: str | None = None) -> Any:
    normalized_key = key.lower() if key else ""
    if normalized_key in SENSITIVE_KEYS or any(
        marker in normalized_key for marker in ("secret", "credential", "signed_url")
    ):
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            parsed = urlsplit(value)
            return f"{parsed.scheme}://{parsed.netloc}/[redacted]"
        return "[redacted]"
    if isinstance(value, dict):
        return {str(k): _redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str) and len(value) > 500:
        return value[:500] + "...[truncated]"
    return value


def log_event(event: str, *, level: str = "INFO", request_id: str | None = None, **fields: Any) -> None:
    payload = {
        "timestamp_ms": now_ms(),
        "level": level.upper(),
        "event": event,
        "request_id": request_id,
        **_redact(fields),
    }
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)
