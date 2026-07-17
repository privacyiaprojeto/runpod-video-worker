from __future__ import annotations

import ipaddress
import mimetypes
import socket
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

from .config import Settings
from .errors import DownloadError
from .telemetry import log_event

MAX_REDIRECTS = 5


def _validate_public_http_url(url: str, settings: Settings) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise DownloadError("A mídia precisa usar uma URL http(s) válida.")

    hostname = parsed.hostname.lower()
    if settings.media_allowed_hosts and hostname not in settings.media_allowed_hosts:
        if not any(hostname.endswith("." + allowed) for allowed in settings.media_allowed_hosts):
            raise DownloadError("Host de mídia fora da allowlist configurada.")

    if settings.allow_private_download_hosts:
        return

    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(hostname, parsed.port or 443)}
    except socket.gaierror as error:
        raise DownloadError("Não foi possível resolver o host da mídia.") from error

    for address in addresses:
        ip = ipaddress.ip_address(address)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise DownloadError("Download bloqueado para endereço de rede não público.")


def _extension(url: str, content_type: str, fallback: str) -> str:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix and len(suffix) <= 8:
        return suffix
    guessed = mimetypes.guess_extension(content_type.split(";")[0].strip()) if content_type else None
    return guessed or fallback


def _request_with_safe_redirects(
    session: requests.Session,
    *,
    url: str,
    settings: Settings,
) -> requests.Response:
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        _validate_public_http_url(current, settings)
        response = session.get(
            current,
            stream=True,
            timeout=(15, settings.download_timeout_seconds),
            allow_redirects=False,
            headers={"User-Agent": "privacy-wan-worker/2.0"},
        )
        if response.is_redirect or response.is_permanent_redirect:
            location = response.headers.get("location")
            response.close()
            if not location:
                raise DownloadError("Redirecionamento de mídia sem destino.")
            current = urljoin(current, location)
            continue
        return response
    raise DownloadError(f"Mídia excedeu o limite de {MAX_REDIRECTS} redirecionamentos.")


def download_media(
    *,
    url: str,
    destination_dir: Path,
    stem: str,
    max_mb: int,
    fallback_extension: str,
    settings: Settings,
    request_id: str,
    expected_content_prefix: str | None = None,
) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = max_mb * 1024 * 1024
    downloaded = 0
    output_path: Path | None = None

    try:
        with requests.Session() as session:
            response = _request_with_safe_redirects(session, url=url, settings=settings)
            with response:
                response.raise_for_status()
                content_length = int(response.headers.get("content-length") or 0)
                if content_length > max_bytes:
                    raise DownloadError(f"Mídia excede o limite de {max_mb} MB.")

                content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
                if expected_content_prefix and content_type:
                    if not (
                        content_type.startswith(expected_content_prefix)
                        or content_type == "application/octet-stream"
                    ):
                        raise DownloadError(
                            f"Content-Type inesperado: {content_type}; esperado {expected_content_prefix}*."
                        )

                suffix = _extension(response.url, content_type, fallback_extension)
                output_path = destination_dir / f"{stem}{suffix}"
                with output_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        downloaded += len(chunk)
                        if downloaded > max_bytes:
                            raise DownloadError(
                                f"Mídia excedeu o limite de {max_mb} MB durante o download."
                            )
                        handle.write(chunk)
    except DownloadError:
        if output_path:
            output_path.unlink(missing_ok=True)
        raise
    except requests.RequestException as error:
        if output_path:
            output_path.unlink(missing_ok=True)
        raise DownloadError("Falha ao baixar mídia assinada.") from error

    if not output_path or not output_path.exists() or output_path.stat().st_size <= 0:
        raise DownloadError("O download de mídia retornou um arquivo vazio.")

    log_event(
        "media_downloaded",
        request_id=request_id,
        stem=stem,
        size_bytes=output_path.stat().st_size,
        extension=output_path.suffix,
    )
    return output_path
