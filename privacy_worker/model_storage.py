from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Settings
from .errors import EphemeralDiskError, ModelStorageError
from .telemetry import log_event


MODEL_CATEGORIES = (
    "diffusion_models",
    "text_encoders",
    "vae",
    "clip_vision",
)
MODEL_SOURCE_MODES = {"network_volume", "cached_model"}
ONE_SHOT_LOCK_BACKENDS = {"filesystem", "r2"}
ALLOWED_TOP_LEVEL_METADATA = {".gitattributes", "README.md"}


@dataclass(frozen=True)
class ModelStorageState:
    mode: str
    model_root: Path
    snapshot_root: Path | None
    extra_model_paths_config: Path
    ephemeral_free_bytes: int | None


def cached_model_directory_name(model_id: str) -> str:
    parts = str(model_id or "").strip().split("/")
    component = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,95})$")
    if len(parts) != 2 or any(not component.fullmatch(part) for part in parts):
        raise ModelStorageError("CACHED_MODEL_ID precisa usar o formato seguro org/repo.")
    return f"models--{parts[0]}--{parts[1]}"


def _require_pinned_revision(revision: str) -> str:
    revision = str(revision or "").strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ModelStorageError(
            "CACHED_MODEL_REVISION deve ser um SHA Git imutável de exatamente "
            "40 caracteres hexadecimais."
        )
    return revision.lower()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _is_directory_link(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or bool(is_junction and is_junction())


def _create_directory_link(destination: Path, source: Path) -> None:
    try:
        destination.symlink_to(source, target_is_directory=True)
        return
    except OSError as exc:
        if os.name != "nt" or getattr(exc, "winerror", None) != 1314:
            raise
    try:
        subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(destination), str(source)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise OSError("Falha ao criar junction Windows equivalente ao symlink.") from exc


def resolve_cached_model_snapshot(settings: Settings) -> Path:
    repository_name = cached_model_directory_name(settings.cached_model_id)
    revision = _require_pinned_revision(settings.cached_model_revision)
    repository_root = settings.cached_model_cache_root / repository_name
    snapshots_root = repository_root / "snapshots"
    snapshot = snapshots_root / revision
    if not snapshot.is_dir():
        raise ModelStorageError(
            "Snapshot pinado do Cached Model não está disponível.",
            details={
                "cached_model_id": settings.cached_model_id,
                "cached_model_revision": revision,
                "snapshot_path": str(snapshot),
            },
        )
    try:
        resolved_snapshots = snapshots_root.resolve(strict=True)
        resolved_snapshot = snapshot.resolve(strict=True)
    except OSError as exc:
        raise ModelStorageError("Não foi possível resolver o snapshot pinado do Cached Model.") from exc
    if not _inside(resolved_snapshot, resolved_snapshots):
        raise ModelStorageError("O snapshot pinado resolve para fora do cache esperado.")
    return snapshot


def _expected_model_paths(snapshot: Path, settings: Settings) -> tuple[Path, ...]:
    return (
        snapshot / "diffusion_models" / settings.i2v_model_name,
        snapshot / "diffusion_models" / settings.v2v_model_name,
        snapshot / "text_encoders" / settings.text_encoder_name,
        snapshot / "vae" / settings.vae_name,
        snapshot / "clip_vision" / settings.clip_vision_name,
    )


def _validate_model_filenames(settings: Settings) -> None:
    configured = (
        settings.i2v_model_name,
        settings.v2v_model_name,
        settings.text_encoder_name,
        settings.vae_name,
        settings.clip_vision_name,
    )
    invalid = [
        name
        for name in configured
        if not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or Path(name).name != name
    ]
    if invalid:
        raise ModelStorageError(
            "Nomes de arquivos do Cached Model precisam ser basenames seguros.",
            details={"invalid_model_names": invalid},
        )


def _top_level_link_is_safe(entry: Path, repository_root: Path) -> bool:
    if not _is_directory_link(entry):
        return True
    try:
        resolved = entry.resolve(strict=True)
    except OSError:
        return False
    return _inside(resolved, repository_root)


def validate_cached_model_snapshot(snapshot: Path, settings: Settings) -> None:
    _validate_model_filenames(settings)
    try:
        entries = {entry.name: entry for entry in snapshot.iterdir()}
    except OSError as exc:
        raise ModelStorageError("Não foi possível listar o bundle do Cached Model.") from exc
    repository_root = snapshot.parent.parent.resolve(strict=True)
    missing_categories: list[str] = []
    unexpected_directories: list[str] = []
    unexpected_files: list[str] = []

    for category in MODEL_CATEGORIES:
        entry = entries.get(category)
        if (
            entry is None
            or not entry.is_dir()
            or not _top_level_link_is_safe(entry, repository_root)
        ):
            missing_categories.append(category)
            if entry is not None:
                target = unexpected_directories if entry.is_dir() else unexpected_files
                target.append(category)

    for name, entry in entries.items():
        if name in MODEL_CATEGORIES:
            continue
        if name in ALLOWED_TOP_LEVEL_METADATA:
            if entry.is_file() and _top_level_link_is_safe(entry, repository_root):
                continue
            target = unexpected_directories if entry.is_dir() else unexpected_files
            target.append(name)
            continue
        target = unexpected_directories if entry.is_dir() else unexpected_files
        target.append(name)

    if missing_categories or unexpected_directories or unexpected_files:
        raise ModelStorageError(
            "Layout top-level do Cached Model inválido.",
            details={
                "missing_categories": sorted(missing_categories),
                "unexpected_directories": sorted(unexpected_directories),
                "unexpected_files": sorted(unexpected_files),
            },
        )

    missing: list[str] = []
    escaped: list[str] = []
    for path in _expected_model_paths(snapshot, settings):
        if not path.is_file():
            missing.append(str(path))
            continue
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            missing.append(str(path))
            continue
        if not _inside(resolved, repository_root):
            escaped.append(str(path))
    if missing or escaped:
        raise ModelStorageError(
            "Arquivos obrigatórios do Cached Model estão ausentes ou fora do cache.",
            details={"missing_models": missing, "escaped_models": escaped},
        )


def create_writable_model_overlay(snapshot: Path, model_root: Path) -> Path:
    if _is_directory_link(model_root) or (model_root.exists() and not model_root.is_dir()):
        raise ModelStorageError("MODEL_ROOT do overlay precisa ser um diretório local real.")
    model_root.mkdir(parents=True, exist_ok=True)
    for category in MODEL_CATEGORIES:
        source = (snapshot / category).resolve(strict=True)
        destination = model_root / category
        if _is_directory_link(destination):
            try:
                current = destination.resolve(strict=True)
            except OSError as exc:
                raise ModelStorageError(f"Symlink quebrado no overlay: {destination}.") from exc
            if current != source:
                raise ModelStorageError(f"Symlink do overlay aponta para fonte inesperada: {destination}.")
            continue
        if destination.exists():
            raise ModelStorageError(
                f"O overlay não substituirá diretório ou arquivo existente: {destination}."
            )
        try:
            _create_directory_link(destination, source)
        except OSError as exc:
            raise ModelStorageError(f"Falha ao criar symlink do overlay: {destination}.") from exc

    loras = model_root / "loras"
    if _is_directory_link(loras) or (loras.exists() and not loras.is_dir()):
        raise ModelStorageError("MODEL_ROOT/loras precisa ser um diretório local gravável.")
    loras.mkdir(parents=True, exist_ok=True)
    probe = loras / f".write-probe-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        with probe.open("x", encoding="utf-8") as handle:
            handle.write("writable\n")
    except OSError as exc:
        raise ModelStorageError("MODEL_ROOT/loras não é gravável.") from exc
    finally:
        probe.unlink(missing_ok=True)
    return model_root


def write_extra_model_paths_config(settings: Settings) -> Path:
    settings.runtime_root.mkdir(parents=True, exist_ok=True)
    path = settings.runtime_root / "extra_model_paths.runtime.yaml"
    payload: dict[str, Any] = {
        "privacy_wan_models": {
            "base_path": str(settings.model_root),
            "diffusion_models": "diffusion_models",
            "text_encoders": "text_encoders",
            "vae": "vae",
            "clip_vision": "clip_vision",
            "loras": "loras",
        }
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if not path.is_file() or path.read_text(encoding="utf-8") != rendered:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(rendered, encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return path


def ensure_ephemeral_disk_ready(settings: Settings) -> dict[str, int | float] | None:
    if settings.model_source_mode != "cached_model":
        return None
    roots = (settings.runtime_root, settings.model_root)
    free_values: list[int] = []
    for root in roots:
        try:
            free_values.append(int(shutil.disk_usage(root).free))
        except OSError as exc:
            raise EphemeralDiskError(
                "Não foi possível medir o espaço efêmero disponível.",
                details={"filesystem_root": str(root)},
            ) from exc
    free_bytes = min(free_values)
    required_bytes = int(settings.ephemeral_min_free_gb * 1024**3)
    if free_bytes < required_bytes:
        raise EphemeralDiskError(
            "Espaço efêmero insuficiente antes de baixar adapter ou mídia.",
            details={
                "free_bytes": free_bytes,
                "free_gb": round(free_bytes / 1024**3, 3),
                "required_bytes": required_bytes,
                "required_gb": settings.ephemeral_min_free_gb,
            },
        )
    return {
        "free_bytes": free_bytes,
        "free_gb": round(free_bytes / 1024**3, 3),
        "required_bytes": required_bytes,
        "required_gb": settings.ephemeral_min_free_gb,
    }


def _validate_storage_settings(settings: Settings) -> None:
    if settings.model_source_mode not in MODEL_SOURCE_MODES:
        raise ModelStorageError("MODEL_SOURCE_MODE inválido; use network_volume ou cached_model.")
    if settings.identity_one_shot_lock_backend not in ONE_SHOT_LOCK_BACKENDS:
        raise ModelStorageError(
            "IDENTITY_ONE_SHOT_LOCK_BACKEND inválido; use filesystem ou r2."
        )
    if settings.model_source_mode == "cached_model":
        if settings.identity_one_shot_lock_backend != "r2":
            raise ModelStorageError(
                "MODEL_SOURCE_MODE=cached_model exige IDENTITY_ONE_SHOT_LOCK_BACKEND=r2."
            )
        if not settings.r2_configured:
            raise ModelStorageError(
                "O backend global r2 exige o R2 privado já configurado no worker."
            )
        cache_root = settings.cached_model_cache_root.resolve(strict=False)
        model_root = settings.model_root.resolve(strict=False)
        runtime_root = settings.runtime_root.resolve(strict=False)
        if _inside(model_root, runtime_root) or _inside(runtime_root, model_root):
            raise ModelStorageError("MODEL_ROOT e RUNTIME_ROOT precisam ser raízes separadas.")
        if _is_directory_link(settings.runtime_root):
            raise ModelStorageError("RUNTIME_ROOT efêmero precisa ser um diretório local real.")
        for name, root in (
            ("MODEL_ROOT", settings.model_root),
            ("RUNTIME_ROOT", settings.runtime_root),
        ):
            resolved = root.resolve(strict=False)
            if _inside(resolved, cache_root) or _inside(cache_root, resolved):
                raise ModelStorageError(f"{name} não pode sobrepor o cache read-only de modelos.")


def prepare_model_storage(settings: Settings) -> ModelStorageState:
    _validate_storage_settings(settings)
    snapshot: Path | None = None
    disk: dict[str, int | float] | None = None
    if settings.model_source_mode == "cached_model":
        snapshot = resolve_cached_model_snapshot(settings)
        validate_cached_model_snapshot(snapshot, settings)
        create_writable_model_overlay(snapshot, settings.model_root)
        settings.ensure_runtime_dirs()
        disk = ensure_ephemeral_disk_ready(settings)
    else:
        settings.ensure_runtime_dirs()

    extra_paths = write_extra_model_paths_config(settings)
    free_bytes = int(disk["free_bytes"]) if disk else None
    log_event(
        "model_storage_ready",
        model_source_mode=settings.model_source_mode,
        cached_model_id=settings.cached_model_id or None,
        cached_model_revision=(settings.cached_model_revision[:12] or None),
        ephemeral_free_bytes=free_bytes,
        ephemeral_free_gb=(disk["free_gb"] if disk else None),
        one_shot_lock_backend=settings.identity_one_shot_lock_backend,
    )
    return ModelStorageState(
        mode=settings.model_source_mode,
        model_root=settings.model_root,
        snapshot_root=snapshot,
        extra_model_paths_config=extra_paths,
        ephemeral_free_bytes=free_bytes,
    )
