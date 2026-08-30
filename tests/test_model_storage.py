import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from privacy_worker.config import Settings
from privacy_worker.config import _storage_path
from privacy_worker.errors import EphemeralDiskError, ModelStorageError
from privacy_worker.model_storage import (
    MODEL_CATEGORIES,
    _create_directory_link,
    _require_pinned_revision,
    cached_model_directory_name,
    create_writable_model_overlay,
    ensure_ephemeral_disk_ready,
    prepare_model_storage,
    resolve_cached_model_snapshot,
    validate_cached_model_snapshot,
    write_extra_model_paths_config,
)
from privacy_worker.telemetry import log_event


REVISION = "0123456789abcdef0123456789abcdef01234567"


def cached_settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "model_source_mode": "cached_model",
        "cached_model_id": "privacy/wan-bundle",
        "cached_model_revision": REVISION,
        "cached_model_cache_root": tmp_path / "cache" / "hub",
        "model_root": tmp_path / "ephemeral" / "privacy-models",
        "runtime_root": tmp_path / "ephemeral" / "privacy-wan-runtime",
        "identity_one_shot_lock_backend": "r2",
        "identity_one_shot_lock_prefix": "tests/private",
        "ephemeral_min_free_gb": 0,
        "r2_endpoint_url": "https://example.invalid",
        "r2_access_key_id": "test-access",
        "r2_secret_access_key": "test-secret",
        "r2_bucket_name": "private-test",
    }
    values.update(overrides)
    return replace(Settings(), **values)


def create_snapshot(settings: Settings) -> Path:
    snapshot = (
        settings.cached_model_cache_root
        / cached_model_directory_name(settings.cached_model_id)
        / "snapshots"
        / settings.cached_model_revision
    )
    for category in MODEL_CATEGORIES:
        (snapshot / category).mkdir(parents=True, exist_ok=True)
    required = (
        snapshot / "diffusion_models" / settings.i2v_model_name,
        snapshot / "diffusion_models" / settings.v2v_model_name,
        snapshot / "text_encoders" / settings.text_encoder_name,
        snapshot / "vae" / settings.vae_name,
        snapshot / "clip_vision" / settings.clip_vision_name,
    )
    for path in required:
        path.write_bytes(b"test-model-header")
    return snapshot


def test_cached_model_id_maps_to_huggingface_cache_directory():
    assert cached_model_directory_name("privacy/wan-bundle") == "models--privacy--wan-bundle"
    with pytest.raises(ModelStorageError, match="org/repo"):
        cached_model_directory_name("privacy/wan/bundle")


def test_cached_mode_replaces_inherited_legacy_roots(monkeypatch):
    monkeypatch.setenv("MODEL_SOURCE_MODE", "cached_model")
    monkeypatch.setenv("MODEL_ROOT", "/runpod-volume/models/")
    assert _storage_path(
        "MODEL_ROOT", legacy="/runpod-volume/models", cached="/tmp/privacy-models"
    ) == Path("/tmp/privacy-models")


def test_r2_registry_mode_replaces_inherited_legacy_roots(monkeypatch):
    monkeypatch.setenv("MODEL_SOURCE_MODE", "r2_registry")
    monkeypatch.setenv("MODEL_ROOT", "/runpod-volume/models")
    assert _storage_path(
        "MODEL_ROOT", legacy="/runpod-volume/models", cached="/tmp/privacy-models"
    ) == Path("/tmp/privacy-models")


@pytest.mark.parametrize(
    "revision",
    ("", "main", "release-v1", "a" * 39, "g" * 40),
    ids=("empty", "main", "tag", "short-sha", "non-hex"),
)
def test_cached_model_revision_requires_immutable_sha40(revision):
    with pytest.raises(ModelStorageError, match="SHA Git imutável"):
        _require_pinned_revision(revision)


def test_cached_model_revision_accepts_sha40_and_normalizes_uppercase():
    assert _require_pinned_revision(REVISION) == REVISION
    assert _require_pinned_revision(REVISION.upper()) == REVISION


def test_resolver_uses_normalized_lowercase_sha40_directory(tmp_path):
    uppercase_settings = cached_settings(tmp_path, cached_model_revision=REVISION.upper())
    lowercase_settings = replace(uppercase_settings, cached_model_revision=REVISION)
    snapshot = create_snapshot(lowercase_settings)
    assert resolve_cached_model_snapshot(uppercase_settings) == snapshot


def test_missing_pinned_snapshot_fails_closed(tmp_path):
    settings = cached_settings(tmp_path)
    with pytest.raises(ModelStorageError, match="Snapshot pinado"):
        resolve_cached_model_snapshot(settings)


@pytest.mark.parametrize(
    "metadata_files",
    ((), (".gitattributes",), ("README.md",), (".gitattributes", "README.md")),
    ids=("categories-only", "gitattributes", "readme", "all-metadata"),
)
def test_top_level_allows_only_optional_huggingface_metadata(tmp_path, metadata_files):
    settings = cached_settings(tmp_path)
    snapshot = create_snapshot(settings)
    for name in metadata_files:
        (snapshot / name).write_text("metadata\n", encoding="utf-8")
    validate_cached_model_snapshot(snapshot, settings)


def test_required_model_file_remains_mandatory(tmp_path):
    settings = cached_settings(tmp_path)
    snapshot = create_snapshot(settings)
    (snapshot / "vae" / settings.vae_name).unlink()
    with pytest.raises(ModelStorageError, match="Arquivos obrigatórios"):
        validate_cached_model_snapshot(snapshot, settings)


def test_extra_top_level_directory_is_blocked(tmp_path):
    settings = cached_settings(tmp_path)
    snapshot = create_snapshot(settings)
    (snapshot / "models").mkdir()
    with pytest.raises(ModelStorageError, match="Layout top-level") as captured:
        validate_cached_model_snapshot(snapshot, settings)
    assert captured.value.details == {
        "missing_categories": [],
        "unexpected_directories": ["models"],
        "unexpected_files": [],
    }


def test_unexpected_top_level_file_is_blocked(tmp_path):
    settings = cached_settings(tmp_path)
    snapshot = create_snapshot(settings)
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ModelStorageError, match="Layout top-level") as captured:
        validate_cached_model_snapshot(snapshot, settings)
    assert captured.value.details == {
        "missing_categories": [],
        "unexpected_directories": [],
        "unexpected_files": ["config.json"],
    }


def test_missing_top_level_category_is_reported_separately(tmp_path):
    settings = cached_settings(tmp_path)
    snapshot = create_snapshot(settings)
    clip_dir = snapshot / "clip_vision"
    (clip_dir / settings.clip_vision_name).unlink()
    clip_dir.rmdir()
    with pytest.raises(ModelStorageError, match="Layout top-level") as captured:
        validate_cached_model_snapshot(snapshot, settings)
    assert captured.value.details == {
        "missing_categories": ["clip_vision"],
        "unexpected_directories": [],
        "unexpected_files": [],
    }


def test_top_level_category_link_cannot_escape_repository_cache(tmp_path):
    settings = cached_settings(tmp_path)
    snapshot = create_snapshot(settings)
    diffusion_dir = snapshot / "diffusion_models"
    for path in tuple(diffusion_dir.iterdir()):
        path.unlink()
    diffusion_dir.rmdir()

    outside = tmp_path / "outside-diffusion-models"
    outside.mkdir()
    (outside / settings.i2v_model_name).write_bytes(b"model")
    (outside / settings.v2v_model_name).write_bytes(b"model")
    _create_directory_link(diffusion_dir, outside)

    with pytest.raises(ModelStorageError, match="Layout top-level") as captured:
        validate_cached_model_snapshot(snapshot, settings)
    assert captured.value.details == {
        "missing_categories": ["diffusion_models"],
        "unexpected_directories": ["diffusion_models"],
        "unexpected_files": [],
    }


def test_cached_model_filenames_cannot_escape_categories(tmp_path):
    settings = cached_settings(tmp_path, v2v_model_name="../outside.safetensors")
    snapshot = create_snapshot(replace(settings, v2v_model_name="v2v.safetensors"))
    with pytest.raises(ModelStorageError, match="basenames seguros"):
        validate_cached_model_snapshot(snapshot, settings)


def test_overlay_symlinks_read_only_categories_and_keeps_loras_writable(tmp_path):
    settings = cached_settings(tmp_path)
    snapshot = create_snapshot(settings)
    validate_cached_model_snapshot(snapshot, settings)
    create_writable_model_overlay(snapshot, settings.model_root)
    create_writable_model_overlay(snapshot, settings.model_root)

    for category in MODEL_CATEGORIES:
        link = settings.model_root / category
        is_junction = getattr(link, "is_junction", lambda: False)
        assert link.is_symlink() or is_junction()
        assert link.resolve() == (snapshot / category).resolve()
    assert (settings.model_root / "diffusion_models" / settings.v2v_model_name).is_file()
    writable = settings.model_root / "loras" / "translated.safetensors"
    writable.write_bytes(b"translated")
    assert writable.read_bytes() == b"translated"


def test_dynamic_extra_model_paths_uses_settings_model_root(tmp_path):
    settings = replace(
        Settings(),
        runtime_root=tmp_path / "runtime",
        model_root=tmp_path / "models",
    )
    path = write_extra_model_paths_config(settings)
    payload = json.loads(path.read_text(encoding="utf-8"))
    configured = payload["privacy_wan_models"]
    assert configured["base_path"] == str(settings.model_root)
    assert tuple(key for key in configured if key != "base_path") == (
        "diffusion_models",
        "text_encoders",
        "vae",
        "clip_vision",
        "loras",
    )
    assert "secret" not in path.read_text(encoding="utf-8").lower()


def test_legacy_filesystem_defaults_keep_network_volume_behavior(tmp_path):
    settings = replace(
        Settings(),
        model_source_mode="network_volume",
        identity_one_shot_lock_backend="filesystem",
        runtime_root=tmp_path / "legacy-runtime",
        model_root=tmp_path / "legacy-models",
    )
    state = prepare_model_storage(settings)
    assert state.mode == "network_volume"
    assert state.snapshot_root is None
    assert settings.input_dir.is_dir()
    assert settings.output_dir.is_dir()
    assert settings.temp_dir.is_dir()
    assert not settings.model_root.exists()


def test_cached_mode_requires_global_r2_lock(tmp_path):
    settings = cached_settings(tmp_path, identity_one_shot_lock_backend="filesystem")
    with pytest.raises(ModelStorageError, match="exige IDENTITY_ONE_SHOT_LOCK_BACKEND=r2"):
        prepare_model_storage(settings)


def test_ephemeral_disk_readiness_is_fail_closed(monkeypatch, tmp_path):
    settings = cached_settings(tmp_path, ephemeral_min_free_gb=10)
    settings.runtime_root.mkdir(parents=True)
    settings.model_root.mkdir(parents=True)
    monkeypatch.setattr(
        "privacy_worker.model_storage.shutil.disk_usage",
        lambda path: SimpleNamespace(free=9 * 1024**3),
    )
    with pytest.raises(EphemeralDiskError, match="Espaço efêmero insuficiente"):
        ensure_ephemeral_disk_ready(settings)

    monkeypatch.setattr(
        "privacy_worker.model_storage.shutil.disk_usage",
        lambda path: SimpleNamespace(free=11 * 1024**3),
    )
    readiness = ensure_ephemeral_disk_ready(settings)
    assert readiness["free_bytes"] == 11 * 1024**3
    assert readiness["required_gb"] == 10


def test_prepare_cached_storage_is_local_and_idempotent(tmp_path, capsys):
    settings = cached_settings(tmp_path)
    snapshot = create_snapshot(settings)
    first = prepare_model_storage(settings)
    second = prepare_model_storage(settings)
    assert first.snapshot_root == second.snapshot_root == snapshot
    assert first.model_root == settings.model_root
    telemetry = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert telemetry["model_source_mode"] == "cached_model"
    assert telemetry["cached_model_id"] == settings.cached_model_id
    assert telemetry["cached_model_revision"] == REVISION[:12]
    assert isinstance(telemetry["ephemeral_free_bytes"], int)
    assert telemetry["one_shot_lock_backend"] == "r2"


def test_security_telemetry_redacts_storage_credentials_and_private_urls(capsys):
    log_event(
        "security-redaction-test",
        r2_secret_access_key="r2-secret-value",
        r2_access_key_id="r2-access-value",
        hf_token="hf-secret-value",
        signed_url="https://private.invalid/signed-token",
        kyc_url="https://private.invalid/kyc-face",
    )
    rendered = capsys.readouterr().out
    for secret in (
        "r2-secret-value",
        "r2-access-value",
        "hf-secret-value",
        "signed-token",
        "kyc-face",
    ):
        assert secret not in rendered
