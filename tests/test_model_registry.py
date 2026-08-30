import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from privacy_worker.config import Settings
from privacy_worker.errors import ModelStorageError
from privacy_worker.model_registry import (
    CANONICAL_MODEL_BUNDLE_BYTES,
    CANONICAL_MODEL_OBJECTS,
    CANONICAL_MODEL_REGISTRY_BUCKET,
    materialize_canonical_model_registry,
)


class FakeS3:
    def __init__(self, objects):
        self.objects = dict(objects)
        self.head_calls = []
        self.download_calls = []

    def head_object(self, *, Bucket, Key):
        self.head_calls.append((Bucket, Key))
        data = self.objects[Key]
        return {
            "ContentLength": len(data),
            "Metadata": {"sha256": hashlib.sha256(data).hexdigest()},
        }

    def download_file(self, bucket, key, destination, Config=None):
        self.download_calls.append((bucket, key))
        Path(destination).write_bytes(self.objects[key])


def registry_settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "model_source_mode": "r2_registry",
        "model_root": tmp_path / "models",
        "runtime_root": tmp_path / "runtime",
        "ephemeral_min_free_gb": 0,
        "model_registry_r2_endpoint_url": "https://registry.invalid",
        "model_registry_r2_access_key_id": "registry-access",
        "model_registry_r2_secret_access_key": "registry-secret",
        "model_registry_r2_bucket_name": CANONICAL_MODEL_REGISTRY_BUCKET,
    }
    values.update(overrides)
    return replace(Settings(), **values)


def tiny_specs():
    rows = []
    payloads = {
        "models/diffusion_models/a.safetensors": b"aaa",
        "models/text_encoders/b.safetensors": b"bbbb",
        "models/vae/c.safetensors": b"ccccc",
        "models/clip_vision/d.safetensors": b"dddddd",
    }
    for key, data in payloads.items():
        rows.append(
            {
                "key": key,
                "relative_path": key.removeprefix("models/"),
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return tuple(rows), payloads


def test_canonical_registry_lock_is_exact_and_large():
    assert CANONICAL_MODEL_REGISTRY_BUCKET == "ia-adulta-model-registry"
    assert len(CANONICAL_MODEL_OBJECTS) == 5
    assert CANONICAL_MODEL_BUNDLE_BYTES == 75720642755
    assert {row["relative_path"] for row in CANONICAL_MODEL_OBJECTS} == {
        "diffusion_models/wan2.1_vace_14B_fp16.safetensors",
        "diffusion_models/wan2.1_i2v_480p_14B_fp16.safetensors",
        "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "vae/wan_2.1_vae.safetensors",
        "clip_vision/clip_vision_h.safetensors",
    }


def test_materialize_downloads_verifies_and_reuses_local_marker(tmp_path):
    specs, payloads = tiny_specs()
    settings = registry_settings(tmp_path)
    client = FakeS3(payloads)

    first = materialize_canonical_model_registry(
        settings,
        client=client,
        specs=specs,
    )

    assert first["downloaded_count"] == len(specs)
    assert first["reused_count"] == 0
    assert len(client.head_calls) == len(specs)
    assert len(client.download_calls) == len(specs)

    for row in specs:
        path = settings.model_root / row["relative_path"]
        assert path.read_bytes() == payloads[row["key"]]

    client.head_calls.clear()
    client.download_calls.clear()

    second = materialize_canonical_model_registry(
        settings,
        client=client,
        specs=specs,
    )

    assert second["downloaded_count"] == 0
    assert second["reused_count"] == len(specs)
    assert client.head_calls == []
    assert client.download_calls == []


def test_registry_bucket_is_fail_closed(tmp_path):
    specs, payloads = tiny_specs()
    settings = registry_settings(
        tmp_path,
        model_registry_r2_bucket_name="privacy-media",
    )
    with pytest.raises(ModelStorageError, match="MASTER canônico"):
        materialize_canonical_model_registry(
            settings,
            client=FakeS3(payloads),
            specs=specs,
        )


def test_hash_mismatch_never_promotes_part_file(tmp_path):
    specs, payloads = tiny_specs()
    broken_payloads = dict(payloads)
    first_key = specs[0]["key"]
    broken_payloads[first_key] = b"xxx"
    settings = registry_settings(tmp_path)

    with pytest.raises(ModelStorageError, match="tamanho divergente|SHA-256.*diverge"):
        materialize_canonical_model_registry(
            settings,
            client=FakeS3(broken_payloads),
            specs=specs,
        )

    target = settings.model_root / specs[0]["relative_path"]
    assert not target.exists()
    assert list(target.parent.glob("*.part")) == []
