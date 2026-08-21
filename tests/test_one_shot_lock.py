import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from privacy_worker.config import Settings
from privacy_worker.errors import ContractError
from privacy_worker.one_shot_lock import reserve_r2_one_shot, r2_one_shot_lock_key


class FakeS3Error(Exception):
    def __init__(self, code="InternalError", status=500):
        super().__init__(code)
        self.response = {
            "Error": {"Code": code},
            "ResponseMetadata": {"HTTPStatusCode": status},
        }


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.calls = []
        self.fail = False

    def put_object(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise FakeS3Error()
        identity = (kwargs["Bucket"], kwargs["Key"])
        if kwargs.get("IfNoneMatch") == "*" and identity in self.objects:
            raise FakeS3Error("PreconditionFailed", 412)
        self.objects[identity] = bytes(kwargs["Body"])
        return {"ETag": '"test"'}


def request(request_id="request-1", actor="actor", run="run", adapter="adapter"):
    return SimpleNamespace(
        request_id=request_id,
        actor_profile_id=actor,
        training_run_id=run,
        adapter_id=adapter,
        contract_version="privacy-identity-motion-abc-v1",
    )


def settings() -> Settings:
    return replace(
        Settings(),
        identity_one_shot_lock_backend="r2",
        identity_one_shot_lock_prefix="wan/private-locks",
        r2_endpoint_url="https://account.example.invalid",
        r2_access_key_id="ACCESS-CREDENTIAL-MUST-NOT-LEAK",
        r2_secret_access_key="SECRET-CREDENTIAL-MUST-NOT-LEAK",
        r2_bucket_name="private-bucket",
    )


def payload(item):
    return {
        "lock_version": 2,
        "request_id": item.request_id,
        "actor_profile_id": item.actor_profile_id,
        "training_run_id": item.training_run_id,
        "adapter_id": item.adapter_id,
        "contract_version": item.contract_version,
        "status": "reserved",
        "automatic_retry": False,
    }


def test_first_atomic_reservation_succeeds_and_duplicate_is_denied():
    client = FakeS3()
    item = request()
    lock = reserve_r2_one_shot(client, item, settings(), payload(item))
    assert client.calls[0]["IfNoneMatch"] == "*"
    assert lock.payload["status"] == "reserved"
    with pytest.raises(ContractError, match="já foi reservado"):
        reserve_r2_one_shot(client, item, settings(), payload(item))


def test_different_request_and_scope_create_different_keys():
    configured = settings()
    base = r2_one_shot_lock_key(request(), configured)
    keys = {
        r2_one_shot_lock_key(request(request_id="request-2"), configured),
        r2_one_shot_lock_key(request(actor="actor-2"), configured),
        r2_one_shot_lock_key(request(run="run-2"), configured),
        r2_one_shot_lock_key(request(adapter="adapter-2"), configured),
    }
    assert base not in keys
    assert len(keys) == 4
    assert base.endswith(".json")
    assert "/identity-one-shot/privacy-identity-motion-abc-v1/" in base
    client = FakeS3()
    first = request()
    second = request(request_id="request-2")
    reserve_r2_one_shot(client, first, configured, payload(first))
    reserve_r2_one_shot(client, second, configured, payload(second))
    assert len(client.objects) == 2


def test_r2_reservation_failure_is_fail_closed():
    client = FakeS3()
    client.fail = True
    item = request()
    with pytest.raises(ContractError, match="execução bloqueada"):
        reserve_r2_one_shot(client, item, settings(), payload(item))


def test_status_update_overwrites_same_object_and_never_releases_lock():
    client = FakeS3()
    item = request()
    lock = reserve_r2_one_shot(client, item, settings(), payload(item))
    identity = (lock.bucket, lock.key)
    lock.update("running", workflow_id="workflow-1")
    assert len(client.objects) == 1
    running = json.loads(client.objects[identity])
    assert running["status"] == "running"
    assert running["workflow_id"] == "workflow-1"
    assert "IfNoneMatch" not in client.calls[-1]

    client.fail = True
    with pytest.raises(ContractError, match="reserva permanece ativa"):
        lock.update("completed", asset_count=3)
    assert json.loads(client.objects[identity])["status"] == "running"


def test_lock_payload_never_contains_r2_credentials():
    client = FakeS3()
    item = request()
    configured = settings()
    lock = reserve_r2_one_shot(client, item, configured, payload(item))
    lock.update(
        "failed",
        error_code="TEST_ERROR",
        r2_secret_access_key=configured.r2_secret_access_key,
        signed_url="https://secret.invalid/token",
    )
    serialized = client.objects[(lock.bucket, lock.key)].decode("utf-8")
    assert configured.r2_access_key_id not in serialized
    assert configured.r2_secret_access_key not in serialized
    assert "secret.invalid" not in serialized
    assert "signed_url" not in serialized


def test_identity_and_motion_wrappers_share_r2_backend(monkeypatch):
    from privacy_worker import identity_ab, identity_motion_abc

    client = FakeS3()
    configured = settings()
    monkeypatch.setattr(identity_ab, "r2_client", lambda current: client)
    first = request(request_id="identity-wrapper")
    second = request(request_id="motion-wrapper")
    identity_lock = identity_ab.reserve_one_shot(first, configured)
    motion_lock = identity_motion_abc.reserve_one_shot(second, configured)
    identity_ab.update_lock(identity_lock, "running", workflow_id="identity")
    identity_motion_abc.update_lock(motion_lock, "running", workflow_id="motion")
    assert identity_lock.key != motion_lock.key
    assert len(client.objects) == 2
