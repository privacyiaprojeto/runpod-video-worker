import inspect

import pod_server


def test_pod_runtime_supports_r2_registry_without_serverless_fallback():
    assert "r2_registry" in pod_server.POD_SUPPORTED_MODEL_SOURCE_MODES
    source = inspect.getsource(pod_server.readiness_report)
    assert "MODEL_SOURCE_MODE_NOT_NETWORK_VOLUME" not in source
    assert "MODEL_STORAGE_BOOTSTRAP_IN_PROGRESS" in source


def test_pod_runtime_bootstrap_is_background_single_state_machine():
    source = inspect.getsource(pod_server.start_storage_bootstrap)
    assert "threading.Thread" in source
    assert "daemon=True" in source
