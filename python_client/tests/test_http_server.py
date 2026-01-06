import os

# IMPORTANT: Set these BEFORE importing http_server, as it reads env vars at import time
os.environ["KT_LOG_STREAMING_ENABLED"] = "false"
os.environ["KT_METRICS_ENABLED"] = "false"

import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from kubetorch.serving import http_server
from kubetorch.serving.http_server import app

from .utils import _update_metadata_env_vars, load_callable_from_test_dir


def load_test_assets(names):
    """Get all test directories from the assets directory"""
    assets_dir = Path(__file__).parent / "assets"
    return [str(d) for d in assets_dir.glob("*") if d.is_dir() and d.name in names]


def _reset_http_server_state():
    """Reset all http_server globals to ensure test isolation."""
    if http_server.SUPERVISOR:
        try:
            http_server.SUPERVISOR.cleanup()
        except Exception as e:
            print(f"Warning: Supervisor cleanup failed: {e}")

    http_server.SUPERVISOR = None
    http_server._CACHED_CALLABLES.clear()
    http_server._LAST_DEPLOYED = 0
    http_server._CACHED_IMAGE.clear()

    # Give processes and threads time to fully terminate
    time.sleep(0.2)


@pytest.fixture(scope="class")
def reset_state_per_class(request):
    """Reset http_server state before and after each test class.

    Using class scope to avoid restarting server for every test within a class.
    """
    _reset_http_server_state()
    yield
    _reset_http_server_state()


@pytest.fixture(scope="class")
def setup_test_env(request, reset_state_per_class):
    """Set up environment for a specific test directory.

    This fixture (class-scoped):
    1. Resets http_server state once per test class
    2. Sets up environment variables for the test
    3. Loads test inputs from the assets directory
    4. Cleans up environment after test class completes
    """
    test_dir = request.param
    assets_dir = Path(__file__).parent / "assets" / test_dir

    # Set environment variables
    os.environ["KT_DIRECTORY"] = str(assets_dir)
    os.environ["POD_NAMESPACE"] = "kubetorch"
    os.environ["POD_NAME"] = "kubetorch-test"
    os.environ["POD_IP"] = "localhost"
    os.environ["LOCAL_IPS"] = "localhost"
    os.environ["KT_SERVICE_NAME"] = test_dir

    _update_metadata_env_vars(assets_dir, set=True)

    # Load test inputs
    with open(assets_dir / "inputs.yaml") as f:
        test_inputs = yaml.safe_load(f)

    yield {"test_dir": test_dir, "inputs": test_inputs, "assets_dir": assets_dir}

    # Cleanup env vars
    os.environ.pop("KT_DIRECTORY", None)
    _update_metadata_env_vars(assets_dir, set=False)


@pytest.fixture(scope="class")
def http_client(setup_test_env):
    """Create a test client scoped to each test class.

    Class scope means the server starts once per test class (per parametrized value),
    making tests much faster while still providing isolation between test classes.

    The TestClient context manager ensures lifespan events are called.
    """
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


# ============ Local Execution Tests ============
# These test the ExecutionSupervisor with subprocess isolation


@pytest.mark.parametrize("setup_test_env", load_test_assets(["number", "summer"]), indirect=True)
class TestLocalExecution:
    """Test local execution flow using ExecutionSupervisor with subprocess."""

    @pytest.mark.level("unit")
    def test_health_check(self, http_client):
        response = http_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    @pytest.mark.level("unit")
    def test_supervisor_initialized_on_startup(self, http_client, setup_test_env):
        """Verify supervisor and process pool are created during server startup."""
        # Supervisor is created during lifespan startup (via load_callable)
        # This is the new behavior - subprocess isolation by default
        assert http_server.SUPERVISOR is not None
        assert http_server.SUPERVISOR.process_pool is not None

        # Verify we can make successful calls through the supervisor
        test_inputs = setup_test_env["inputs"]
        name, _, is_class = load_callable_from_test_dir(setup_test_env["test_dir"])

        method = list(test_inputs.keys())[0]
        test_case = test_inputs[method]["valid"][0]
        url = f"/{name.title()}/{method}" if is_class else f"/{method}"

        response = http_client.post(url, json={"args": test_case["args"]})
        assert response.status_code == 200

    @pytest.mark.level("unit")
    def test_valid_calls(self, http_client, setup_test_env):
        test_inputs = setup_test_env["inputs"]
        name, _, is_class = load_callable_from_test_dir(setup_test_env["test_dir"])

        for method, cases in test_inputs.items():
            for test_case in cases.get("valid", []):
                url = f"/{name.title()}/{method}" if is_class else f"/{method}"
                response = http_client.post(
                    url,
                    json={"args": test_case["args"]},
                    headers={"Content-Type": "application/json"},
                )
                assert response.status_code == 200
                assert response.json() == test_case["expected"]

    @pytest.mark.level("unit")
    def test_invalid_calls(self, http_client, setup_test_env):
        test_inputs = setup_test_env["inputs"]
        name, _, is_class = load_callable_from_test_dir(setup_test_env["test_dir"])

        for method, cases in test_inputs.items():
            for test_case in cases.get("invalid", []):
                url = f"/{name.title()}/{method}" if is_class else f"/{method}"
                response = http_client.post(
                    url,
                    json={"args": test_case["args"]},
                    headers={"Content-Type": "application/json"},
                )
                assert response.status_code == 422

    @pytest.mark.level("unit")
    def test_non_existent_callable(self, http_client, setup_test_env):
        response = http_client.post("/non_existent_callable/random_method")
        assert response.status_code == 404
        assert "Callable 'non_existent_callable' not found in metadata configuration" in response.json()["detail"]


# ============ Reload/Redeployment Tests ============
# These tests verify reload behavior and need to run within the same class-scoped server.
# The tests check relative state changes, not absolute values, so they work with shared state.


@pytest.mark.parametrize("setup_test_env", load_test_assets(["summer"]), indirect=True)
class TestReloadBehavior:
    """Test hot reload and redeployment behavior."""

    @pytest.mark.level("unit")
    def test_reload_updates_last_deployed(self, http_client, setup_test_env):
        """Verify _LAST_DEPLOYED is updated when deployed_as_of changes."""
        name, _, is_class = load_callable_from_test_dir(setup_test_env["test_dir"])
        url = f"/{name.title()}/add" if is_class else "/summer"

        # First call - get current state
        first_timestamp = datetime.now(timezone.utc).isoformat()
        response = http_client.post(
            url,
            json={"args": [1, 2]},
            headers={"X-Deployed-As-Of": first_timestamp},
        )
        assert response.status_code == 200

        first_deployed = http_server._LAST_DEPLOYED
        assert first_deployed > 0

        # Wait to ensure different timestamp
        time.sleep(0.1)

        # Second call with newer timestamp - should trigger reload
        second_timestamp = datetime.now(timezone.utc).isoformat()
        response = http_client.post(
            url,
            json={"args": [3, 4]},
            headers={"X-Deployed-As-Of": second_timestamp},
        )
        assert response.status_code == 200
        assert http_server._LAST_DEPLOYED > first_deployed

    @pytest.mark.level("unit")
    def test_no_reload_without_timestamp_change(self, http_client, setup_test_env):
        """Verify no reload happens when deployed_as_of is the same."""
        name, _, is_class = load_callable_from_test_dir(setup_test_env["test_dir"])
        url = f"/{name.title()}/add" if is_class else "/summer"

        # First call with new timestamp
        timestamp = datetime.now(timezone.utc).isoformat()
        response = http_client.post(
            url,
            json={"args": [1, 2]},
            headers={"X-Deployed-As-Of": timestamp},
        )
        assert response.status_code == 200

        after_first_call = http_server._LAST_DEPLOYED

        # Second call with same timestamp - should NOT reload
        response = http_client.post(
            url,
            json={"args": [3, 4]},
            headers={"X-Deployed-As-Of": timestamp},
        )
        assert response.status_code == 200
        # _LAST_DEPLOYED should be unchanged after second call
        assert http_server._LAST_DEPLOYED == after_first_call

    @pytest.mark.level("unit")
    def test_callable_cached_between_calls(self, http_client, setup_test_env):
        """Verify callable is cached and reused between calls."""
        name, _, is_class = load_callable_from_test_dir(setup_test_env["test_dir"])
        url = f"/{name.title()}/add" if is_class else "/summer"

        # Verify callable is cached (should be cached from previous tests or startup)
        callable_name = os.environ["KT_CLS_OR_FN_NAME"]
        assert callable_name in http_server._CACHED_CALLABLES

        # Make a call - should use cached callable
        response = http_client.post(url, json={"args": [1, 2]})
        assert response.status_code == 200

        # Verify still cached
        assert callable_name in http_server._CACHED_CALLABLES


# ============ Distributed Execution Tests ============
# These tests run distributed supervisors with local processes only (no remote workers).
# This tests single-pod distributed scenarios (e.g., 1 pod with 4 GPUs = 4 local processes).


@pytest.mark.parametrize("setup_test_env", load_test_assets(["torch_summer", "torch_ddp"]), indirect=True)
class TestDistributedExecution:
    """Test distributed execution flow using DistributedSupervisor variants."""

    @pytest.mark.level("unit")
    def test_health_check(self, http_client):
        response = http_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    @pytest.mark.level("unit")
    def test_distributed_supervisor_created(self, http_client, setup_test_env):
        """Verify DistributedSupervisor is created for distributed config."""
        from kubetorch.serving.distributed_supervisor import DistributedSupervisor

        test_inputs = setup_test_env["inputs"]
        name, _, is_class = load_callable_from_test_dir(setup_test_env["test_dir"])

        # Make a call to trigger initialization
        method = list(test_inputs.keys())[0]
        test_case = test_inputs[method]["valid"][0]
        url = f"/{name.title()}/{method}" if is_class else f"/{method}"

        http_client.post(url, json={"args": test_case["args"]})

        # Verify it's a distributed supervisor (or subclass)
        assert http_server.SUPERVISOR is not None
        assert isinstance(http_server.SUPERVISOR, DistributedSupervisor)

    @pytest.mark.level("unit")
    def test_valid_calls(self, http_client, setup_test_env):
        test_inputs = setup_test_env["inputs"]
        name, _, is_class = load_callable_from_test_dir(setup_test_env["test_dir"])

        for method, cases in test_inputs.items():
            for test_case in cases.get("valid", []):
                url = f"/{name.title()}/{method}" if is_class else f"/{method}"
                response = http_client.post(
                    url,
                    json={"args": test_case["args"]},
                    headers={"Content-Type": "application/json"},
                )
                assert response.status_code == 200
                assert response.json() == test_case["expected"]

    @pytest.mark.level("unit")
    def test_invalid_calls(self, http_client, setup_test_env):
        test_inputs = setup_test_env["inputs"]
        name, _, is_class = load_callable_from_test_dir(setup_test_env["test_dir"])

        for method, cases in test_inputs.items():
            for test_case in cases.get("invalid", []):
                url = f"/{name.title()}/{method}" if is_class else f"/{method}"
                response = http_client.post(
                    url,
                    json={"args": test_case["args"]},
                    headers={"Content-Type": "application/json"},
                )
                err_type = test_case["error"]
                err_code = test_case["error_code"]
                assert err_type == response.json()["error_type"]
                assert response.json()["message"]
                assert response.json()["traceback"]
                assert response.status_code == err_code

    @pytest.mark.level("unit")
    def test_non_existent_callable(self, http_client, setup_test_env):
        response = http_client.post("/non_existent_callable/random_method")
        assert response.status_code == 404
        assert "Callable 'non_existent_callable' not found in metadata configuration" in response.json()["detail"]


# ============ Supervisor Lifecycle Tests ============


@pytest.mark.parametrize("setup_test_env", load_test_assets(["summer"]), indirect=True)
class TestSupervisorLifecycle:
    """Test supervisor creation, caching, and cleanup."""

    @pytest.mark.level("unit")
    def test_supervisor_cached_by_config_hash(self, http_client, setup_test_env):
        """Verify supervisor is reused when config hash is the same."""
        name, _, is_class = load_callable_from_test_dir(setup_test_env["test_dir"])
        url = f"/{name.title()}/add" if is_class else "/summer"

        # First call creates supervisor
        response = http_client.post(url, json={"args": [1, 2]})
        assert response.status_code == 200

        first_supervisor = http_server.SUPERVISOR
        first_config_hash = first_supervisor.config_hash

        # Second call should reuse same supervisor
        response = http_client.post(url, json={"args": [3, 4]})
        assert response.status_code == 200

        assert http_server.SUPERVISOR is first_supervisor
        assert http_server.SUPERVISOR.config_hash == first_config_hash

    @pytest.mark.level("unit")
    def test_process_pool_has_workers(self, http_client, setup_test_env):
        """Verify process pool is created with workers."""
        name, _, is_class = load_callable_from_test_dir(setup_test_env["test_dir"])
        url = f"/{name.title()}/add" if is_class else "/summer"

        response = http_client.post(url, json={"args": [1, 2]})
        assert response.status_code == 200

        # Verify process pool exists and has processes
        assert http_server.SUPERVISOR.process_pool is not None
        assert len(http_server.SUPERVISOR.process_pool) >= 1


# ============ Error Handling Tests ============


@pytest.mark.parametrize("setup_test_env", load_test_assets(["summer"]), indirect=True)
class TestErrorHandling:
    """Test error handling and exception propagation."""

    @pytest.mark.level("unit")
    def test_method_not_found(self, http_client, setup_test_env):
        """Verify proper error for non-existent method."""
        name, _, is_class = load_callable_from_test_dir(setup_test_env["test_dir"])
        url = f"/{name.title()}/nonexistent_method" if is_class else "/summer"

        if is_class:
            response = http_client.post(url, json={"args": [1, 2]})
            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()

    @pytest.mark.level("unit")
    def test_error_response_format(self, http_client, setup_test_env):
        """Verify error responses have expected structure."""
        test_inputs = setup_test_env["inputs"]
        name, _, is_class = load_callable_from_test_dir(setup_test_env["test_dir"])

        for method, cases in test_inputs.items():
            for test_case in cases.get("invalid", []):
                url = f"/{name.title()}/{method}" if is_class else f"/{method}"
                response = http_client.post(url, json={"args": test_case["args"]})

                # Verify error response structure
                error_json = response.json()
                assert "error_type" in error_json
                assert "message" in error_json
                assert "traceback" in error_json
                assert "pod_name" in error_json
                break  # Just test one error case per method
            break  # Just test first method
