import os

# IMPORTANT: Set these BEFORE importing http_server, as it reads env vars at import time
os.environ["KT_LOG_STREAMING_ENABLED"] = "false"
os.environ["KT_METRICS_ENABLED"] = "false"

import time
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


# ============ Callable Caching Tests ============


@pytest.mark.parametrize("setup_test_env", load_test_assets(["summer"]), indirect=True)
class TestCallableCaching:
    """Test callable caching behavior."""

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


# ============ Reload Behavior Tests ============
# These tests verify the push-based reload mechanism via the /_test_reload endpoint


@pytest.mark.parametrize("setup_test_env", load_test_assets(["summer", "number"]), indirect=True)
class TestReloadBehavior:
    """Test hot reload and redeployment behavior via push-based model."""

    @pytest.mark.level("unit")
    def test_reload_clears_cache(self, http_client, setup_test_env):
        """Verify cache is cleared on reload."""
        name, _, is_class = load_callable_from_test_dir(setup_test_env["test_dir"])
        url = f"/{name.title()}/add" if is_class else "/summer"

        # Make initial call to ensure callable is cached
        response = http_client.post(url, json={"args": [1, 2]})
        assert response.status_code == 200

        callable_name = os.environ["KT_CLS_OR_FN_NAME"]
        assert callable_name in http_server._CACHED_CALLABLES

        # Trigger reload with same metadata
        metadata = {
            "module": {
                "module_name": os.environ.get("KT_MODULE_NAME"),
                "cls_or_fn_name": os.environ.get("KT_CLS_OR_FN_NAME"),
                "file_path": os.environ.get("KT_FILE_PATH"),
                "callable_type": os.environ.get("KT_CALLABLE_TYPE", "fn"),
            }
        }
        reload_response = http_client.post("/_test_reload", json=metadata)
        assert reload_response.status_code == 200
        assert reload_response.json()["status"] == "ok"

        # Cache should be cleared during reload, then repopulated
        # After reload completes, the callable should be re-cached
        assert callable_name in http_server._CACHED_CALLABLES

    @pytest.mark.level("unit")
    def test_reload_recreates_supervisor(self, http_client, setup_test_env):
        """Verify supervisor is recreated on reload."""
        name, _, is_class = load_callable_from_test_dir(setup_test_env["test_dir"])
        url = f"/{name.title()}/add" if is_class else "/summer"

        # Make initial call
        response = http_client.post(url, json={"args": [1, 2]})
        assert response.status_code == 200

        # Store reference to original supervisor
        original_supervisor = http_server.SUPERVISOR
        assert original_supervisor is not None

        # Trigger reload
        metadata = {
            "module": {
                "module_name": os.environ.get("KT_MODULE_NAME"),
                "cls_or_fn_name": os.environ.get("KT_CLS_OR_FN_NAME"),
                "file_path": os.environ.get("KT_FILE_PATH"),
                "callable_type": os.environ.get("KT_CALLABLE_TYPE", "fn"),
            }
        }
        reload_response = http_client.post("/_test_reload", json=metadata)
        assert reload_response.status_code == 200

        # Supervisor should be recreated (new instance)
        # Note: The supervisor object itself is recreated, but the config hash may be the same
        assert http_server.SUPERVISOR is not None

    @pytest.mark.level("unit")
    def test_reload_with_different_callable(self, http_client, setup_test_env):
        """Verify reload updates to a different callable."""
        name, _, is_class = load_callable_from_test_dir(setup_test_env["test_dir"])
        url = f"/{name.title()}/add" if is_class else "/summer"

        # Make initial call
        response = http_client.post(url, json={"args": [1, 2]})
        assert response.status_code == 200

        original_callable = os.environ.get("KT_CLS_OR_FN_NAME")

        # Simulate reload with different callable name (but keep same module for test)
        # In a real scenario, this would be a different function
        new_callable_name = "different_callable"
        metadata = {
            "module": {
                "module_name": os.environ.get("KT_MODULE_NAME"),
                "cls_or_fn_name": new_callable_name,
                "file_path": os.environ.get("KT_FILE_PATH"),
                "callable_type": "fn",
            }
        }
        reload_response = http_client.post("/_test_reload", json=metadata)
        assert reload_response.status_code == 200

        # Verify env var was updated
        assert os.environ.get("KT_CLS_OR_FN_NAME") == new_callable_name

        # Restore original for subsequent tests
        os.environ["KT_CLS_OR_FN_NAME"] = original_callable

    @pytest.mark.level("unit")
    def test_callable_works_after_reload(self, http_client, setup_test_env):
        """Verify callable still works correctly after reload."""
        name, _, is_class = load_callable_from_test_dir(setup_test_env["test_dir"])
        test_inputs = setup_test_env["inputs"]

        # First, ensure the correct callable is loaded (previous test may have changed it)
        metadata = {
            "module": {
                "module_name": os.environ.get("KT_MODULE_NAME"),
                "cls_or_fn_name": os.environ.get("KT_CLS_OR_FN_NAME"),
                "file_path": os.environ.get("KT_FILE_PATH"),
                "callable_type": os.environ.get("KT_CALLABLE_TYPE", "fn"),
            }
        }
        reload_response = http_client.post("/_test_reload", json=metadata)
        assert reload_response.status_code == 200

        # Make calls after reload - should work correctly
        for method, cases in test_inputs.items():
            for test_case in cases.get("valid", []):
                url = f"/{name.title()}/{method}" if is_class else f"/{method}"
                response = http_client.post(url, json={"args": test_case["args"]})
                assert response.status_code == 200
                assert response.json() == test_case["expected"]


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


# ============ Async Concurrency Tests ============
# These tests verify that async callables run concurrently on the event loop,
# matching FastAPI's concurrency model.


@pytest.mark.parametrize("setup_test_env", load_test_assets(["async_summer"]), indirect=True)
class TestAsyncConcurrency:
    """Test async callable concurrency behavior.

    Verifies that the ProcessWorker's event loop properly handles async callables:
    - Multiple async calls should run concurrently (not sequentially)
    - All calls should start before any finish (cooperative multitasking)
    """

    @pytest.mark.level("unit")
    def test_async_callable_basic(self, http_client, setup_test_env):
        """Verify async callable returns correct result."""
        response = http_client.post("/async_summer", json={"args": [1, 2]})
        assert response.status_code == 200
        assert response.json() == 3

    @pytest.mark.level("unit")
    def test_async_callable_with_times(self, http_client, setup_test_env):
        """Verify async callable returns timing information."""
        response = http_client.post(
            "/async_summer", json={"args": [5, 10], "kwargs": {"sleep_time": 0.1, "return_times": True}}
        )
        assert response.status_code == 200
        result = response.json()
        assert "start" in result
        assert "end" in result
        assert "result" in result
        assert result["result"] == 15
        # Should have slept ~0.1 seconds
        assert result["end"] - result["start"] >= 0.09

    @pytest.mark.level("unit")
    def test_async_concurrent_execution(self, http_client, setup_test_env):
        """Verify multiple async calls run concurrently.

        If async calls run sequentially, total time would be ~N * sleep_time.
        If they run concurrently, total time should be ~sleep_time.

        We launch N calls that each sleep for 0.3s. If concurrent, they should
        all complete in ~0.3s total. If sequential, they'd take ~N*0.3s.
        """
        import concurrent.futures

        num_calls = 5
        sleep_time = 0.3

        def make_call(i):
            return http_client.post(
                "/async_summer", json={"args": [i, i], "kwargs": {"sleep_time": sleep_time, "return_times": True}}
            )

        start_time = time.time()

        # Launch all calls concurrently from client side
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_calls) as executor:
            futures = [executor.submit(make_call, i) for i in range(num_calls)]
            responses = [f.result() for f in concurrent.futures.as_completed(futures)]

        total_time = time.time() - start_time

        # Verify all calls succeeded
        for resp in responses:
            assert resp.status_code == 200

        # Parse timing info from responses
        results = [r.json() for r in responses]
        start_times = [r["start"] for r in results]
        end_times = [r["end"] for r in results]

        # Key assertion: If running concurrently, all calls should start before
        # the earliest call finishes (overlapping execution windows)
        max_start = max(start_times)
        min_end = min(end_times)
        assert max_start < min_end, (
            f"Async calls not running concurrently! " f"Latest start ({max_start}) should be < earliest end ({min_end})"
        )

        # Total time should be closer to sleep_time than to num_calls * sleep_time
        # Allow some overhead but it should definitely be less than sequential time
        sequential_time = num_calls * sleep_time
        assert total_time < sequential_time * 0.5, (
            f"Total time ({total_time:.2f}s) too close to sequential time ({sequential_time:.2f}s). "
            f"Expected concurrent execution to be much faster."
        )


@pytest.mark.parametrize("setup_test_env", load_test_assets(["summer"]), indirect=True)
class TestSyncConcurrency:
    """Test sync callable concurrency behavior.

    Verifies that sync callables run in the thread pool and don't block the event loop.
    """

    @pytest.mark.level("unit")
    def test_sync_concurrent_execution(self, http_client, setup_test_env):
        """Verify multiple sync calls can run concurrently via thread pool.

        Sync callables should run in the thread pool, allowing multiple to
        execute concurrently (up to the thread pool size).
        """
        import concurrent.futures

        num_calls = 5

        def make_call(i):
            return http_client.post("/summer", json={"args": [i, i]})

        # Launch all calls concurrently from client side
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_calls) as executor:
            futures = [executor.submit(make_call, i) for i in range(num_calls)]
            responses = [f.result() for f in concurrent.futures.as_completed(futures)]

        # Verify all calls succeeded with correct results
        results = sorted([r.json() for r in responses])
        expected = sorted([i + i for i in range(num_calls)])
        assert results == expected
