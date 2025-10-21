import os
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from kubetorch.servers.http.http_server import app

from .utils import _update_metadata_env_vars, load_callable_from_test_dir


def load_test_assets(names):
    """Get all test directories from the assets directory"""
    assets_dir = Path(__file__).parent / "assets"
    return [str(d) for d in assets_dir.glob("*") if d.is_dir() and d.name in names]


@pytest.fixture
def setup_kt_directory(test_dir):
    assets_dir = Path(__file__).parent / "assets" / test_dir
    os.environ["KT_DIRECTORY"] = str(assets_dir)
    with open(assets_dir / "inputs.yaml") as f:
        test_inputs = yaml.safe_load(f)

    _update_metadata_env_vars(assets_dir, set=True)
    # Set the other Kubernetes downward API env vars and the LOCAL_IPS env var for distributed testing
    os.environ["POD_NAMESPACE"] = "kubetorch"
    os.environ["POD_NAME"] = "kubetorch-test"
    os.environ["POD_IP"] = "localhost"
    os.environ["LOCAL_IPS"] = "localhost"
    os.environ["KT_SERVICE_NAME"] = test_dir
    yield test_inputs

    # Clean up the DISTRIBUTED_SUPERVISOR and cached callables if they exist
    from kubetorch.servers.http import http_server

    if http_server.DISTRIBUTED_SUPERVISOR:
        http_server.DISTRIBUTED_SUPERVISOR.cleanup()
        http_server.DISTRIBUTED_SUPERVISOR = None
    # Also clear the cached callables to override reload with new config
    http_server._CACHED_CALLABLES.clear()

    os.environ.pop("KT_DIRECTORY", None)
    _update_metadata_env_vars(assets_dir, set=False)


@pytest.fixture(scope="class")
def http_client():
    # Use TestClient as a context manager to ensure lifespan events are called (setup and cleanup)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


# We run these tests for each callable defined in the assets folder
# NOTE: http_client must be passed as fixture after setup_kt_directory so the env vars are set when the
# server is started.
@pytest.mark.parametrize("test_dir", load_test_assets(["number", "summer"]))
class TestHTTPServer:
    @pytest.mark.level("unit")
    def test_health_check(self, setup_kt_directory, test_dir, http_client):
        response = http_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    @pytest.mark.level("unit")
    def test_valid_calls(self, setup_kt_directory, test_dir, http_client):
        test_inputs = setup_kt_directory
        name, callable_obj, is_class = load_callable_from_test_dir(test_dir)

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
    def test_invalid_calls(self, setup_kt_directory, test_dir, http_client):
        test_inputs = setup_kt_directory
        name, callable_obj, is_class = load_callable_from_test_dir(test_dir)

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
    def test_non_existent_callable(self, setup_kt_directory, test_dir, http_client):
        response = http_client.post("/non_existent_callable/random_method")
        assert response.status_code == 404
        assert (
            "Callable 'non_existent_callable' not found in metadata configuration"
            in response.json()["detail"]
        )


@pytest.mark.parametrize("test_dir", load_test_assets(["torch_summer", "torch_ddp"]))
class TestDistributedHTTPServer:
    @pytest.mark.level("unit")
    def test_health_check(self, setup_kt_directory, test_dir, http_client):
        response = http_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    @pytest.mark.level("unit")
    def test_valid_calls(self, setup_kt_directory, test_dir, http_client):
        test_inputs = setup_kt_directory
        name, callable_obj, is_class = load_callable_from_test_dir(test_dir)

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
    def test_invalid_calls(self, setup_kt_directory, test_dir, http_client):
        test_inputs = setup_kt_directory
        name, callable_obj, is_class = load_callable_from_test_dir(test_dir)

        for method, cases in test_inputs.items():
            for test_case in cases.get("invalid", []):
                url = f"/{name.title()}/{method}" if is_class else f"/{method}"
                response = http_client.post(
                    url,
                    json={"args": test_case["args"]},
                    headers={"Content-Type": "application/json"},
                )
                print(response.json())
                err_type = test_case["error"]
                err_code = test_case["error_code"]
                assert err_type == response.json()["error_type"]
                assert response.json()["message"]
                assert response.json()["traceback"]
                assert response.status_code == err_code

    @pytest.mark.level("unit")
    def test_non_existent_callable(self, setup_kt_directory, test_dir, http_client):
        response = http_client.post("/non_existent_callable/random_method")
        assert response.status_code == 404
        assert (
            "Callable 'non_existent_callable' not found in metadata configuration"
            in response.json()["detail"]
        )
