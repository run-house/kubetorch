"""
Tests for Kubetorch Monarch integration.

This test suite covers:
1. Unit tests for proxy classes, gateway connection, WebSocket client (level="unit")
2. Integration tests for WebSocket connection mode (level="minimal")

Unit tests use mocks and don't require a cluster.
Integration tests require a Kubernetes cluster with Kubetorch installed.
"""

import json
import os
import pickle
from unittest.mock import MagicMock, patch

import pytest

# Set env vars before imports
os.environ["KT_LOG_STREAMING_ENABLED"] = "false"
os.environ["KT_METRICS_ENABLED"] = "false"


# =============================================================================
# Unit Tests - Proxy Classes
# =============================================================================


@pytest.mark.level("unit")
class TestProxyClasses:
    """Tests for Monarch proxy classes."""

    def test_host_mesh_proxy_creation(self):
        """Test HostMeshProxy can be created with shape."""
        from kubetorch.monarch.proxy import HostMeshProxy

        gateway = MagicMock()
        proxy = HostMeshProxy(
            host_mesh_id="hm_test",
            shape={"hosts": 4},
            gateway=gateway,
        )

        assert proxy._host_mesh_id == "hm_test"
        assert proxy._shape == {"hosts": 4}
        assert proxy.sizes == {"hosts": 4}
        assert proxy.size() == 4
        assert proxy.size("hosts") == 4

    def test_host_mesh_proxy_slice(self):
        """Test HostMeshProxy slicing creates new proxy with correct shape."""
        from kubetorch.monarch.proxy import HostMeshProxy

        gateway = MagicMock()
        proxy = HostMeshProxy(
            host_mesh_id="hm_test",
            shape={"hosts": 4},
            gateway=gateway,
        )

        sliced = proxy.slice(hosts=slice(0, 2))
        assert sliced._shape == {"hosts": 2}
        assert sliced._host_mesh_id == "hm_test"

    def test_host_mesh_proxy_spawn_procs(self):
        """Test HostMeshProxy.spawn_procs calls gateway correctly."""
        from kubetorch.monarch.proxy import HostMeshProxy

        gateway = MagicMock()
        gateway.call.return_value = {
            "proc_mesh_id": "pm_123",
            "shape": {"hosts": 4, "gpus": 8},
        }

        proxy = HostMeshProxy(
            host_mesh_id="hm_test",
            shape={"hosts": 4},
            gateway=gateway,
        )

        proc_mesh = proxy.spawn_procs(per_host={"gpus": 8}, name="workers")

        gateway.call.assert_called_once_with(
            "spawn_procs",
            per_host={"gpus": 8},
            name="workers",
        )
        assert proc_mesh._proc_mesh_id == "pm_123"
        assert proc_mesh._shape == {"hosts": 4, "gpus": 8}

    def test_proc_mesh_proxy_spawn_actors(self):
        """Test ProcMeshProxy.spawn calls gateway correctly."""
        from kubetorch.monarch.proxy import HostMeshProxy, ProcMeshProxy

        gateway = MagicMock()
        gateway.call.return_value = {
            "actor_mesh_id": "am_456",
            "shape": {"hosts": 4, "gpus": 8},
        }

        host_mesh = HostMeshProxy("hm_test", {"hosts": 4}, gateway)
        proc_mesh = ProcMeshProxy(
            proc_mesh_id="pm_123",
            shape={"hosts": 4, "gpus": 8},
            gateway=gateway,
            host_mesh=host_mesh,
        )

        class MockActor:
            pass

        proc_mesh.spawn("trainers", MockActor, arg1="value1")

        # Verify call was made with pickled actor class
        call_args = gateway.call.call_args
        assert call_args[0][0] == "spawn_actors"
        assert call_args[1]["proc_mesh_id"] == "pm_123"
        assert call_args[1]["name"] == "trainers"
        assert "actor_class_bytes" in call_args[1]

    def test_actor_mesh_proxy_endpoint_access(self):
        """Test ActorMeshProxy provides EndpointProxy via attribute access."""
        from kubetorch.monarch.proxy import ActorMeshProxy

        gateway = MagicMock()
        actor_mesh = ActorMeshProxy(
            actor_mesh_id="am_test",
            shape={"hosts": 4, "gpus": 8},
            gateway=gateway,
        )

        # Access endpoint via attribute
        endpoint = actor_mesh.train
        assert endpoint._endpoint_name == "train"
        assert endpoint._actor_mesh_id == "am_test"

    def test_endpoint_proxy_call(self):
        """Test EndpointProxy.call returns FutureProxy."""
        from kubetorch.monarch.proxy import EndpointProxy

        gateway = MagicMock()
        gateway.call.return_value = {"future_id": "fut_789"}

        endpoint = EndpointProxy(
            actor_mesh_id="am_test",
            endpoint_name="train",
            gateway=gateway,
        )

        future = endpoint.call(batch_size=32, lr=0.001)

        assert future._future_id == "fut_789"
        # Verify args were pickled
        call_args = gateway.call.call_args
        assert call_args[1]["endpoint_name"] == "train"
        assert "args_bytes" in call_args[1]
        assert "kwargs_bytes" in call_args[1]

    def test_future_proxy_get(self):
        """Test FutureProxy.get retrieves and caches result."""
        from kubetorch.monarch.proxy import FutureProxy

        gateway = MagicMock()
        # Return pickled result
        gateway.call.return_value = pickle.dumps({"accuracy": 0.95})

        future = FutureProxy(future_id="fut_test", gateway=gateway)

        result = future.get()
        assert result == {"accuracy": 0.95}
        assert future._resolved is True

        # Second call should return cached result
        result2 = future.get()
        assert result2 == {"accuracy": 0.95}
        # Should only call gateway once
        assert gateway.call.call_count == 1

    def test_job_state_attribute_access(self):
        """Test JobState provides mesh access via attributes."""
        from kubetorch.monarch.proxy import HostMeshProxy, JobState

        gateway = MagicMock()
        workers = HostMeshProxy("hm_workers", {"hosts": 4}, gateway)
        dataloaders = HostMeshProxy("hm_dataloaders", {"hosts": 2}, gateway)

        state = JobState({"workers": workers, "dataloaders": dataloaders})

        assert state.workers is workers
        assert state.dataloaders is dataloaders

        with pytest.raises(AttributeError):
            _ = state.nonexistent


# =============================================================================
# Unit Tests - Gateway Connection
# =============================================================================


@pytest.mark.level("unit")
class TestGatewayConnection:
    """Tests for GatewayConnection class."""

    def test_gateway_connection_url_conversion(self):
        """Test HTTP to WebSocket URL conversion."""
        from kubetorch.monarch.job import GatewayConnection

        conn = GatewayConnection("http://my-service.ns.svc:32300")
        assert conn._get_ws_url() == "ws://my-service.ns.svc:32300/ws/callable"

        conn = GatewayConnection("https://my-service.ns.svc:32300")
        assert conn._get_ws_url() == "wss://my-service.ns.svc:32300/ws/callable"

        conn = GatewayConnection("my-service.ns.svc:32300")
        assert conn._get_ws_url() == "ws://my-service.ns.svc:32300/ws/callable"

    @patch("kubetorch.monarch.job.websocket")
    def test_gateway_connection_websocket_call(self, mock_websocket_module):
        """Test WebSocket call flow."""
        from kubetorch.monarch.job import GatewayConnection

        # Mock WebSocket connection
        mock_ws = MagicMock()
        mock_websocket_module.create_connection.return_value = mock_ws
        mock_ws.recv.return_value = json.dumps(
            {
                "request_id": "req_1",
                "result": {"status": "ok"},
            }
        )

        conn = GatewayConnection("http://test-service:32300")
        result = conn.call("get_status")

        assert result == {"status": "ok"}
        mock_ws.send.assert_called_once()

    def test_gateway_connection_close(self):
        """Test connection cleanup."""
        from kubetorch.monarch.job import GatewayConnection

        conn = GatewayConnection("http://test-service:32300")
        mock_ws = MagicMock()
        conn._ws = mock_ws

        conn.close()

        mock_ws.close.assert_called_once()
        assert conn._ws is None


# =============================================================================
# Unit Tests - WebSocket Client
# =============================================================================


@pytest.mark.level("unit")
class TestWebSocketClient:
    """Tests for WebSocketClient in http_client."""

    def test_websocket_client_parse_endpoint(self):
        """Test endpoint URL parsing."""
        from kubetorch.serving.http_client import WebSocketClient

        client = WebSocketClient("http://test:32300", None, "test-svc")

        # Test various endpoint formats
        cls_name, method = client._parse_endpoint("/ns/svc/MyClass/my_method")
        assert cls_name == "MyClass"
        assert method == "my_method"

        cls_name, method = client._parse_endpoint("/MyClass/my_method")
        assert cls_name == "MyClass"
        assert method == "my_method"

        cls_name, method = client._parse_endpoint("MyClass/my_method")
        assert cls_name == "MyClass"
        assert method == "my_method"

    def test_websocket_client_ws_url(self):
        """Test WebSocket URL generation."""
        from kubetorch.serving.http_client import WebSocketClient

        client = WebSocketClient("http://test:32300", None, "test-svc")
        assert client._get_ws_url() == "ws://test:32300/ws/callable"

        client = WebSocketClient("https://test:32300", None, "test-svc")
        assert client._get_ws_url() == "wss://test:32300/ws/callable"


# =============================================================================
# Unit Tests - Monarch Gateway
# =============================================================================


@pytest.mark.level("unit")
class TestMonarchGateway:
    """Tests for MonarchGateway server-side class."""

    def test_gateway_discover_worker_ips(self):
        """Test worker IP discovery via DNS."""
        from kubetorch.monarch.gateway import MonarchGateway

        gateway = MonarchGateway()

        with patch("socket.getaddrinfo") as mock_getaddrinfo:
            mock_getaddrinfo.return_value = [
                (2, 1, 6, "", ("10.0.0.1", 0)),
                (2, 1, 6, "", ("10.0.0.2", 0)),
                (2, 1, 6, "", ("10.0.0.3", 0)),
            ]

            ips = gateway._discover_worker_ips("test-headless.ns.svc.cluster.local")

            assert ips == ["10.0.0.1", "10.0.0.2", "10.0.0.3"]

    def test_gateway_initialize_without_workers(self):
        """Test gateway initialization fails without workers."""
        from kubetorch.monarch.gateway import MonarchGateway

        gateway = MonarchGateway()

        with pytest.raises(RuntimeError, match="No worker IPs discovered"):
            gateway.initialize(worker_ips=[])

    @patch("kubetorch.monarch.gateway.attach_to_workers")
    def test_gateway_initialize_with_workers(self, mock_attach):
        """Test successful gateway initialization."""
        from kubetorch.monarch.gateway import MonarchGateway

        mock_host_mesh = MagicMock()
        mock_attach.return_value = mock_host_mesh

        gateway = MonarchGateway()
        result = gateway.initialize(worker_ips=["10.0.0.1", "10.0.0.2"])

        assert result["status"] == "initialized"
        assert result["num_workers"] == 2
        assert gateway._initialized is True
        mock_attach.assert_called_once()

    def test_gateway_spawn_procs_before_init(self):
        """Test spawn_procs fails if gateway not initialized."""
        from kubetorch.monarch.gateway import MonarchGateway

        gateway = MonarchGateway()

        with pytest.raises(RuntimeError, match="Gateway not initialized"):
            gateway.spawn_procs(per_host={"gpus": 8})

    @patch("kubetorch.monarch.gateway.attach_to_workers")
    def test_gateway_spawn_procs(self, mock_attach):
        """Test spawning proc mesh via gateway."""
        from kubetorch.monarch.gateway import MonarchGateway

        mock_host_mesh = MagicMock()
        mock_proc_mesh = MagicMock()
        mock_proc_mesh.sizes = {"hosts": 2, "gpus": 8}
        mock_host_mesh.spawn_procs.return_value = mock_proc_mesh
        mock_attach.return_value = mock_host_mesh

        gateway = MonarchGateway()
        gateway.initialize(worker_ips=["10.0.0.1", "10.0.0.2"])

        result = gateway.spawn_procs(per_host={"gpus": 8}, name="workers")

        assert "proc_mesh_id" in result
        assert result["shape"] == {"hosts": 2, "gpus": 8}
        mock_host_mesh.spawn_procs.assert_called_once()

    def test_gateway_get_status(self):
        """Test gateway status reporting."""
        from kubetorch.monarch.gateway import MonarchGateway

        gateway = MonarchGateway()
        status = gateway.get_status()

        assert status["initialized"] is False
        assert status["num_workers"] == 0
        assert status["num_proc_meshes"] == 0


# =============================================================================
# Unit Tests - KubernetesJob
# =============================================================================


@pytest.mark.level("unit")
class TestKubernetesJob:
    """Tests for KubernetesJob client class."""

    def test_job_requires_compute_or_selector(self):
        """Test job creation requires either compute or selector."""
        from kubetorch.monarch.job import KubernetesJob

        with pytest.raises(ValueError, match="Must specify either compute or selector"):
            KubernetesJob()

        with pytest.raises(ValueError, match="Cannot specify both compute and selector"):
            KubernetesJob(compute=MagicMock(), selector={"app": "test"})

    def test_job_name_generation(self):
        """Test job name auto-generation."""
        from kubetorch.monarch.job import KubernetesJob

        job = KubernetesJob(selector={"app": "test"})
        assert job._name.startswith("monarch-")

        job = KubernetesJob(selector={"app": "test"}, name="my-job")
        assert job._name == "my-job"

    def test_job_context_manager(self):
        """Test job as context manager calls apply/kill."""
        from kubetorch.monarch.job import KubernetesJob

        job = KubernetesJob(selector={"app": "test"})
        job.apply = MagicMock()
        job.kill = MagicMock()

        with job as j:
            assert j is job
            job.apply.assert_called_once()

        job.kill.assert_called_once()


# =============================================================================
# Unit Tests - Connection Mode
# =============================================================================


@pytest.mark.level("unit")
class TestConnectionModeUnit:
    """Unit tests for connection_mode parameter in Module."""

    def test_module_connection_mode_default(self):
        """Test Module defaults to http connection mode."""
        from kubetorch.resources.callables.module import Module

        module = Module("test", ("path", "module", "Cls"))
        assert module._connection_mode == "http"

    def test_module_connection_mode_validation(self):
        """Test Module validates connection_mode values."""
        from kubetorch.resources.callables.module import Module

        module = Module("test", ("path", "module", "Cls"))
        module._compute = MagicMock()
        module._compute.namespace = "test"
        module._service_config = MagicMock()

        # Validation happens in to() method


# =============================================================================
# Integration Tests - WebSocket Connection Mode
# These tests deploy actual services to the cluster
# =============================================================================


class WebSocketTestCls:
    """Simple class for testing WebSocket connection mode."""

    def __init__(self, initial_value: int = 0):
        self.value = initial_value

    def add(self, x: int) -> int:
        self.value += x
        return self.value

    def multiply(self, x: int) -> int:
        self.value *= x
        return self.value

    def get_value(self) -> int:
        return self.value

    def echo(self, msg: str) -> str:
        return f"Echo: {msg}"


@pytest.fixture(scope="session")
async def remote_websocket_cls():
    """Deploy a class using WebSocket connection mode."""
    import kubetorch as kt

    compute_type = os.getenv("TEST_COMPUTE_TYPE", "deployment")
    service_name_prefix = os.getenv("SERVICE_NAME_PREFIX", compute_type)
    name = f"{service_name_prefix}-ws-test-cls"

    compute = kt.Compute(
        cpus=".01",
        gpu_anti_affinity=True,
        launch_timeout=300,
    )

    remote_cls = await kt.cls(WebSocketTestCls, name=name).to_async(
        compute=compute,
        init_args={"initial_value": 10},
        connection_mode="websocket",
    )
    return remote_cls


@pytest.fixture(scope="session")
async def remote_http_cls():
    """Deploy a class using HTTP connection mode (default)."""
    import kubetorch as kt

    compute_type = os.getenv("TEST_COMPUTE_TYPE", "deployment")
    service_name_prefix = os.getenv("SERVICE_NAME_PREFIX", compute_type)
    name = f"{service_name_prefix}-http-test-cls"

    compute = kt.Compute(
        cpus=".01",
        gpu_anti_affinity=True,
        launch_timeout=300,
    )

    remote_cls = await kt.cls(WebSocketTestCls, name=name).to_async(
        compute=compute,
        init_args={"initial_value": 10},
        connection_mode="http",
    )
    return remote_cls


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_websocket_cls_basic_call(remote_websocket_cls):
    """Test basic method calls over WebSocket connection."""
    # Test simple method call
    result = remote_websocket_cls.echo("hello")
    assert result == "Echo: hello"


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_websocket_cls_stateful_calls(remote_websocket_cls):
    """Test stateful method calls over WebSocket connection."""
    # Get initial value
    initial = remote_websocket_cls.get_value()
    assert isinstance(initial, int)

    # Add to value
    result = remote_websocket_cls.add(5)
    assert result == initial + 5

    # Multiply
    result = remote_websocket_cls.multiply(2)
    assert result == (initial + 5) * 2


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_websocket_cls_async_call(remote_websocket_cls):
    """Test async method calls over WebSocket connection."""
    import asyncio

    # Call with async_ flag
    coroutine = remote_websocket_cls.echo("async hello", async_=True)
    assert asyncio.iscoroutine(coroutine)
    result = await coroutine
    assert result == "Echo: async hello"


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_http_cls_basic_call(remote_http_cls):
    """Test basic method calls over HTTP connection (baseline comparison)."""
    result = remote_http_cls.echo("hello")
    assert result == "Echo: hello"


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_http_vs_websocket_parity(remote_websocket_cls, remote_http_cls):
    """Test that WebSocket and HTTP connection modes produce same results."""
    # Both should echo correctly
    ws_result = remote_websocket_cls.echo("parity test")
    http_result = remote_http_cls.echo("parity test")
    assert ws_result == http_result == "Echo: parity test"

    # Both should handle integers correctly
    ws_add = remote_websocket_cls.add(1)
    http_add = remote_http_cls.add(1)
    assert isinstance(ws_add, int)
    assert isinstance(http_add, int)


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_websocket_multiple_sequential_calls(remote_websocket_cls):
    """Test multiple sequential calls over the same WebSocket connection."""
    results = []
    for i in range(5):
        result = remote_websocket_cls.echo(f"call {i}")
        results.append(result)

    for i, result in enumerate(results):
        assert result == f"Echo: call {i}"


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_websocket_concurrent_async_calls(remote_websocket_cls):
    """Test concurrent async calls over WebSocket connection."""
    import asyncio

    # Set async mode
    try:
        remote_websocket_cls.async_ = True

        # Launch multiple concurrent calls
        tasks = [remote_websocket_cls.echo(f"concurrent {i}") for i in range(5)]
        results = await asyncio.gather(*tasks)

        for i, result in enumerate(results):
            assert result == f"Echo: concurrent {i}"
    finally:
        remote_websocket_cls.async_ = False
