"""Tests for pool and apply routes.

To run: `python -m pytest tests/test_routes.py`
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kubernetes.client.rest import ApiException


# =============================================================================
# Apply Route Tests (/controller/apply)
# =============================================================================


class TestApplyRoute:
    """Tests for the /controller/apply endpoint."""

    def test_apply_deployment_creates_resource(self, client, mock_k8s_clients):
        """Test applying a new deployment creates it in K8s."""
        apps, core, custom = mock_k8s_clients

        body = {
            "service_name": "my-deployment",
            "namespace": "default",
            "resource_type": "deployment",
            "resource_manifest": {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {"name": "my-deployment"},
                "spec": {"replicas": 1},
            },
        }

        resp = client.post("/controller/apply", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["message"] == "Resource applied"
        assert data["service_name"] == "my-deployment"
        assert data["resource_type"] == "deployment"

        apps.create_namespaced_deployment.assert_called_once()

    def test_apply_knative_creates_custom_resource(self, client, mock_k8s_clients):
        """Test applying a Knative service creates custom resource."""
        apps, core, custom = mock_k8s_clients

        body = {
            "service_name": "my-knative-svc",
            "namespace": "default",
            "resource_type": "knative",
            "resource_manifest": {
                "apiVersion": "serving.knative.dev/v1",
                "kind": "Service",
                "metadata": {"name": "my-knative-svc"},
            },
        }

        resp = client.post("/controller/apply", json=body)
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

        custom.create_namespaced_custom_object.assert_called_once()
        call_kwargs = custom.create_namespaced_custom_object.call_args
        assert call_kwargs.kwargs["group"] == "serving.knative.dev"
        assert call_kwargs.kwargs["plural"] == "services"

    def test_apply_raycluster_creates_custom_resource(self, client, mock_k8s_clients):
        """Test applying a RayCluster creates custom resource."""
        apps, core, custom = mock_k8s_clients

        body = {
            "service_name": "my-ray-cluster",
            "namespace": "default",
            "resource_type": "raycluster",
            "resource_manifest": {
                "apiVersion": "ray.io/v1",
                "kind": "RayCluster",
                "metadata": {"name": "my-ray-cluster"},
            },
        }

        resp = client.post("/controller/apply", json=body)
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

        custom.create_namespaced_custom_object.assert_called_once()
        call_kwargs = custom.create_namespaced_custom_object.call_args
        assert call_kwargs.kwargs["group"] == "ray.io"
        assert call_kwargs.kwargs["plural"] == "rayclusters"

    def test_apply_pytorchjob_creates_custom_resource(self, client, mock_k8s_clients):
        """Test applying a PyTorchJob creates custom resource."""
        apps, core, custom = mock_k8s_clients

        body = {
            "service_name": "my-pytorch-job",
            "namespace": "default",
            "resource_type": "pytorchjob",
            "resource_manifest": {
                "apiVersion": "kubeflow.org/v1",
                "kind": "PyTorchJob",
                "metadata": {"name": "my-pytorch-job"},
            },
        }

        resp = client.post("/controller/apply", json=body)
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

        custom.create_namespaced_custom_object.assert_called_once()
        call_kwargs = custom.create_namespaced_custom_object.call_args
        assert call_kwargs.kwargs["group"] == "kubeflow.org"
        assert call_kwargs.kwargs["plural"] == "pytorchjobs"

    def test_apply_service_creates_k8s_service(self, client, mock_k8s_clients):
        """Test applying a Service creates K8s service."""
        apps, core, custom = mock_k8s_clients

        body = {
            "service_name": "my-svc",
            "namespace": "default",
            "resource_type": "service",
            "resource_manifest": {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {"name": "my-svc"},
                "spec": {"ports": [{"port": 80}]},
            },
        }

        resp = client.post("/controller/apply", json=body)
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

        core.create_namespaced_service.assert_called_once()

    def test_apply_conflict_updates_existing_resource(self, client, mock_k8s_clients):
        """Test that 409 conflict triggers an update instead of create."""
        apps, core, custom = mock_k8s_clients

        # First create raises 409 (already exists)
        apps.create_namespaced_deployment.side_effect = ApiException(
            status=409, reason="AlreadyExists"
        )

        # Mock the read and replace for update
        mock_existing = MagicMock()
        mock_existing.metadata.resource_version = "12345"
        mock_existing.spec.selector.match_labels = {"app": "test"}
        mock_existing.spec.selector.match_expressions = None
        apps.read_namespaced_deployment.return_value = mock_existing
        apps.replace_namespaced_deployment.return_value = mock_existing

        body = {
            "service_name": "existing-deployment",
            "namespace": "default",
            "resource_type": "deployment",
            "resource_manifest": {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {"name": "existing-deployment"},
                "spec": {"replicas": 2},
            },
        }

        resp = client.post("/controller/apply", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["message"] == "Resource updated"

        apps.replace_namespaced_deployment.assert_called_once()

    def test_apply_unknown_resource_type_returns_error(self, client, mock_k8s_clients):
        """Test that unknown resource type with bad apiVersion returns error."""
        apps, core, custom = mock_k8s_clients

        body = {
            "service_name": "unknown",
            "namespace": "default",
            "resource_type": "unknown",
            "resource_manifest": {
                "apiVersion": "v1",  # No group prefix
                "kind": "Unknown",
                "metadata": {"name": "unknown"},
            },
        }

        resp = client.post("/controller/apply", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"
        assert "Unknown resource type" in data["message"]

    def test_apply_generic_crd(self, client, mock_k8s_clients):
        """Test applying a generic CRD parses apiVersion correctly."""
        apps, core, custom = mock_k8s_clients

        body = {
            "service_name": "my-custom-resource",
            "namespace": "default",
            "resource_type": "customresource",
            "resource_manifest": {
                "apiVersion": "example.com/v1beta1",
                "kind": "CustomResource",
                "metadata": {"name": "my-custom-resource"},
            },
        }

        resp = client.post("/controller/apply", json=body)
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

        custom.create_namespaced_custom_object.assert_called_once()
        call_kwargs = custom.create_namespaced_custom_object.call_args
        assert call_kwargs.kwargs["group"] == "example.com"
        assert call_kwargs.kwargs["version"] == "v1beta1"
        assert call_kwargs.kwargs["plural"] == "customresources"

    def test_apply_k8s_error_returns_error_response(self, client, mock_k8s_clients):
        """Test that K8s API errors return proper error response."""
        apps, core, custom = mock_k8s_clients

        apps.create_namespaced_deployment.side_effect = ApiException(
            status=403, reason="Forbidden"
        )

        body = {
            "service_name": "forbidden-deployment",
            "namespace": "default",
            "resource_type": "deployment",
            "resource_manifest": {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {"name": "forbidden-deployment"},
            },
        }

        resp = client.post("/controller/apply", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"
        assert "Forbidden" in data["message"]


# =============================================================================
# Deploy Route Tests (/controller/deploy)
# =============================================================================


class TestDeployRoute:
    """Tests for the /controller/deploy endpoint."""

    @pytest.fixture(autouse=True)
    def mock_pool_helpers(self):
        """Mock pool helper functions that interact with K8s."""
        with patch(
            "routes.pool.create_service_if_not_exists", new_callable=AsyncMock
        ) as mock_create_svc, patch(
            "routes.pool.discover_resource_from_pods", return_value=(None, None)
        ) as mock_discover, patch(
            "routes.pool.broadcast_reload_via_websocket",
            new_callable=AsyncMock,
            return_value={"status": "success", "sent": 0, "total": 0},
        ) as mock_broadcast:
            self.mock_create_svc = mock_create_svc
            self.mock_discover = mock_discover
            self.mock_broadcast = mock_broadcast
            yield

    def test_deploy_creates_resource_and_registers_pool(self, client, mock_k8s_clients):
        """Test deploy creates K8s resource and registers pool in one call."""
        apps, core, custom = mock_k8s_clients

        body = {
            "service_name": "my-service",
            "namespace": "default",
            "resource_type": "deployment",
            "resource_manifest": {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {"name": "my-service"},
                "spec": {"replicas": 1},
            },
            "specifier": {"type": "label_selector", "selector": {"app": "my-service"}},
        }

        resp = client.post("/controller/deploy", json=body)
        assert resp.status_code == 200
        data = resp.json()

        # Check apply result
        assert data["apply_status"] == "success"
        assert data["apply_message"] == "Resource applied"
        assert data["service_name"] == "my-service"
        assert data["resource_type"] == "deployment"

        # Check pool result
        assert data["pool_status"] == "success"
        assert data["pool_message"] == "Pool registered"
        assert "my-service.default.svc.cluster.local" in data["service_url"]
        assert data["resource_kind"] == "Deployment"

        # Verify apply was called
        apps.create_namespaced_deployment.assert_called_once()

    def test_deploy_apply_failure_skips_pool_registration(
        self, client, mock_k8s_clients
    ):
        """Test that apply failure returns error and skips pool registration."""
        apps, core, custom = mock_k8s_clients

        apps.create_namespaced_deployment.side_effect = ApiException(
            status=403, reason="Forbidden"
        )

        body = {
            "service_name": "forbidden-service",
            "namespace": "default",
            "resource_type": "deployment",
            "resource_manifest": {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {"name": "forbidden-service"},
            },
            "specifier": {"type": "label_selector", "selector": {"app": "forbidden"}},
        }

        resp = client.post("/controller/deploy", json=body)
        assert resp.status_code == 200
        data = resp.json()

        assert data["apply_status"] == "error"
        assert "Forbidden" in data["apply_message"]
        assert data["pool_status"] == "skipped"
        assert "Apply failed" in data["pool_message"]

        # Broadcast should not be called when apply fails
        self.mock_broadcast.assert_not_called()

    def test_deploy_resource_kind_mapping(self, client, mock_k8s_clients):
        """Test resource_type to resource_kind mapping for various types."""
        apps, core, custom = mock_k8s_clients

        test_cases = [
            ("deployment", "Deployment"),
            ("raycluster", "RayCluster"),
            ("pytorchjob", "PyTorchJob"),
            ("knative", "KnativeService"),
        ]

        for resource_type, expected_kind in test_cases:
            body = {
                "service_name": f"test-{resource_type}",
                "namespace": "default",
                "resource_type": resource_type,
                "resource_manifest": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "metadata": {"name": f"test-{resource_type}"},
                },
                "specifier": {"type": "label_selector", "selector": {"app": "test"}},
            }

            resp = client.post("/controller/deploy", json=body)
            assert resp.status_code == 200
            data = resp.json()
            assert data["resource_kind"] == expected_kind, f"Failed for {resource_type}"

    def test_deploy_knative_skips_service_creation(self, client, mock_k8s_clients):
        """Test deploy with Knative skips K8s service creation (Knative manages its own)."""
        apps, core, custom = mock_k8s_clients

        body = {
            "service_name": "my-knative",
            "namespace": "default",
            "resource_type": "knative",
            "resource_manifest": {
                "apiVersion": "serving.knative.dev/v1",
                "kind": "Service",
                "metadata": {"name": "my-knative"},
            },
            "specifier": {"type": "label_selector", "selector": {"app": "my-knative"}},
        }

        resp = client.post("/controller/deploy", json=body)
        assert resp.status_code == 200
        data = resp.json()

        assert data["apply_status"] == "success"
        assert data["pool_status"] == "success"

        # Knative should not create K8s service
        self.mock_create_svc.assert_not_called()


# =============================================================================
# Pool Route Tests (/controller/pool)
# =============================================================================


class TestPoolRoutes:
    """Tests for the /controller/pool endpoints."""

    @pytest.fixture(autouse=True)
    def mock_pool_helpers(self):
        """Mock pool helper functions that interact with K8s."""
        with patch(
            "routes.pool.create_service_if_not_exists", new_callable=AsyncMock
        ) as mock_create_svc, patch(
            "routes.pool.discover_resource_from_pods", return_value=(None, None)
        ) as mock_discover, patch(
            "routes.pool.broadcast_reload_via_websocket",
            new_callable=AsyncMock,
            return_value={"status": "success", "sent": 0, "total": 0},
        ) as mock_broadcast:
            self.mock_create_svc = mock_create_svc
            self.mock_discover = mock_discover
            self.mock_broadcast = mock_broadcast
            yield

    def test_register_pool_success(self, client, mock_k8s_clients):
        """Test registering a new pool."""
        body = {
            "name": "my-pool",
            "namespace": "default",
            "specifier": {"type": "label_selector", "selector": {"app": "workers"}},
        }

        resp = client.post("/controller/pool", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "my-pool"
        assert data["namespace"] == "default"
        assert data["status"] == "success"
        assert data["message"] == "Pool registered"
        assert "my-pool.default.svc.cluster.local" in data["service_url"]

        self.mock_create_svc.assert_called()

    def test_register_pool_with_custom_port(self, client, mock_k8s_clients):
        """Test registering a pool with custom server port."""
        body = {
            "name": "custom-port-pool",
            "namespace": "default",
            "specifier": {"type": "label_selector", "selector": {"app": "workers"}},
            "server_port": 8080,
        }

        resp = client.post("/controller/pool", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["server_port"] == 8080

    def test_register_pool_with_headless_service(self, client, mock_k8s_clients):
        """Test registering a pool with headless service for SPMD."""
        body = {
            "name": "spmd-pool",
            "namespace": "default",
            "specifier": {"type": "label_selector", "selector": {"app": "workers"}},
            "create_headless_service": True,
        }

        resp = client.post("/controller/pool", json=body)
        assert resp.status_code == 200

        # Should create both regular and headless services
        assert self.mock_create_svc.call_count == 2

    def test_register_pool_with_url_service_config(self, client, mock_k8s_clients):
        """Test registering a pool with user-provided service URL."""
        body = {
            "name": "url-pool",
            "namespace": "default",
            "specifier": {"type": "label_selector", "selector": {"app": "workers"}},
            "service": {"url": "https://my-custom-service.example.com"},
        }

        resp = client.post("/controller/pool", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["service_url"] == "https://my-custom-service.example.com"

        # Should NOT create K8s service when URL is provided
        self.mock_create_svc.assert_not_called()

    def test_register_pool_with_module(self, client, mock_k8s_clients):
        """Test registering a pool with module configuration."""
        body = {
            "name": "module-pool",
            "namespace": "default",
            "specifier": {"type": "label_selector", "selector": {"app": "workers"}},
            "module": {
                "type": "fn",
                "pointers": {"fn_name": "my_func"},
                "dispatch": "regular",
                "procs": 1,
            },
        }

        resp = client.post("/controller/pool", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["module"]["type"] == "fn"
        assert data["module"]["pointers"]["fn_name"] == "my_func"

    def test_register_pool_update_existing(self, client, mock_k8s_clients):
        """Test updating an existing pool."""
        body = {
            "name": "update-pool",
            "namespace": "default",
            "specifier": {"type": "label_selector", "selector": {"app": "workers"}},
        }

        # Create pool
        resp1 = client.post("/controller/pool", json=body)
        assert resp1.status_code == 200

        # Update pool with new port
        body["server_port"] = 9000
        resp2 = client.post("/controller/pool", json=body)
        assert resp2.status_code == 200
        data = resp2.json()
        assert data["server_port"] == 9000

        # Service should only be created once (on first registration)
        assert self.mock_create_svc.call_count == 1

    def test_list_pools(self, client, mock_k8s_clients):
        """Test listing pools in a namespace."""
        # Create multiple pools
        for i in range(3):
            body = {
                "name": f"pool-{i}",
                "namespace": "test-ns",
                "specifier": {
                    "type": "label_selector",
                    "selector": {"app": f"app-{i}"},
                },
            }
            client.post("/controller/pool", json=body)

        resp = client.get("/controller/pools/test-ns")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["pools"]) == 3
        pool_names = [p["name"] for p in data["pools"]]
        assert "pool-0" in pool_names
        assert "pool-1" in pool_names
        assert "pool-2" in pool_names

    def test_list_pools_empty_namespace(self, client, mock_k8s_clients):
        """Test listing pools in empty namespace returns empty list."""
        resp = client.get("/controller/pools/empty-ns")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pools"] == []

    def test_get_pool(self, client, mock_k8s_clients):
        """Test getting a specific pool."""
        body = {
            "name": "get-pool",
            "namespace": "default",
            "specifier": {"type": "label_selector", "selector": {"app": "workers"}},
            "server_port": 8080,
        }
        client.post("/controller/pool", json=body)

        resp = client.get("/controller/pool/default/get-pool")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "get-pool"
        assert data["namespace"] == "default"
        assert data["status"] == "active"
        assert data["server_port"] == 8080

    def test_get_pool_not_found(self, client, mock_k8s_clients):
        """Test getting a non-existent pool returns 404."""
        resp = client.get("/controller/pool/default/nonexistent")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]

    def test_delete_pool(self, client, mock_k8s_clients):
        """Test deleting a pool."""
        apps, core, custom = mock_k8s_clients

        body = {
            "name": "delete-pool",
            "namespace": "default",
            "specifier": {"type": "label_selector", "selector": {"app": "workers"}},
        }
        client.post("/controller/pool", json=body)

        resp = client.delete("/controller/pool/default/delete-pool")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "deleted" in data["message"].lower()

        # Verify pool is gone
        resp = client.get("/controller/pool/default/delete-pool")
        assert resp.status_code == 404

    def test_delete_pool_not_found(self, client, mock_k8s_clients):
        """Test deleting a non-existent pool returns 404."""
        resp = client.delete("/controller/pool/default/nonexistent")
        assert resp.status_code == 404

    def test_delete_pool_cleans_up_services(self, client, mock_k8s_clients):
        """Test deleting a pool attempts to delete K8s services."""
        apps, core, custom = mock_k8s_clients

        body = {
            "name": "cleanup-pool",
            "namespace": "default",
            "specifier": {"type": "label_selector", "selector": {"app": "workers"}},
        }
        client.post("/controller/pool", json=body)

        resp = client.delete("/controller/pool/default/cleanup-pool")
        assert resp.status_code == 200

        # Should try to delete regular and headless services
        assert core.delete_namespaced_service.call_count == 2

    def test_register_pool_discovers_resource_type(self, client, mock_k8s_clients):
        """Test pool registration discovers resource type from pods."""
        self.mock_discover.return_value = ("Deployment", "my-deployment")

        body = {
            "name": "discover-pool",
            "namespace": "default",
            "specifier": {"type": "label_selector", "selector": {"app": "workers"}},
        }

        resp = client.post("/controller/pool", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["resource_kind"] == "Deployment"
        assert data["resource_name"] == "my-deployment"

    def test_register_pool_with_labels_and_annotations(self, client, mock_k8s_clients):
        """Test registering a pool with labels and annotations."""
        body = {
            "name": "labeled-pool",
            "namespace": "default",
            "specifier": {"type": "label_selector", "selector": {"app": "workers"}},
            "labels": {"team": "ml", "env": "prod"},
            "annotations": {"description": "ML worker pool"},
        }

        resp = client.post("/controller/pool", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["labels"] == {"team": "ml", "env": "prod"}
        assert data["annotations"] == {"description": "ML worker pool"}


# =============================================================================
# Check Ready Route Tests (/controller/check-ready)
# =============================================================================


class TestCheckReadyRoute:
    """Tests for the /controller/check-ready endpoint."""

    @pytest.fixture(autouse=True)
    def mock_readiness_checks(self):
        """Mock readiness check functions."""
        with patch(
            "routes.pool.check_deployment_ready", new_callable=AsyncMock
        ) as mock_deploy, patch(
            "routes.pool.check_knative_ready", new_callable=AsyncMock
        ) as mock_knative, patch(
            "routes.pool.check_raycluster_ready", new_callable=AsyncMock
        ) as mock_ray, patch(
            "routes.pool.check_trainjob_ready", new_callable=AsyncMock
        ) as mock_train, patch(
            "routes.pool.check_selector_ready", new_callable=AsyncMock
        ) as mock_selector:
            self.mock_deploy = mock_deploy
            self.mock_knative = mock_knative
            self.mock_ray = mock_ray
            self.mock_train = mock_train
            self.mock_selector = mock_selector
            yield

    def test_check_ready_deployment(self, client, mock_k8s_clients):
        """Test checking deployment readiness."""
        from core.models import ReadinessResponse

        self.mock_deploy.return_value = ReadinessResponse(
            ready=True,
            message="Deployment ready",
            resource_type="deployment",
            details={"replicas": 3, "ready_replicas": 3},
        )

        resp = client.get(
            "/controller/check-ready/default/my-deployment?resource_type=deployment"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ready"] is True
        assert data["resource_type"] == "deployment"

        self.mock_deploy.assert_called_once()

    def test_check_ready_knative(self, client, mock_k8s_clients):
        """Test checking Knative service readiness."""
        from core.models import ReadinessResponse

        self.mock_knative.return_value = ReadinessResponse(
            ready=True,
            message="Knative service ready",
            resource_type="knative",
        )

        resp = client.get(
            "/controller/check-ready/default/my-knative?resource_type=knative"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ready"] is True

        self.mock_knative.assert_called_once()

    def test_check_ready_raycluster(self, client, mock_k8s_clients):
        """Test checking RayCluster readiness."""
        from core.models import ReadinessResponse

        self.mock_ray.return_value = ReadinessResponse(
            ready=False,
            message="RayCluster not ready",
            resource_type="raycluster",
        )

        resp = client.get(
            "/controller/check-ready/default/my-ray?resource_type=raycluster"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ready"] is False

        self.mock_ray.assert_called_once()

    def test_check_ready_pytorchjob(self, client, mock_k8s_clients):
        """Test checking PyTorchJob readiness."""
        from core.models import ReadinessResponse

        self.mock_train.return_value = ReadinessResponse(
            ready=True,
            message="PyTorchJob ready",
            resource_type="pytorchjob",
        )

        resp = client.get(
            "/controller/check-ready/default/my-job?resource_type=pytorchjob"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ready"] is True

        self.mock_train.assert_called_once()

    def test_check_ready_selector(self, client, mock_k8s_clients):
        """Test checking selector-based readiness."""
        from core.models import ReadinessResponse

        self.mock_selector.return_value = ReadinessResponse(
            ready=True,
            message="Pods ready",
            resource_type="selector",
        )

        resp = client.get(
            "/controller/check-ready/default/my-selector?resource_type=selector"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ready"] is True

        self.mock_selector.assert_called_once()

    def test_check_ready_unknown_type(self, client, mock_k8s_clients):
        """Test checking readiness with unknown resource type returns 400."""
        resp = client.get(
            "/controller/check-ready/default/my-resource?resource_type=unknown"
        )
        assert resp.status_code == 400
        assert "Unknown resource type" in resp.json()["detail"]

    def test_check_ready_with_timeout(self, client, mock_k8s_clients):
        """Test checking readiness with custom timeout."""
        from core.models import ReadinessResponse

        self.mock_deploy.return_value = ReadinessResponse(
            ready=True,
            message="Ready",
            resource_type="deployment",
        )

        resp = client.get(
            "/controller/check-ready/default/my-deployment"
            "?resource_type=deployment&timeout=600&poll_interval=5"
        )
        assert resp.status_code == 200

        # Verify timeout and poll_interval were passed
        call_args = self.mock_deploy.call_args
        assert call_args.args[2] == 600  # timeout
        assert call_args.args[3] == 5  # poll_interval


# =============================================================================
# Debug Routes Tests
# =============================================================================


class TestDebugRoutes:
    """Tests for debug endpoints."""

    @pytest.fixture(autouse=True)
    def mock_pool_helpers(self):
        """Mock pool helper functions."""
        with patch(
            "routes.pool.create_service_if_not_exists", new_callable=AsyncMock
        ), patch(
            "routes.pool.discover_resource_from_pods", return_value=(None, None)
        ), patch(
            "routes.pool.broadcast_reload_via_websocket",
            new_callable=AsyncMock,
            return_value={"status": "success", "sent": 0, "total": 0},
        ):
            yield

    def test_debug_connections(self, client, mock_k8s_clients):
        """Test debug connections endpoint."""
        # Create a pool first
        body = {
            "name": "debug-pool",
            "namespace": "default",
            "specifier": {"type": "label_selector", "selector": {"app": "workers"}},
        }
        client.post("/controller/pool", json=body)

        resp = client.get("/controller/debug/connections")
        assert resp.status_code == 200
        data = resp.json()

        assert "debug-pool" in data
        assert data["debug-pool"]["namespace"] == "default"
        assert data["debug-pool"]["pod_count"] == 0  # No WebSocket connections in test


# =============================================================================
# Discover Route Tests (/controller/discover)
# =============================================================================


class TestDiscoverRoute:
    """Tests for the /controller/discover endpoint."""

    @pytest.fixture(autouse=True)
    def mock_k8s_list_calls(self, mock_k8s_clients):
        """Mock K8s list calls for discover route."""
        apps, core, custom = mock_k8s_clients

        # Default empty returns
        apps.list_namespaced_deployment.return_value = MagicMock(items=[])
        custom.list_namespaced_custom_object.return_value = {"items": []}

        self.apps = apps
        self.custom = custom

    def test_discover_returns_empty_for_empty_namespace(self, client, mock_k8s_clients):
        """Test discover returns empty lists when no resources exist."""
        resp = client.get("/controller/discover/empty-ns")
        assert resp.status_code == 200
        data = resp.json()

        assert data["knative_services"] == []
        assert data["deployments"] == []
        assert data["rayclusters"] == []
        assert data["training_jobs"] == []
        assert data["pools"] == []

    def test_discover_returns_pools_from_database(self, client, mock_k8s_clients):
        """Test discover returns pools stored in the database."""
        with patch(
            "routes.pool.create_service_if_not_exists", new_callable=AsyncMock
        ), patch("routes.pool.discover_resource_from_pods", return_value=(None, None)):
            body = {
                "name": "discover-test-pool",
                "namespace": "test-ns",
                "specifier": {"type": "label_selector", "selector": {"app": "test"}},
            }
            client.post("/controller/pool", json=body)

        resp = client.get("/controller/discover/test-ns")
        assert resp.status_code == 200
        data = resp.json()

        assert len(data["pools"]) == 1
        assert data["pools"][0]["name"] == "discover-test-pool"
        assert data["pools"][0]["namespace"] == "test-ns"

    def test_discover_with_name_filter(self, client, mock_k8s_clients):
        """Test discover filters pools by name substring."""
        with patch(
            "routes.pool.create_service_if_not_exists", new_callable=AsyncMock
        ), patch("routes.pool.discover_resource_from_pods", return_value=(None, None)):
            for name in ["ml-training", "ml-inference", "data-pipeline"]:
                body = {
                    "name": name,
                    "namespace": "filter-ns",
                    "specifier": {"type": "label_selector", "selector": {"app": name}},
                }
                client.post("/controller/pool", json=body)

        resp = client.get("/controller/discover/filter-ns?name_filter=ml")
        assert resp.status_code == 200
        data = resp.json()

        assert len(data["pools"]) == 2
        pool_names = [p["name"] for p in data["pools"]]
        assert "ml-training" in pool_names
        assert "ml-inference" in pool_names
        assert "data-pipeline" not in pool_names
