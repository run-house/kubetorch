import os

import httpx
import kubetorch as kt
import pytest
from kubernetes import client as k8s_client, config as k8s_config

from kubetorch.resources.secrets import Secret
from kubetorch.resources.secrets.kubernetes_secrets_client import KubernetesSecretsClient


@pytest.fixture(autouse=True, scope="session")
def setup_test_env():
    """Setup test environment"""
    os.environ["KT_GPU_ANTI_AFFINITY"] = "True"
    yield


@pytest.mark.level("unit")
def test_controller_health():
    """Test that the controller is accessible and healthy"""
    controller_client = kt.globals.controller_client()

    # Test health endpoint through the controller base URL
    response = httpx.get(f"{controller_client.base_url}/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.level("unit")
def test_volume_create_and_delete():
    """Test creating and deleting a volume through the controller"""
    volume_name = "test-controller-vol"

    # Create volume with ReadWriteOnce (available in all clusters)
    vol = kt.Volume(name=volume_name, size="1Gi", access_mode="ReadWriteOnce", mount_path="/tmp/controller-test")
    result = vol.create()

    assert result is not None
    assert vol.name == volume_name
    assert vol.exists() is True

    # Delete volume
    vol.delete()

    # Verify deletion
    assert vol.exists() is False


@pytest.mark.level("unit")
def test_volume_from_name():
    """Test loading a volume from name through the controller"""
    volume_name = "test-controller-vol-load"

    # Create volume with ReadWriteOnce
    vol = kt.Volume(name=volume_name, size="2Gi", mount_path="/test/path", access_mode="ReadWriteOnce")
    vol.create()

    try:
        # Load from name
        loaded_vol = kt.Volume.from_name(volume_name)

        assert loaded_vol.name == volume_name
        assert loaded_vol.size == "2Gi"
        assert loaded_vol.mount_path == "/test/path"

    finally:
        # Cleanup
        vol.delete()


@pytest.mark.level("unit")
def test_volume_storage_class_detection():
    """Test storage class auto-detection through the controller"""
    vol = kt.Volume(name="test-storage-class", size="1Gi", access_mode="ReadWriteOnce", mount_path="/test/path")

    # This should auto-detect the storage class via controller
    storage_class = vol.storage_class

    assert storage_class is not None
    assert isinstance(storage_class, str)
    assert len(storage_class) > 0


@pytest.mark.level("unit")
def test_secret_create_and_delete():
    """Test creating and deleting a secret through the controller"""
    secret_client = KubernetesSecretsClient()

    secret = Secret(
        name="test-controller-secret",
        values={"TEST_KEY": "test_value", "ANOTHER_KEY": "another_value"},
        env_vars={"TEST_KEY": "TEST_KEY", "ANOTHER_KEY": "ANOTHER_KEY"},
    )

    # Create secret
    success = secret_client.create_secret(secret)
    assert success is True

    try:
        # Read secret back
        secret_config = secret_client._read_secret("test-controller-secret")

        assert secret_config is not None
        assert "values" in secret_config
        assert secret_config["values"]["TEST_KEY"] == "test_value"
        assert secret_config["values"]["ANOTHER_KEY"] == "another_value"

    finally:
        # Cleanup
        secret_client._delete_secret("test-controller-secret")


@pytest.mark.level("unit")
def test_secret_update():
    """Test updating a secret through the controller"""
    secret_client = KubernetesSecretsClient()

    # Create initial secret
    secret = Secret(
        name="test-controller-secret-update",
        values={"KEY1": "value1"},
        env_vars={"KEY1": "KEY1"},
        override=True,  # Allow updates
    )

    secret_client.create_secret(secret)

    try:
        # Update secret with new values
        updated_secret = Secret(
            name="test-controller-secret-update",
            values={"KEY1": "updated_value", "KEY2": "value2"},
            env_vars={"KEY1": "KEY1", "KEY2": "KEY2"},
            override=True,
        )

        success = secret_client.update_secret(updated_secret)
        assert success is True

        # Verify update
        secret_config = secret_client._read_secret("test-controller-secret-update")
        assert secret_config["values"]["KEY1"] == "updated_value"
        assert secret_config["values"]["KEY2"] == "value2"

    finally:
        # Cleanup
        secret_client._delete_secret("test-controller-secret-update")


@pytest.mark.level("unit")
def test_volume_config():
    """Test volume configuration through the controller"""
    vol = kt.Volume(
        name="test-vol-config",
        size="5Gi",
        storage_class="standard",
        mount_path="/custom/mount",
        access_mode="ReadWriteOnce",
    )

    config = vol.config()

    assert config["name"] == "test-vol-config"
    assert config["size"] == "5Gi"
    assert config["storage_class"] == "standard"
    assert config["mount_path"] == "/custom/mount"
    assert config["access_mode"] == "ReadWriteOnce"


@pytest.mark.level("unit")
def test_secret_list_and_delete_all():
    """Test listing and deleting all secrets for a user through the controller"""
    secret_client = KubernetesSecretsClient()

    # Create multiple secrets
    secret1 = Secret(name="test-list-1", values={"key": "val1"}, env_vars={"key": "key"})
    secret2 = Secret(name="test-list-2", values={"key": "val2"}, env_vars={"key": "key"})

    secret_client.create_secret(secret1)
    secret_client.create_secret(secret2)

    try:
        # List should include our secrets
        # Note: This will list all secrets for the current user
        result = secret_client.controller_client.list_secrets(
            secret_client.namespace, label_selector=f"kubetorch.com/username={kt.config.username}"
        )

        secret_names = [s["metadata"]["name"] for s in result.get("items", [])]

        # Our secrets should be in the list (with formatted names)
        formatted_name1 = secret_client._format_secret_name("test-list-1")
        formatted_name2 = secret_client._format_secret_name("test-list-2")

        assert formatted_name1 in secret_names
        assert formatted_name2 in secret_names

    finally:
        # Cleanup
        secret_client._delete_secret("test-list-1")
        secret_client._delete_secret("test-list-2")


@pytest.mark.level("unit")
def test_controller_client_initialization():
    """Test controller client initialization and base URL detection"""
    from kubetorch.serving.utils import is_running_in_kubernetes

    controller = kt.globals.controller_client()

    assert controller is not None
    assert controller.base_url is not None

    # Base URL should either be from config or auto-detected
    if not is_running_in_kubernetes():
        # Out of cluster should require KT_API_URL or raise error
        assert kt.config.api_url is not None or "localhost" in controller.base_url


# =============================================================================
# PVC Tests - Comprehensive coverage of /api/v1/namespaces/{ns}/persistentvolumeclaims
# =============================================================================
@pytest.mark.level("unit")
def test_pvc_list_with_label_selector():
    """Test GET /api/v1/namespaces/{ns}/persistentvolumeclaims with label_selector"""
    controller_client = kt.globals.controller_client()

    result = controller_client.list_resources(
        "pvcs", namespace=kt.config.namespace, label_selector=f"kubetorch.com/username={kt.config.username}"
    )

    assert "items" in result
    assert isinstance(result["items"], list)


@pytest.mark.level("unit")
def test_pvc_get_nonexistent():
    """Test GET PVC returns None with ignore_not_found=True"""
    controller_client = kt.globals.controller_client()

    result = controller_client.get_pvc(
        namespace=kt.config.namespace, name="nonexistent-pvc-12345", ignore_not_found=True
    )

    assert result is None


@pytest.mark.level("unit")
def test_service_get_nonexistent():
    """Test GET service returns None with ignore_not_found=True"""
    controller_client = kt.globals.controller_client()

    result = controller_client.get_service(
        namespace=kt.config.namespace, name="nonexistent-service-12345", ignore_not_found=True
    )

    assert result is None


@pytest.mark.level("unit")
def test_deployment_get_nonexistent():
    """Test GET deployment returns None with ignore_not_found=True"""
    controller_client = kt.globals.controller_client()

    result = controller_client.get_deployment(
        namespace=kt.config.namespace, name="nonexistent-deploy-12345", ignore_not_found=True
    )

    assert result is None


# =============================================================================
# Secret Tests - /api/v1/namespaces/{ns}/secrets
# =============================================================================
@pytest.mark.level("unit")
def test_secret_patch():
    """Test PATCH /api/v1/namespaces/{ns}/secrets/{name}"""
    secret_client = KubernetesSecretsClient()
    controller_client = kt.globals.controller_client()

    secret = Secret(
        name="test-secret-patch",
        values={"KEY1": "value1"},
        env_vars={"KEY1": "KEY1"},
    )
    secret_client.create_secret(secret)

    try:
        formatted_name = secret_client._format_secret_name("test-secret-patch")
        patch_body = {"stringData": {"KEY2": "value2"}}

        patched = controller_client.patch_secret(namespace=kt.config.namespace, name=formatted_name, body=patch_body)

        assert patched is not None

        # Verify patch
        secret_config = secret_client._read_secret("test-secret-patch")
        assert "KEY2" in secret_config["values"]

    finally:
        secret_client._delete_secret("test-secret-patch")


@pytest.mark.level("unit")
def test_secret_list_all_namespaces():
    """Test GET /api/v1/secrets (all namespaces)"""
    controller_client = kt.globals.controller_client()

    try:
        result = controller_client.list_secrets_all_namespaces()
        assert "items" in result
    except kt.ControllerRequestError as e:
        # May not have permission across all namespaces - that's okay
        assert e.status_code in [403, 404]


# =============================================================================
# Pod Tests - /api/v1/namespaces/{ns}/pods
# =============================================================================
@pytest.mark.level("unit")
def test_pod_list():
    """Test GET /api/v1/namespaces/{ns}/pods"""
    controller_client = kt.globals.controller_client()

    result = controller_client.list_pods(namespace=kt.config.namespace)

    assert "items" in result
    assert isinstance(result["items"], list)


# =============================================================================
# Namespace Tests - /api/v1/namespaces
# NOTE: These are cluster-scoped resources. By design, the controller should
# not have cluster-level permissions.
# =============================================================================
# @pytest.mark.level("unit")
# def test_namespace_operations():
#     """Test GET /api/v1/namespaces - Requires cluster-level permissions"""
#     controller_client = kt.globals.controller_client()
#     namespaces = controller_client.list_namespaces()
#     assert "items" in namespaces


# =============================================================================
# Node Tests - /api/v1/nodes
# NOTE: These are cluster-scoped resources. By design, the controller should
# NOT have cluster-level permissions. These tests are commented out as they
# test operations the controller intentionally cannot perform for security.
# =============================================================================
# @pytest.mark.level("unit")
# def test_node_operations():
#     """Test GET /api/v1/nodes - Requires cluster-level permissions"""
#     controller_client = kt.globals.controller_client()
#     nodes = controller_client.list_nodes()
#     assert "items" in nodes


# =============================================================================
# StorageClass Tests - /apis/storage.k8s.io/v1/storageclasses
# =============================================================================
@pytest.mark.level("unit")
def test_storageclass_operations():
    """Test GET /apis/storage.k8s.io/v1/storageclasses"""
    controller_client = kt.globals.controller_client()

    scs = controller_client.list_storage_classes()
    assert "items" in scs
    assert len(scs["items"]) > 0


# =============================================================================
# Ingress Tests - /apis/networking.k8s.io/v1/namespaces/{ns}/ingresses
# =============================================================================
@pytest.mark.level("unit")
def test_ingress_list():
    """Test GET /apis/networking.k8s.io/v1/namespaces/{ns}/ingresses"""
    controller_client = kt.globals.controller_client()

    result = controller_client.list_ingresses(namespace=kt.config.namespace)
    assert "items" in result


# =============================================================================
# Error Handling Tests
# =============================================================================
@pytest.mark.level("unit")
def test_error_handling_404():
    """Test 404 error handling with and without ignore_not_found"""
    controller_client = kt.globals.controller_client()
    namespace = kt.config.namespace

    result = controller_client.get_secret(namespace=namespace, name="nonexistent-404", ignore_not_found=True)
    assert result is None

    with pytest.raises(kt.ControllerRequestError) as exc_info:
        controller_client.get_pod(namespace=namespace, name="nonexistent-404")
    assert exc_info.value.status_code == 404


@pytest.mark.level("unit")
def test_controller_http_methods():
    """Test base HTTP methods (GET/POST/PATCH/DELETE)"""
    controller_client = kt.globals.controller_client()

    # Test GET with list endpoint
    result = controller_client.list_pods(namespace=kt.config.namespace)
    assert result is not None
    assert "items" in result

    # Test GET with ignore_not_found
    result = controller_client.get_pod(namespace=kt.config.namespace, name="nonexistent", ignore_not_found=True)
    assert result is None


@pytest.mark.level("unit")
def test_discover_resources():
    """Test GET /controller/discover/{namespace} endpoint"""
    controller_client = kt.globals.controller_client()
    namespace = kt.config.namespace

    result = controller_client.discover_resources(namespace=namespace)

    # Verify the response structure contains expected resource types
    assert result is not None
    assert isinstance(result, dict)

    expected_keys = ["knative_services", "deployments", "rayclusters", "training_jobs", "pools"]
    for key in expected_keys:
        assert key in result, f"Missing expected key: {key}"
        assert isinstance(result[key], list), f"{key} should be a list"


@pytest.mark.level("unit")
def test_discover_resources_with_filters():
    """Test discover_resources with name_filter and prefix_filter"""
    controller_client = kt.globals.controller_client()
    namespace = kt.config.namespace

    # Test with name_filter (should return empty or matching resources)
    result = controller_client.discover_resources(
        namespace=namespace,
        name_filter="nonexistent-filter-12345",
    )
    assert result is not None
    assert isinstance(result, dict)
    assert [result.get(key) == [] for key in result.keys()]

    # Test with prefix_filter
    result = controller_client.discover_resources(
        namespace=namespace,
        prefix_filter="test-prefix-",
    )
    assert result is not None
    assert isinstance(result, dict)
    assert [result.get(key) == [] for key in result.keys()]


@pytest.mark.level("unit")
def test_all_list_operations_structure():
    """Test that all list operations return consistent structure

    Note: Only tests namespace-scoped resources that the controller has permissions for.
    Cluster-scoped resources (namespaces, nodes) are excluded by design for security.
    """
    controller_client = kt.globals.controller_client()
    namespace = kt.config.namespace

    # Only test namespace-scoped resources (controller has namespace-level permissions)
    list_ops = [
        ("list_pvcs", {"namespace": namespace}),
        ("list_secrets", {"namespace": namespace}),
        ("list_pods", {"namespace": namespace}),
        ("list_config_maps", {"namespace": namespace}),
        ("list_storage_classes", {}),  # Cluster-scoped but read-only, usually allowed
    ]

    for method_name, kwargs in list_ops:
        method = getattr(controller_client, method_name)
        result = method(**kwargs)

        assert result is not None, f"{method_name} returned None"
        assert "items" in result, f"{method_name} missing 'items'"
        assert isinstance(result["items"], list), f"{method_name} items not a list"

    # Namespace-scoped resources available via list_resources endpoint
    ns_resource_types = ["pvcs", "configmaps", "secrets", "pods", "replicasets", "events", "ingresses"]
    for resource_type in ns_resource_types:
        result = controller_client.list_resources(resource_type, namespace=namespace)

        assert result is not None, f"list_resources({resource_type}) returned None"
        assert "items" in result, f"list_resources({resource_type}) missing 'items'"
        assert isinstance(result["items"], list), f"list_resources({resource_type}) items not a list"


# =============================================================================
# Deploy Endpoint Tests (/controller/deploy)
# =============================================================================
@pytest.mark.level("unit")
def test_deploy_method_signature():
    """Test that deploy method exists and has correct signature"""
    controller_client = kt.globals.controller_client()

    # Check deploy method exists
    assert hasattr(controller_client, "deploy")
    assert callable(controller_client.deploy)

    # Check it has the expected parameters (via introspection)
    import inspect

    sig = inspect.signature(controller_client.deploy)
    param_names = list(sig.parameters.keys())

    # Required parameters
    assert "service_name" in param_names
    assert "namespace" in param_names
    assert "resource_type" in param_names
    assert "resource_manifest" in param_names
    assert "specifier" in param_names

    # Optional parameters
    assert "server_port" in param_names
    assert "create_headless_service" in param_names


# =============================================================================
# Full Stack Integration Test
# =============================================================================
@pytest.mark.level("minimal")
def test_integration_full_stack():
    """Test end-to-end: service + secret + volume"""
    from tests.conftest import get_test_hash
    from tests.utils import get_env_var, get_test_fn_name

    secret_client = KubernetesSecretsClient()
    service_name = get_test_fn_name()
    secret_name = f"{get_test_hash()}-{service_name}-sec"
    volume_name = f"{get_test_hash()}-{service_name}-vol"

    secret = Secret(name=secret_name, values={"TEST_KEY": "test_val"}, env_vars={"TEST_KEY": "TEST_KEY"})
    secret_client.create_secret(secret)

    vol = kt.Volume(name=volume_name, size="1Gi", access_mode="ReadWriteOnce", mount_path="/data")
    vol.create()

    try:
        # Deploy with resources - (pass Secret object instead of string to avoid provider inference)
        remote_fn = kt.fn(get_env_var, name=service_name).to(
            kt.Compute(cpus=".01", gpu_anti_affinity=True, secrets=[secret], volumes=[vol], launch_timeout=300)
        )

        assert remote_fn("TEST_KEY") == "test_val"

        controller_client = kt.globals.controller_client()
        namespace = remote_fn.compute.namespace
        actual_service_name = remote_fn.service_name

        assert controller_client.get_deployment(namespace=namespace, name=actual_service_name) is not None
        assert controller_client.get_service(namespace=namespace, name=actual_service_name) is not None

        pods = controller_client.list_pods(
            namespace=namespace, label_selector=f"kubetorch.com/service={actual_service_name}"
        )
        assert len(pods["items"]) > 0

    finally:
        secret_client._delete_secret(secret_name)
        vol.delete()


# =============================================================================
# Service Tests
# =============================================================================
@pytest.mark.level("minimal")
def test_service_operations():
    """Test GET /api/v1/namespaces/{ns}/services/{name}"""
    from tests.utils import get_test_fn_name, summer

    service_name = get_test_fn_name()

    remote_fn = kt.fn(summer, name=service_name).to(kt.Compute(cpus=".01", gpu_anti_affinity=True))
    result = remote_fn(1, 2)
    assert result == 3

    controller_client = kt.globals.controller_client()
    namespace = remote_fn.compute.namespace
    actual_service_name = remote_fn.service_name

    service = controller_client.get_service(namespace=namespace, name=actual_service_name)
    assert service is not None
    assert service["metadata"]["name"] == actual_service_name
    assert service["kind"] == "Service"


# =============================================================================
# Deployment Tests
# =============================================================================
@pytest.mark.level("minimal")
def test_deployment_operations():
    """Test GET /apis/apps/v1/namespaces/{ns}/deployments/{name}"""
    from tests.utils import get_test_fn_name, summer

    service_name = get_test_fn_name()
    remote_fn = kt.fn(summer, name=service_name).to(kt.Compute(cpus=".01", gpu_anti_affinity=True))
    result = remote_fn(1, 2)
    assert result == 3

    controller_client = kt.globals.controller_client()
    namespace = remote_fn.compute.namespace
    actual_service_name = remote_fn.service_name

    deployment = controller_client.get_deployment(namespace=namespace, name=actual_service_name)
    assert deployment is not None
    assert deployment["kind"] == "Deployment"


# =============================================================================
# ConfigMap Tests
# =============================================================================
@pytest.mark.level("minimal")
def test_configmap_operations():
    """Test GET /api/v1/namespaces/{ns}/configmaps"""
    from tests.utils import get_test_fn_name, summer

    service_name = get_test_fn_name()

    remote_fn = kt.fn(summer, name=service_name).to(kt.Compute(cpus=".01", gpu_anti_affinity=True))
    result = remote_fn(1, 2)
    assert result == 3

    controller_client = kt.globals.controller_client()
    namespace = remote_fn.compute.namespace

    configmaps = controller_client.list_resources("configmaps", namespace=namespace)
    assert "items" in configmaps

    cms = controller_client.list_resources(
        "configmaps", namespace=namespace, label_selector=f"kubetorch.com/service={service_name}"
    )

    if len(cms["items"]) > 0:
        cm_name = cms["items"][0]["metadata"]["name"]
        cm = controller_client.get_config_map(namespace=namespace, name=cm_name)
        assert cm["kind"] == "ConfigMap"


# =============================================================================
# Pod Tests
# =============================================================================
@pytest.mark.level("minimal")
def test_pod_get_and_logs():
    """Test GET pod and pod logs"""
    from tests.utils import get_test_fn_name, summer

    service_name = get_test_fn_name()

    remote_fn = kt.fn(summer, name=service_name).to(kt.Compute(cpus=".01", gpu_anti_affinity=True))
    result = remote_fn(1, 2)
    assert result == 3

    controller_client = kt.globals.controller_client()
    namespace = remote_fn.compute.namespace
    actual_service_name = remote_fn.service_name

    pods = controller_client.list_pods(
        namespace=namespace, label_selector=f"kubetorch.com/service={actual_service_name}"
    )
    assert len(pods["items"]) > 0

    pod_name = pods["items"][0]["metadata"]["name"]

    pod = controller_client.get_pod(namespace=namespace, name=pod_name)
    assert pod["kind"] == "Pod"

    logs = controller_client.get_pod_logs(namespace=namespace, name=pod_name)
    assert isinstance(logs, str)
    assert len(logs) > 0


# =============================================================================
# Custom Resource Tests
# =============================================================================
@pytest.mark.level("minimal")
def test_custom_object_knative():
    """Test Custom Object GET for Knative Services"""
    from tests.utils import get_test_fn_name, summer

    service_name = get_test_fn_name()
    remote_fn = kt.fn(summer, name=service_name).to(
        kt.Compute(cpus=".01", gpu_anti_affinity=True).autoscale(min_replicas=1, max_replicas=2)
    )
    result = remote_fn(1, 2)
    assert result == 3

    namespace = remote_fn.compute.namespace
    actual_service_name = remote_fn.service_name

    # Load k8s config
    try:
        k8s_config.load_incluster_config()
    except k8s_config.ConfigException:
        k8s_config.load_kube_config()

    custom_api = k8s_client.CustomObjectsApi()
    ksvc = custom_api.get_namespaced_custom_object(
        group="serving.knative.dev",
        version="v1",
        namespace=namespace,
        plural="services",
        name=actual_service_name,
    )
    assert ksvc["kind"] == "Service"
    assert ksvc["apiVersion"] == "serving.knative.dev/v1"


@pytest.mark.level("minimal")
def test_custom_object_ray():
    """Test Custom Object GET for Ray Clusters"""
    from tests.utils import get_test_fn_name, summer

    service_name = get_test_fn_name()
    remote_fn = kt.fn(summer, name=service_name).to(
        kt.Compute(
            cpus="2",
            memory="3Gi",
            gpu_anti_affinity=True,
            image=kt.images.Ray(),
            launch_timeout=450,  # Ray clusters take longer to start
        ).distribute("ray", workers=2)
    )
    result = remote_fn(1, 2)
    assert result == 3

    namespace = remote_fn.compute.namespace
    actual_service_name = remote_fn.service_name

    # Load k8s config
    try:
        k8s_config.load_incluster_config()
    except k8s_config.ConfigException:
        k8s_config.load_kube_config()

    custom_api = k8s_client.CustomObjectsApi()
    raycluster = custom_api.get_namespaced_custom_object(
        group="ray.io",
        version="v1",
        namespace=namespace,
        plural="rayclusters",
        name=actual_service_name,
    )
    assert raycluster["kind"] == "RayCluster"
    assert raycluster["apiVersion"] == "ray.io/v1"
