import os

import kubetorch as kt
import pytest
import requests
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
    response = requests.get(f"{controller_client.base_url}/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.level("unit")
def test_volume_create_and_delete():
    """Test creating and deleting a volume through the controller"""
    volume_name = "test-controller-vol"

    # Create volume with ReadWriteOnce (available in all clusters)
    vol = kt.Volume(name=volume_name, size="1Gi", access_mode="ReadWriteOnce")
    result = vol.create()

    assert result is not None
    assert vol.name == volume_name

    # Check volume exists
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

    # Don't create the volume, just testing storage class detection


@pytest.mark.level("unit")
def test_secret_create_and_delete():
    """Test creating and deleting a secret through the controller"""
    secret_client = KubernetesSecretsClient()

    secret = Secret(
        name="test-controller-secret",
        values={"TEST_KEY": "test_value", "ANOTHER_KEY": "another_value"},
        env_vars=["TEST_KEY", "ANOTHER_KEY"],
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
        env_vars=["KEY1"],
        override=True,  # Allow updates
    )

    secret_client.create_secret(secret)

    try:
        # Update secret with new values
        updated_secret = Secret(
            name="test-controller-secret-update",
            values={"KEY1": "updated_value", "KEY2": "value2"},
            env_vars=["KEY1", "KEY2"],
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
    secret1 = Secret(name="test-list-1", values={"key": "val1"}, env_vars=["key"])
    secret2 = Secret(name="test-list-2", values={"key": "val2"}, env_vars=["key"])

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
    from kubetorch.servers.http.utils import is_running_in_kubernetes

    controller = kt.globals.controller_client()

    assert controller is not None
    assert controller.base_url is not None

    # Base URL should either be from config or auto-detected
    if not is_running_in_kubernetes():
        # Out of cluster should require KT_API_URL or raise error
        assert kt.config.api_url is not None or "localhost" in controller.base_url
