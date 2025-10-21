from unittest.mock import Mock, patch

import kubetorch.serving.constants as serving_constants
import pytest
from kubetorch.resources.compute.compute import Compute
from kubetorch.resources.images.image import Image


@pytest.fixture
def mock_service_manager():
    """Mock the ServiceManager to return a specific config"""
    # Create a mock instance that will be returned when any service manager is instantiated
    mock_instance = Mock()

    # Configure the mock to return our test config
    mock_instance.fetch_kubetorch_config.return_value = {
        "COMPUTE_DEFAULTS": """
            inactivity_ttl: "1h"
            labels:
                - key: global-label
                  value: global-value
            annotations:
                - key: global-annotation
                  value: global-value
            env_vars:
                - key: GLOBAL_ENV
                  value: global-env-value
            image_id: "global/default:latest"
            gpu_anti_affinity: true
            launch_timeout: 300
        """
    }

    # Patch all the service manager classes that could be instantiated
    with patch(
        "kubetorch.resources.compute.compute.KnativeServiceManager",
        return_value=mock_instance,
    ) as mock_knative, patch(
        "kubetorch.resources.compute.compute.DeploymentServiceManager",
        return_value=mock_instance,
    ) as mock_deployment, patch(
        "kubetorch.resources.compute.compute.RayClusterServiceManager",
        return_value=mock_instance,
    ) as mock_ray:

        yield {
            "knative": mock_knative,
            "deployment": mock_deployment,
            "ray": mock_ray,
            "instance": mock_instance,
        }


@pytest.mark.skip("Compute config is not used currently")
@pytest.mark.level("unit")
def test_compute_global_config_initialization(mock_service_manager):
    """Test that global config values are properly loaded and set"""
    compute = Compute()

    # Test that values from global config were set
    assert compute.inactivity_ttl == "1h"
    assert compute.labels == {"global-label": "global-value"}
    assert compute.annotations == {"global-annotation": "global-value"}
    assert compute.env_vars == {"GLOBAL_ENV": "global-env-value"}
    assert compute.gpu_anti_affinity is True
    assert compute.launch_timeout == 300


@pytest.mark.skip("Compute config is not used currently")
@pytest.mark.level("unit")
def test_compute_image_override(mock_service_manager):
    """Test that explicitly set image overrides global config"""
    compute = Compute(image=Image(image_id="custom/image:latest"))

    # Test that explicit image takes precedence
    assert compute.server_image == "custom/image:latest"


@pytest.mark.skip("Compute config is not used currently")
@pytest.mark.level("unit")
def test_compute_trapdoor_image_handling(mock_service_manager):
    """Test that trapdoor image is replaced with default server image"""
    compute = Compute(image=Image(image_id=serving_constants.KUBETORCH_IMAGE_TRAPDOOR))
    assert compute.server_image in [
        serving_constants.SERVER_IMAGE_MINIMAL,
        serving_constants.SERVER_IMAGE_WITH_OTEL,
    ]


@pytest.mark.skip("Compute config is not used currently")
@pytest.mark.level("unit")
def test_compute_image_precedence(mock_service_manager):
    """Test image precedence: explicit > global config > default"""
    # First test: global config
    compute1 = Compute()
    assert compute1.server_image == "global/default:latest"

    # Second test: explicit setting
    compute2 = Compute(image=Image(image_id="explicit/image:latest"))
    assert compute2.server_image == "explicit/image:latest"

    # Third test: default when neither is set
    # Create a new mock instance for this specific test case
    mock_instance = Mock()
    mock_instance.fetch_kubetorch_config.return_value = {
        "COMPUTE_DEFAULTS": """
            inactivity_ttl: "1h"
        """
    }

    # Temporarily replace the return values for all service managers
    mock_service_manager["knative"].return_value = mock_instance
    mock_service_manager["deployment"].return_value = mock_instance
    mock_service_manager["ray"].return_value = mock_instance

    compute3 = Compute()
    assert compute3.server_image in [
        serving_constants.SERVER_IMAGE_MINIMAL,
        serving_constants.SERVER_IMAGE_WITH_OTEL,
    ]


@pytest.mark.skip("Compute config is not used currently")
@pytest.mark.level("unit")
def test_compute_config_override(mock_service_manager):
    """Test that explicitly set values override global config"""
    compute = Compute(
        inactivity_ttl="2h",
        labels={"custom-label": "custom-value"},
        annotations={"custom-annotation": "custom-value"},
        env_vars={"CUSTOM_ENV": "custom-value"},
        gpu_anti_affinity=False,
        launch_timeout=600,
        image=Image(image_id="custom/image:latest"),
    )

    # Test that explicit values take precedence
    assert compute.inactivity_ttl == "2h"
    assert compute.labels == {
        "global-label": "global-value",
        "custom-label": "custom-value",
    }
    assert compute.annotations == {
        "global-annotation": "global-value",
        "custom-annotation": "custom-value",
    }
    assert compute.env_vars == {
        "GLOBAL_ENV": "global-env-value",
        "CUSTOM_ENV": "custom-value",
    }
    assert compute.gpu_anti_affinity is False
    assert compute.launch_timeout == 600
    assert compute.server_image == "custom/image:latest"


@pytest.mark.skip("Compute config is not used currently")
@pytest.mark.level("unit")
def test_compute_partial_global_config(mock_service_manager):
    """Test handling of partial global config"""
    # Create a new mock instance for this specific test case
    mock_instance = Mock()
    mock_instance.fetch_kubetorch_config.return_value = {
        "COMPUTE_DEFAULTS": """
            inactivity_ttl: "1h"
            labels:
                - key: global-label
                  value: global-value
        """
    }

    # Temporarily replace the return values for all service managers
    mock_service_manager["knative"].return_value = mock_instance
    mock_service_manager["deployment"].return_value = mock_instance
    mock_service_manager["ray"].return_value = mock_instance

    compute = Compute()

    # Test that partial config is handled correctly
    assert compute.inactivity_ttl == "1h"
    assert compute.labels == {"global-label": "global-value"}
    assert compute.annotations == {}
    assert compute.env_vars == {}
    assert compute.gpu_anti_affinity is False  # Default value
    assert compute.server_image in [
        serving_constants.SERVER_IMAGE_MINIMAL,
        serving_constants.SERVER_IMAGE_WITH_OTEL,
    ]
