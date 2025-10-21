# Backward compatibility imports - all service manager functionality is now in separate files
from kubetorch.logger import get_logger

# Import all service managers for backward compatibility and centralized access
from kubetorch.serving.base_service_manager import BaseServiceManager
from kubetorch.serving.deployment_service_manager import DeploymentServiceManager
from kubetorch.serving.knative_service_manager import KnativeServiceManager
from kubetorch.serving.raycluster_service_manager import RayClusterServiceManager

# Export all service managers
__all__ = [
    "BaseServiceManager",
    "DeploymentServiceManager",
    "KnativeServiceManager",
    "RayClusterServiceManager",
]

logger = get_logger(__name__)
