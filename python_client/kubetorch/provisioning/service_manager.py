# Backward compatibility imports - all service manager functionality is now in separate files
from kubetorch.logger import get_logger

# Import all service managers for backward compatibility and centralized access
from kubetorch.provisioning.base_service_manager import BaseServiceManager
from kubetorch.provisioning.deployment_service_manager import DeploymentServiceManager
from kubetorch.provisioning.knative_service_manager import KnativeServiceManager
from kubetorch.provisioning.raycluster_service_manager import RayClusterServiceManager
from kubetorch.provisioning.trainjob_service_manager import TrainJobServiceManager

# Export all service managers
__all__ = [
    "BaseServiceManager",
    "DeploymentServiceManager",
    "KnativeServiceManager",
    "RayClusterServiceManager",
    "TrainJobServiceManager",
]

logger = get_logger(__name__)
