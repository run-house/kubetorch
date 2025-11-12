from kubetorch.globals import config, MetricsConfig  # noqa: F401
from kubetorch.resources.callables.cls.cls import Cls, cls  # noqa: F401
from kubetorch.resources.callables.fn.fn import Fn, fn  # noqa: F401
from kubetorch.resources.compute.app import App, app  # noqa: F401
from kubetorch.resources.compute.compute import Compute  # noqa: F401
from kubetorch.resources.compute.decorators import async_, autoscale, compute, distribute  # noqa: F401
from kubetorch.resources.compute.utils import (
    ImagePullError,
    KnativeServiceConflictError,
    PodContainerError,
    QueueUnschedulableError,
    ResourceNotAvailableError,
    RsyncError,
    SecretNotFound,
    ServiceHealthError,
    ServiceTimeoutError,
    VersionMismatchError,
)  # noqa: F401
from kubetorch.resources.images.image import Image  # noqa: F401
from kubetorch.resources.secrets import Secret, secret  # noqa: F401
from kubetorch.resources.volumes.volume import Volume  # noqa: F401  # noqa: F401
from kubetorch.servers.http.utils import (  # noqa: F401
    deep_breakpoint,
    PodTerminatedError,
    StartupError,
    WorkerMembershipChanged,
)
from kubetorch.serving.utils import KubernetesCredentialsError

from .resources import images

# Alias to expose as kt.images
images = images

# Registry of all kubetorch exceptions for serialization/deserialization
EXCEPTION_REGISTRY = {
    "ImagePullError": ImagePullError,
    "KubernetesCredentialsError": KubernetesCredentialsError,
    "PodContainerError": PodContainerError,
    "ResourceNotAvailableError": ResourceNotAvailableError,
    "ServiceHealthError": ServiceHealthError,
    "ServiceTimeoutError": ServiceTimeoutError,
    "StartupError": StartupError,
    "PodTerminatedError": PodTerminatedError,
    "QueueUnschedulableError": QueueUnschedulableError,
    "KnativeServiceConflictError": KnativeServiceConflictError,
    "RsyncError": RsyncError,
    "VersionMismatchError": VersionMismatchError,
    "SecretNotFound": SecretNotFound,
    "WorkerMembershipChanged": WorkerMembershipChanged,
}

# Make exceptions appear to be from the main package (e.g. kubetorch.ImagePullError)
for exception in EXCEPTION_REGISTRY.values():
    exception.__module__ = "kubetorch"

__version__ = "0.2.4"
