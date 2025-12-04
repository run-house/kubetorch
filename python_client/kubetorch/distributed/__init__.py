"""
Distributed utilities for kubetorch services.

This module provides utilities for distributed operations within kubetorch services.
"""

from . import backends
from .base import DistributedProcess, DistributedSupervisor
from .http_worker_pool import RemoteWorkerPool
from .process_pool import DistributedProcessPool
from .spmd import SPMDDistributedSupervisor
from .utils import pod_ips

# Re-export backend classes via lazy loading to avoid importing heavy dependencies
__all_backends__ = [
    "PyTorchProcess",
    "JaxProcess",
    "TensorflowProcess",
    "RayProcess",
    "RayDistributed",
    "MonarchProcess",
    "MonarchDistributed",
]


def __getattr__(name):
    """Lazy load backend classes from backends submodule."""
    if name in __all_backends__:
        return getattr(backends, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    # Core utilities
    "pod_ips",
    # Factory function
    "distributed_supervisor_factory",
    # Base classes
    "DistributedSupervisor",
    "DistributedProcess",
    # Pools
    "DistributedProcessPool",
    "RemoteWorkerPool",
    # SPMD
    "SPMDDistributedSupervisor",
    # Backend module
    "backends",
] + __all_backends__  # Add lazy-loaded backend classes


def distributed_supervisor_factory(distribution_type, *args, **kwargs):
    """
    Factory function to create a distributed supervisor based on the specified type.

    Args:
        distribution_type (str): The type of distributed supervisor to create.
                                Options include 'ray', 'monarch', 'pytorch', 'jax', 'tensorflow', or None for generic SPMD.
        *args: Positional arguments to pass to the supervisor constructor.
        **kwargs: Keyword arguments to pass to the supervisor constructor.
                 Common kwargs include:
                 - quorum_timeout: Timeout in seconds for workers to become ready (default 30 for SPMD, 300 for Ray/Monarch)

    Returns:
        DistributedSupervisor: An instance of the specified distributed supervisor.
    """
    if distribution_type == "ray":
        # Ray uses its own supervisor, not SPMD
        return RayDistributed(*args, **kwargs)
    elif distribution_type == "monarch":
        # Monarch is similar to Ray - single controller framework
        return MonarchDistributed(*args, **kwargs)

    # All other types use SPMDDistributedSupervisor with different process classes
    if distribution_type == "pytorch":
        return SPMDDistributedSupervisor(process_class=PyTorchProcess, *args, **kwargs)
    elif distribution_type == "jax":
        return SPMDDistributedSupervisor(process_class=JaxProcess, *args, **kwargs)
    elif distribution_type == "tensorflow" or distribution_type == "tf":
        return SPMDDistributedSupervisor(process_class=TensorflowProcess, *args, **kwargs)
    elif distribution_type is None or distribution_type == "spmd":
        # Default to base DistributedProcess - no framework-specific dependencies
        return SPMDDistributedSupervisor(process_class=DistributedProcess, *args, **kwargs)
    else:
        raise ValueError(f"Unsupported distributed type: {distribution_type}")
