from kubetorch.serving.monarch_supervisor import MonarchDistributed
from kubetorch.serving.process_worker import ProcessWorker
from kubetorch.serving.ray_supervisor import RayDistributed
from kubetorch.serving.spmd.jax_process import JaxProcess
from kubetorch.serving.spmd.pytorch_process import PyTorchProcess
from kubetorch.serving.spmd.spmd_supervisor import SPMDDistributedSupervisor
from kubetorch.serving.spmd.tensorflow_process import TensorflowProcess


def supervisor_factory(distribution_type, *args, **kwargs):
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
        ExecutionSupervisor: An instance of the specified distributed supervisor.
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
        # Default to base ProcessWorker - no framework-specific dependencies
        return SPMDDistributedSupervisor(process_class=ProcessWorker, *args, **kwargs)
    else:
        raise ValueError(f"Unsupported distributed type: {distribution_type}")
