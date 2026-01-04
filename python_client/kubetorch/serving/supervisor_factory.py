from kubetorch.serving.execution_supervisor import ExecutionSupervisor
from kubetorch.serving.monarch_supervisor import MonarchDistributed
from kubetorch.serving.process_worker import ProcessWorker
from kubetorch.serving.ray_supervisor import RayDistributed
from kubetorch.serving.spmd.jax_process import JaxProcess
from kubetorch.serving.spmd.pytorch_process import PyTorchProcess
from kubetorch.serving.spmd.spmd_supervisor import SPMDDistributedSupervisor
from kubetorch.serving.spmd.tensorflow_process import TensorflowProcess


def supervisor_factory(distribution_type, *args, **kwargs):
    """
    Factory function to create an execution supervisor based on the specified type.

    Args:
        distribution_type (str): The type of supervisor to create.
            Options:
            - 'local': Local subprocess execution (no remote workers)
            - 'ray': Ray distributed (head node only)
            - 'monarch': Monarch distributed (single controller)
            - 'pytorch': PyTorch SPMD distributed
            - 'jax': JAX SPMD distributed
            - 'tensorflow'/'tf': TensorFlow SPMD distributed
            - 'spmd' or None: Generic SPMD distributed

        *args: Positional arguments to pass to the supervisor constructor.
        **kwargs: Keyword arguments to pass to the supervisor constructor.
                 Common kwargs include:
                 - restart_procs: Whether to restart processes on setup (default True)
                 - max_threads_per_proc: Max threads per subprocess (default 10)
                 - quorum_timeout: Timeout for workers to become ready (default 300s)
                 - quorum_workers: Number of workers to wait for

    Returns:
        ExecutionSupervisor: An instance of the specified supervisor.
    """
    # Local execution - subprocess isolation without remote workers
    if distribution_type == "local":
        return ExecutionSupervisor(*args, **kwargs)

    # Single-controller frameworks (manage their own cluster membership)
    if distribution_type == "ray":
        return RayDistributed(*args, **kwargs)
    elif distribution_type == "monarch":
        return MonarchDistributed(*args, **kwargs)

    # SPMD frameworks with different process classes
    if distribution_type == "pytorch":
        return SPMDDistributedSupervisor(process_class=PyTorchProcess, *args, **kwargs)
    elif distribution_type == "jax":
        return SPMDDistributedSupervisor(process_class=JaxProcess, *args, **kwargs)
    elif distribution_type == "tensorflow" or distribution_type == "tf":
        return SPMDDistributedSupervisor(process_class=TensorflowProcess, *args, **kwargs)
    elif distribution_type is None or distribution_type == "spmd":
        # Default SPMD with base ProcessWorker - no framework-specific dependencies
        return SPMDDistributedSupervisor(process_class=ProcessWorker, *args, **kwargs)
    else:
        raise ValueError(f"Unsupported distribution type: {distribution_type}")
