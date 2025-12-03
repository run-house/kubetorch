"""PyTorch distributed backend."""

from kubetorch.distributed.base import DistributedProcess
from kubetorch.logger import get_logger

logger = get_logger(__name__)


class PyTorchProcess(DistributedProcess):
    """PyTorch-specific distributed process."""

    def proc_cleanup(self):
        import torch.distributed as dist

        try:
            dist.destroy_process_group()
            logger.info("Destroyed PyTorch process group.")
        except Exception:
            logger.info("Failed to destroy PyTorch process group, it may not have been initialized: {e}")
            pass
        # Call parent cleanup for debugging sessions
        super().proc_cleanup()

    @classmethod
    def get_distributed_env_vars(cls, worker_ips, node_rank, local_rank, num_local_procs, **settings):
        """Get PyTorch-specific distributed environment variables."""
        port = settings.get("port") or 12345
        env_vars = super().get_distributed_env_vars(worker_ips, node_rank, local_rank, num_local_procs, **settings)
        env_vars.update(
            {
                "MASTER_ADDR": worker_ips[0],
                "MASTER_PORT": str(port),
            }
        )
        return env_vars

    @classmethod
    def get_auto_num_processes(cls):
        """Auto-detect based on GPU availability for PyTorch."""
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.device_count()
        except ImportError:
            pass
        return 1  # Could use os.cpu_count() for CPU-only training
