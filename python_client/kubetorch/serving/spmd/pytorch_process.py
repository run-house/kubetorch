from kubetorch.serving.http_server import logger
from kubetorch.serving.process_worker import ProcessWorker


class PyTorchProcess(ProcessWorker):
    """PyTorch-specific distributed process."""

    def framework_cleanup(self):
        """Clean up PyTorch process group for reloads."""
        import torch.distributed as dist

        try:
            dist.destroy_process_group()
            logger.info("Destroyed PyTorch process group.")
        except Exception as e:
            logger.debug(f"Failed to destroy PyTorch process group, it may not have been initialized: {e}")

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
