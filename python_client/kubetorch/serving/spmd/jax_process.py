from kubetorch.serving.process_worker import ProcessWorker


class JaxProcess(ProcessWorker):
    """JAX-specific distributed process."""

    @classmethod
    def get_distributed_env_vars(cls, worker_ips, node_rank, local_rank, num_local_procs, **settings):
        """Get JAX-specific distributed environment variables.

        JAX uses a coordinator address and process ID for distributed setup.
        """
        port = settings.get("port") or 1234  # JAX default coordinator port
        env_vars = super().get_distributed_env_vars(worker_ips, node_rank, local_rank, num_local_procs, **settings)

        # JAX distributed environment variables
        env_vars.update(
            {
                # Coordinator is the first worker
                "JAX_COORDINATOR_ADDRESS": f"{worker_ips[0]}:{port}",
                # Process ID is global rank
                "JAX_PROCESS_ID": str(node_rank * num_local_procs + local_rank),
                # Total number of processes
                "JAX_NUM_PROCESSES": str(len(worker_ips) * num_local_procs),
                # Local device IDs (for GPU/TPU)
                "JAX_LOCAL_DEVICE_IDS": str(local_rank),
            }
        )
        return env_vars

    @classmethod
    def get_auto_num_processes(cls):
        """Auto-detect based on available accelerators for JAX."""
        try:
            import jax

            # JAX can use TPUs, GPUs, or CPUs
            devices = jax.devices()
            return len(devices)
        except Exception:
            return 1

    # JAX doesn't have a global process group to destroy like PyTorch
    # Cleanup is mostly handled automatically
    # def proc_cleanup(self):
