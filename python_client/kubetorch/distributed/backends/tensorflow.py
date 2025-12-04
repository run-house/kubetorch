"""TensorFlow distributed backend."""

from kubetorch.distributed.base import DistributedProcess
from kubetorch.logger import get_logger

logger = get_logger(__name__)


class TensorflowProcess(DistributedProcess):
    """TensorFlow-specific distributed process."""

    def proc_cleanup(self):
        """TensorFlow-specific cleanup."""
        try:
            import tensorflow as tf

            # Clear the default graph and reset the session
            tf.keras.backend.clear_session()
            logger.info("TensorFlow process cleanup completed.")
        except ImportError:
            logger.info("TensorFlow not available for cleanup")
        except Exception as e:
            logger.info(f"Failed during TensorFlow cleanup: {e}")
        # Call parent cleanup for debugging sessions
        super().proc_cleanup()

    @classmethod
    def get_distributed_env_vars(cls, worker_ips, node_rank, local_rank, num_local_procs, **settings):
        """Get TensorFlow-specific distributed environment variables.

        TensorFlow uses TF_CONFIG for distributed training configuration.
        """
        import json

        port = settings.get("port") or 2222  # TensorFlow default port
        env_vars = super().get_distributed_env_vars(worker_ips, node_rank, local_rank, num_local_procs, **settings)

        # Build TF_CONFIG for MultiWorkerMirroredStrategy
        worker_addresses = [f"{ip}:{port}" for ip in worker_ips]

        tf_config = {
            "cluster": {"worker": worker_addresses},
            "task": {"type": "worker", "index": node_rank},
        }

        env_vars.update(
            {
                "TF_CONFIG": json.dumps(tf_config),
                # Additional TF env vars for performance
                "TF_FORCE_GPU_ALLOW_GROWTH": "true",
                "TF_GPU_THREAD_MODE": "gpu_private",
            }
        )
        return env_vars

    @classmethod
    def get_auto_num_processes(cls):
        """Auto-detect based on available GPUs for TensorFlow."""
        try:
            import tensorflow as tf

            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                return len(gpus)
        except Exception:
            pass
        return 1
