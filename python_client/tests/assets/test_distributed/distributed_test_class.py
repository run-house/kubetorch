"""Test class for distributed framework testing."""

import os

from .distributed_test_functions import (
    jax_distributed_fn,
    pytorch_distributed_fn,
    tensorflow_distributed_fn,
    verify_distributed_env,
)


class DistributedTestClass:
    """Base class for distributed testing."""

    def __init__(self):
        self.call_count = 0

    def increment_and_return(self):
        """Test method to verify multiple calls work correctly."""
        import logging

        logger = logging.getLogger(__name__)

        self.call_count += 1
        rank = os.environ.get("RANK", "unknown")

        # Test logging and print output for log streaming verification
        print(f"DISTRIBUTED_CLS_PRINT rank={rank} call_count={self.call_count}")
        logger.info(f"DISTRIBUTED_CLS_LOG rank={rank} call_count={self.call_count}")

        return {
            "call_count": self.call_count,
            "rank": rank,
            "world_size": os.environ.get("WORLD_SIZE", "unknown"),
        }

    def slow_increment_with_timing(self, delay=2):
        """Test method that takes time and records execution timing."""
        import threading
        import time

        start_time = time.time()

        # Simulate some work
        time.sleep(delay)

        self.call_count += 1
        end_time = time.time()

        return {
            "call_count": self.call_count,
            "rank": os.environ.get("RANK", "unknown"),
            "thread_id": threading.current_thread().ident,
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
        }

    def get_env_info(self):
        """Return environment information."""
        return verify_distributed_env()

    def raise_exception(self, message="Test exception"):
        """Raise an exception with custom message."""
        raise RuntimeError(message)

    def verify_framework(self, framework_type):
        """Verify specific framework is properly initialized."""
        if framework_type == "pytorch":
            return pytorch_distributed_fn()
        elif framework_type == "jax":
            return jax_distributed_fn()
        elif framework_type == "tensorflow":
            return tensorflow_distributed_fn()
        else:
            return verify_distributed_env()
