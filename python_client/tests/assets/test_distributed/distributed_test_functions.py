"""Test functions for distributed framework testing."""

import json
import os


def verify_distributed_env():
    """Generic function to verify distributed environment variables are set."""
    import logging

    logger = logging.getLogger(__name__)

    env_info = {
        "rank": os.environ.get("RANK"),
        "world_size": os.environ.get("WORLD_SIZE"),
        "local_rank": os.environ.get("LOCAL_RANK"),
        "node_rank": os.environ.get("NODE_RANK"),
        "pod_ips": os.environ.get("POD_IPS"),
    }

    # Test logging and print output for log streaming verification
    rank = env_info["rank"]
    print(f"DISTRIBUTED_PRINT rank={rank}")
    logger.info(f"DISTRIBUTED_LOG rank={rank}")

    return env_info


def pytorch_distributed_fn():
    """Test function for PyTorch distributed setup."""
    # Verify PyTorch-specific env vars
    env_info = {
        "master_addr": os.environ.get("MASTER_ADDR"),
        "master_port": os.environ.get("MASTER_PORT"),
        "rank": os.environ.get("RANK"),
        "world_size": os.environ.get("WORLD_SIZE"),
    }

    # Try to initialize PyTorch distributed if available
    try:
        import torch
        import torch.distributed as dist

        if not dist.is_initialized():
            # Use minimal init - PyTorch will read MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE from env
            dist.init_process_group(backend="gloo")  # CPU-only backend

        # Verify we can communicate
        tensor = torch.ones(1) * int(os.environ["RANK"])
        dist.all_reduce(tensor)

        env_info["pytorch_initialized"] = True
        env_info["all_reduce_result"] = tensor.item()
        env_info["backend"] = dist.get_backend()
    except Exception as e:
        env_info["pytorch_initialized"] = False
        env_info["error"] = str(e)

    return env_info


def jax_distributed_fn():
    """Test function for JAX distributed setup."""
    # Verify JAX-specific env vars
    env_info = {
        "coordinator_address": os.environ.get("JAX_COORDINATOR_ADDRESS"),
        "process_id": os.environ.get("JAX_PROCESS_ID"),
        "num_processes": os.environ.get("JAX_NUM_PROCESSES"),
        "local_device_ids": os.environ.get("JAX_LOCAL_DEVICE_IDS"),
    }

    # Try to initialize JAX distributed if available
    try:
        import jax
        import jax.numpy as jnp

        # Use minimal init - JAX will read JAX_COORDINATOR_ADDRESS, JAX_PROCESS_ID, JAX_NUM_PROCESSES from env
        jax.distributed.initialize()

        # Verify we can use JAX
        devices = jax.devices()
        local_devices = jax.local_devices()

        # Simple computation to verify JAX works
        x = jnp.ones(1) * int(os.environ["JAX_PROCESS_ID"])
        print(x)

        env_info["jax_initialized"] = True
        env_info["num_devices"] = len(devices)
        env_info["num_local_devices"] = len(local_devices)
        env_info["process_index"] = jax.process_index()
        env_info["process_count"] = jax.process_count()
    except Exception as e:
        env_info["jax_initialized"] = False
        env_info["error"] = str(e)

    return env_info


def tensorflow_distributed_fn():
    """Test function for TensorFlow distributed setup."""
    # Verify TensorFlow-specific env vars
    tf_config = os.environ.get("TF_CONFIG")
    env_info = {
        "tf_config": json.loads(tf_config) if tf_config else None,
        "rank": os.environ.get("RANK"),
        "world_size": os.environ.get("WORLD_SIZE"),
    }

    # Try to initialize TensorFlow distributed if available
    try:
        import tensorflow as tf

        # TensorFlow reads TF_CONFIG from environment automatically
        # Just create the strategy - it will use TF_CONFIG
        strategy = tf.distribute.MultiWorkerMirroredStrategy()

        # Verify we can use TensorFlow
        with strategy.scope():
            # Simple computation
            x = tf.constant([1.0]) * float(os.environ.get("NODE_RANK", 0))
            print(x)

        env_info["tensorflow_initialized"] = True
        env_info["num_replicas"] = strategy.num_replicas_in_sync

        # Also parse TF_CONFIG for verification
        if tf_config:
            config_dict = json.loads(tf_config)
            env_info["cluster_spec"] = config_dict.get("cluster", {})
            env_info["task_info"] = config_dict.get("task", {})
    except Exception as e:
        env_info["tensorflow_initialized"] = False
        env_info["error"] = str(e)

    return env_info


def raise_test_exception():
    """Function that raises an exception for testing error handling."""
    raise ValueError("Test exception from distributed worker")


def load_balanced_worker_info(sleep_time: float = 0.0):
    """Return information about which worker handled this call.

    Used for load-balanced mode testing to verify calls are distributed.
    """
    import socket
    import threading
    import time

    if sleep_time > 0:
        time.sleep(sleep_time)

    return {
        "pod_name": os.environ.get("POD_NAME", "unknown"),
        "pod_ip": os.environ.get("POD_IP", "unknown"),
        "hostname": socket.gethostname(),
        "thread_id": threading.current_thread().ident,
    }


def adaptive_ray_fn_with_bs4():
    """Ray function that tests package availability on workers."""
    try:
        import os
        import socket

        import ray

        # Initialize Ray to connect to cluster
        ray.init(address="auto")

        @ray.remote
        def worker_task(worker_id):
            """Task that tests package availability and returns info."""
            import time

            time.sleep(0.1)  # Simulate some work

            # Test if beautifulsoup4 package is available (definitely not in base rayproject/ray)
            bs4_available = False
            bs4_version = None
            try:
                import bs4

                bs4_available = True
                bs4_version = bs4.__version__
            except ImportError:
                pass

            return {
                "worker_id": worker_id,
                "hostname": socket.gethostname(),
                "worker_pid": os.getpid(),
                "bs4_available": bs4_available,
                "bs4_version": bs4_version,
                "calculation": worker_id * 10,  # Simple calculation
            }

        # Launch tasks on both workers
        tasks = []
        for i in range(8):  # Launch 4 tasks to ensure both workers are used
            task = worker_task.remote(i)
            tasks.append(task)

        # Get results from all tasks
        results = ray.get(tasks)

        # Get cluster info to verify we're using 2 workers
        cluster_resources = ray.cluster_resources()
        available_resources = ray.available_resources()

        return {
            "task_results": results,
            "cluster_resources": cluster_resources,
            "available_resources": available_resources,
            "num_tasks": len(results),
            "bs4_available": all(r["bs4_available"] for r in results),
            "unique_hostnames": len(set(r["hostname"] for r in results)),
            "sum_calculations": sum(r["calculation"] for r in results),
        }

    except Exception as e:
        return {
            "error": f"Top-level error in my_ray_fn_adaptive: {str(e)}",
            "num_tasks": 0,
            "bs4_available": False,
        }
