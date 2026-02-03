import asyncio
import copy

import kubetorch as kt
import pytest

from kubetorch.utils import capture_stdout

from .assets.test_distributed.distributed_test_class import DistributedTestClass

# Import test functions and classes from assets
from .assets.test_distributed.distributed_test_functions import (
    adaptive_ray_fn_with_bs4,
    jax_distributed_fn,
    load_balanced_worker_info,
    pytorch_distributed_fn,
    raise_test_exception,
    tensorflow_distributed_fn,
    verify_distributed_env,
)
from .utils import get_test_fn_name


# ============================================================================
# Shared Compute Fixture
# ============================================================================


@pytest.fixture
def distributed_compute():
    """Shared compute fixture with replicas=3 and base Debian image.

    Each test should copy this, set call_config, and set image with needed packages.
    """
    return kt.Compute(
        cpus="0.5",
        memory="1Gi",
        replicas=3,
        image=kt.images.Debian().pip_install(
            [
                "numpy",
            ]
        ),
        launch_timeout=600,
    )


# ============================================================================
# Tests for generic SPMD (no specific framework)
# ============================================================================


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_spmd_distributed_fn(distributed_compute):
    """Test generic SPMD distributed with function."""
    compute = copy.deepcopy(distributed_compute)
    compute._call_config = kt.CallConfig(
        call_mode="spmd",
        procs=2,
    )

    remote_fn = await kt.fn(verify_distributed_env, name="distributed-tests").to_async(compute)

    # Test with stream_logs=True and verify logs from subprocesses appear in stdout
    out = ""
    with capture_stdout() as stdout:
        results = remote_fn(stream_logs=True)
        await asyncio.sleep(4)  # wait for logs to stream from subprocesses via queue
        out = out + str(stdout)

    assert len(results) == 6  # 3 workers * 2 processes

    # Verify log streaming works for distributed functions
    # We should see output from all 6 ranks (3 workers * 2 processes)
    for rank in range(6):
        assert f"DISTRIBUTED_PRINT rank={rank}" in out, f"Missing print output for rank {rank}"

    # Sort results by rank since they may come back in any order
    results = sorted(results, key=lambda r: int(r["rank"]))

    # Verify environment variables are set correctly
    for i, result in enumerate(results):
        assert result["rank"] == str(i)
        assert result["world_size"] == "6"
        assert result["local_rank"] in ["0", "1"]
        assert result["node_rank"] in ["0", "1", "2"]
        assert result["pod_ips"] is not None

    # Test workers="any" - returns results from all local processes on coordinator only
    results_any = remote_fn(workers="any", stream_logs=True)
    assert len(results_any) == 2  # num_proc=2 on coordinator
    for result in results_any:
        assert result["rank"] in ["0", "1"]  # Only coordinator's processes
        assert result["world_size"] == "6"
        assert result["local_rank"] in ["0", "1"]
        assert result["node_rank"] == "0"  # Coordinator is always node 0
        assert result["pod_ips"] is not None

    # Test with worker indices (indices refer to worker nodes, not ranks)
    results_idx = remote_fn(workers=[0], stream_logs=True)  # Select first worker node only
    assert len(results_idx) == 2  # 2 processes on worker 0
    for result in results_idx:
        assert int(result["rank"]) in [0, 1]  # Worker 0 has ranks 0 and 1
        assert result["node_rank"] == "0"


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_spmd_distributed_cls(distributed_compute):
    """Test generic SPMD distributed with class."""
    compute = copy.deepcopy(distributed_compute)
    compute._call_config = kt.CallConfig(
        call_mode="spmd",
        procs=2,
    )

    remote_cls = await kt.cls(DistributedTestClass, name="distributed-tests").to_async(compute)

    # Test with stream_logs=True and verify logs from subprocesses appear in stdout
    out = ""
    with capture_stdout() as stdout:
        first_call = remote_cls.increment_and_return(stream_logs=True)
        await asyncio.sleep(4)  # wait for logs to stream from subprocesses via queue
        out = out + str(stdout)

    assert len(first_call) == 6  # 3 workers * 2 processes
    for result in first_call:
        assert result["call_count"] == 1
        assert result["world_size"] == "6"

    # Verify log streaming works for distributed class methods
    for rank in range(6):
        assert f"DISTRIBUTED_CLS_PRINT rank={rank}" in out, f"Missing print output for rank {rank}"

    second_call = remote_cls.increment_and_return(stream_logs=True)
    assert len(second_call) == 6
    for result in second_call:
        assert result["call_count"] == 2

    # Test parallel calls to verify multithreading within worker processes
    import concurrent.futures
    import time

    def make_concurrent_slow_call():
        return remote_cls.slow_increment_with_timing(delay=2, stream_logs=True)

    # Make 3 concurrent calls with 2-second delays - this should test multithreading
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        test_start_time = time.time()
        futures = [executor.submit(make_concurrent_slow_call) for _ in range(3)]
        parallel_results = [future.result() for future in concurrent.futures.as_completed(futures)]
        test_end_time = time.time()

    # All 3 calls should have completed
    assert len(parallel_results) == 3

    # Each call should return 6 results (3 workers * 2 processes each)
    for call_result in parallel_results:
        assert len(call_result) == 6

    # Total execution time should be much less than 3 * 2 = 6 seconds if running in parallel
    total_execution_time = test_end_time - test_start_time
    assert total_execution_time < 5, f"Execution took {total_execution_time}s, expected < 5s for parallel execution"

    # Test exception handling
    with pytest.raises(Exception, match="Test exception") as exc_info:
        remote_cls.raise_exception("Test exception from SPMD")

    assert exc_info.value.remote_traceback is not None


# ============================================================================
# Tests for Load-Balanced mode
# ============================================================================


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_load_balanced_distribution(distributed_compute):
    """Test load-balanced mode distributes calls across all workers."""
    compute = copy.deepcopy(distributed_compute)
    compute._call_config = kt.CallConfig(
        call_mode="load-balanced",
        concurrency=10,
    )

    remote_fn = await kt.fn(load_balanced_worker_info, name="distributed-tests").to_async(compute)

    # Make 50 concurrent calls using asyncio.gather
    num_calls = 50
    tasks = [remote_fn(sleep_time=0.1, async_=True) for _ in range(num_calls)]
    results = await asyncio.gather(*tasks)

    assert len(results) == num_calls

    # Verify calls went to all 3 replicas
    pod_names = set(r["pod_name"] for r in results)
    pod_ips = set(r["pod_ip"] for r in results)

    # All 3 replicas should have received at least some calls
    assert len(pod_names) == 3, f"Expected calls to go to all 3 replicas, but only got: {pod_names}"
    assert len(pod_ips) == 3, f"Expected 3 unique pod IPs, but got: {pod_ips}"

    # Verify roughly even distribution (each should get ~16-17 calls, allow variance)
    pod_call_counts = {}
    for result in results:
        pod = result["pod_name"]
        pod_call_counts[pod] = pod_call_counts.get(pod, 0) + 1

    print(f"Load-balanced distribution: {pod_call_counts}")

    # Each pod should get at least 10 calls (out of 50 with 3 pods)
    for pod, count in pod_call_counts.items():
        assert count >= 10, f"Pod {pod} only received {count} calls, expected at least 10"


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_load_balanced_concurrency_limit(distributed_compute):
    """Test load-balanced mode respects concurrency limits."""
    import time

    compute = copy.deepcopy(distributed_compute)
    compute._call_config = kt.CallConfig(
        call_mode="load-balanced",
        concurrency=2,  # Low concurrency to test queueing
    )

    remote_fn = await kt.fn(load_balanced_worker_info, name="distributed-tests").to_async(compute)

    # Make calls that take some time to verify concurrency limiting
    num_calls = 12  # More than 3 workers * 2 concurrency = 6 max concurrent

    start_time = time.time()
    tasks = [remote_fn(sleep_time=0.5, async_=True) for _ in range(num_calls)]
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start_time

    assert len(results) == num_calls

    # With concurrency=2 per worker and 3 workers, max 6 concurrent calls
    # 12 calls with 0.5s each should take at least 1.0s (two batches)
    # but less than 6s (if fully sequential)
    assert elapsed >= 0.9, f"Calls completed too fast ({elapsed}s), concurrency limit may not be working"
    assert elapsed < 4.0, f"Calls took too long ({elapsed}s), load balancing may not be working"

    # Verify all pods were used
    pod_names = set(r["pod_name"] for r in results)
    assert len(pod_names) == 3, f"Expected all 3 replicas to be used, got: {pod_names}"


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_load_balanced_single_result(distributed_compute):
    """Test load-balanced mode returns single result (not list like SPMD)."""
    compute = copy.deepcopy(distributed_compute)
    compute._call_config = kt.CallConfig(
        call_mode="load-balanced",
        concurrency=10,
    )

    remote_fn = await kt.fn(load_balanced_worker_info, name="distributed-tests").to_async(compute)

    # Each call should return a single dict, not a list
    result = remote_fn()

    assert isinstance(result, dict), f"Expected dict result, got {type(result)}"
    assert "pod_name" in result
    assert "pod_ip" in result
    assert "hostname" in result


# ============================================================================
# Tests for PyTorch distributed
# ============================================================================


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_pytorch_distributed_fn(distributed_compute):
    """Test PyTorch distributed with function."""
    compute = copy.deepcopy(distributed_compute)
    compute.image = kt.images.Debian().pip_install(
        [
            'torch --extra-index-url "https://download.pytorch.org/whl/cpu"',
        ]
    )
    compute._call_config = kt.CallConfig(
        call_mode="pytorch",
        procs=2,
    )

    remote_fn = await kt.fn(pytorch_distributed_fn, name="distributed-tests").to_async(compute)

    results = remote_fn()
    assert len(results) == 6  # 3 workers * 2 processes

    # Sort results by rank since they may come back in any order
    results = sorted(results, key=lambda r: int(r["rank"]))

    # Verify PyTorch-specific environment and initialization
    for i, result in enumerate(results):
        assert result["master_addr"] is not None
        assert result["master_port"] == "12345"  # Default port
        assert result["rank"] == str(i)
        assert result["world_size"] == "6"

        # Check if PyTorch distributed was initialized successfully
        assert (
            result.get("pytorch_initialized") is True
        ), f"PyTorch not initialized on rank {i}: {result.get('pytorch_error', 'unknown error')}"
        assert result["backend"] == "gloo"  # CPU backend
        # all_reduce should sum all ranks: 0+1+2+3+4+5 = 15
        assert result["all_reduce_result"] == 15.0


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_pytorch_distributed_cls(distributed_compute):
    """Test PyTorch distributed with class."""
    compute = copy.deepcopy(distributed_compute)
    compute.image = kt.images.Debian().pip_install(
        [
            'torch --extra-index-url "https://download.pytorch.org/whl/cpu"',
        ]
    )
    compute._call_config = kt.CallConfig(
        call_mode="pytorch",
        procs=2,
    )

    remote_cls = await kt.cls(DistributedTestClass, name="distributed-tests").to_async(compute)

    # Verify framework-specific setup
    results = remote_cls.verify_framework("pytorch")
    assert len(results) == 6

    for result in results:
        assert result["master_addr"] is not None
        assert result["master_port"] is not None

    # Test multiple calls
    first = remote_cls.increment_and_return()
    second = remote_cls.increment_and_return()
    assert all(r["call_count"] == 1 for r in first)
    assert all(r["call_count"] == 2 for r in second)

    # Test exception handling
    with pytest.raises(Exception) as exc_info:
        remote_cls.raise_exception("PyTorch error test")
    assert "PyTorch error test" in str(exc_info.value)


# ============================================================================
# Tests for JAX distributed
# ============================================================================


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_jax_distributed_fn(distributed_compute):
    """Test JAX distributed with function."""
    compute = copy.deepcopy(distributed_compute)
    compute.image = kt.images.Debian().pip_install(["jax jaxlib"])
    compute._call_config = kt.CallConfig(
        call_mode="jax",
        procs=2,
    )

    remote_fn = await kt.fn(jax_distributed_fn, name="distributed-tests").to_async(compute)

    results = remote_fn()
    assert len(results) == 6  # 3 workers * 2 processes

    # Sort results by process_id since they may come back in any order
    results = sorted(results, key=lambda r: int(r["process_id"]))

    # Verify JAX-specific environment
    for i, result in enumerate(results):
        assert result["coordinator_address"] is not None
        assert result["process_id"] == str(i)
        assert result["num_processes"] == "6"
        assert result["local_device_ids"] in ["0", "1"]

        # Check if JAX was initialized successfully
        if result.get("jax_initialized"):
            assert result["process_index"] == i
            assert result["process_count"] == 6


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_jax_distributed_cls(distributed_compute):
    """Test JAX distributed with class."""
    compute = copy.deepcopy(distributed_compute)
    compute.image = kt.images.Debian().pip_install(["jax jaxlib"])
    compute._call_config = kt.CallConfig(
        call_mode="jax",
        procs=2,
    )

    remote_cls = await kt.cls(DistributedTestClass, name="distributed-tests").to_async(compute)

    # Verify framework-specific setup
    results = remote_cls.verify_framework("jax")
    assert len(results) == 6

    for result in results:
        assert result["coordinator_address"] is not None
        assert result["num_processes"] == "6"

    # Test state persistence
    call1 = remote_cls.increment_and_return()
    call2 = remote_cls.increment_and_return()
    call3 = remote_cls.increment_and_return()

    assert all(r["call_count"] == 1 for r in call1)
    assert all(r["call_count"] == 2 for r in call2)
    assert all(r["call_count"] == 3 for r in call3)


# ============================================================================
# Tests for TensorFlow distributed
# ============================================================================


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_tensorflow_distributed_fn(distributed_compute):
    """Test TensorFlow distributed with function."""
    compute = copy.deepcopy(distributed_compute)
    compute.image = kt.images.Debian().pip_install(["tensorflow-cpu"])
    compute._call_config = kt.CallConfig(
        call_mode="tensorflow",
        procs=2,
    )

    remote_fn = await kt.fn(tensorflow_distributed_fn, name="distributed-tests").to_async(compute)

    results = remote_fn()
    assert len(results) == 6  # 3 workers * 2 processes

    # Verify TensorFlow-specific environment
    for result in results:
        assert result["tf_config"] is not None
        tf_config = result["tf_config"]
        assert "cluster" in tf_config
        assert "task" in tf_config
        assert tf_config["task"]["type"] == "worker"

        # Verify cluster configuration
        workers = tf_config["cluster"]["worker"]
        assert len(workers) == 3  # 3 worker nodes
        for worker in workers:
            assert ":2222" in worker  # Default TF port


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_tensorflow_distributed_cls(distributed_compute):
    """Test TensorFlow distributed with class."""
    compute = copy.deepcopy(distributed_compute)
    compute.image = kt.images.Debian().pip_install(["tensorflow-cpu"])
    compute._call_config = kt.CallConfig(
        call_mode="tensorflow",
        procs=2,
    )

    remote_cls = await kt.cls(DistributedTestClass, name="distributed-tests").to_async(compute)

    # Verify framework-specific setup
    results = remote_cls.verify_framework("tensorflow")
    assert len(results) == 6

    for result in results:
        if result.get("tf_config"):
            assert result["tf_config"]["cluster"] is not None
            assert result["tf_config"]["task"] is not None

    # Test exception propagation
    with pytest.raises(Exception, match="TensorFlow test error"):
        remote_cls.raise_exception("TensorFlow test error")


# ============================================================================
# Tests for error handling and edge cases
# ============================================================================


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_distributed_exception_handling(distributed_compute):
    """Test that exceptions are properly propagated from all workers."""
    compute = copy.deepcopy(distributed_compute)
    compute._call_config = kt.CallConfig(
        call_mode="spmd",
        procs=2,
    )

    remote_fn = await kt.fn(raise_test_exception, name="distributed-tests").to_async(compute)

    with pytest.raises(ValueError, match="Test exception from distributed worker") as exc:
        remote_fn()

    # Verify we get proper traceback
    assert exc.value.remote_traceback is not None
    assert "ValueError" in exc.value.remote_traceback


# ============================================================================
# Tests for Ray distributed (existing, keep for compatibility)
# ============================================================================


@pytest.mark.skip("Compute requirements are too high for CI.")
@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_ray_distributed_fn():
    """Test Ray distributed functionality."""
    from .assets.ray_tune.ray_tune_hpo import ray_tune_hpo

    ray_compute = kt.Compute(
        cpus="2",
        memory="3Gi",
        replicas=2,
        image=kt.Image(image_id="rayproject/ray"),
        call_config=kt.CallConfig(call_mode="ray"),
    )

    remote_fn = await kt.fn(ray_tune_hpo, name=get_test_fn_name()).to_async(ray_compute)

    # Run Ray Tune HPO with small parameters for testing
    result = remote_fn(num_samples=2, max_concurrent_trials=1)

    # Verify the HPO completed successfully
    assert result["status"] == "completed"
    assert result["num_trials"] == 2
    assert "best_score" in result
    assert "best_config" in result


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_ray_package_iteration_flow():
    """Test Ray iteration flow - updating package dependencies.

    This test validates the end-to-end iteration workflow:
    1. Create Ray cluster with 2 workers, no additional dependencies
    2. Send function that tests for 'beautifulsoup4' package (should not be available)
    3. Update the image to include 'beautifulsoup4' package
    4. Send same function again with same service name
    5. Verify the new package is installed and available on all workers
    """

    # Step 1: Create Ray compute with 2 workers, no additional dependencies
    ray_compute = kt.Compute(
        cpus="1",
        memory="2Gi",
        replicas=2,
        image=kt.Image(image_id="rayproject/ray"),
        call_config=kt.CallConfig(call_mode="ray"),
    )

    name = f"{get_test_fn_name()}"

    # Deploy the initial function
    remote_fn = await kt.fn(adaptive_ray_fn_with_bs4, name=name).to_async(ray_compute)

    # Step 2: Test the initial function (should not have beautifulsoup4 package)
    result1 = remote_fn()

    # Verify the function used both workers
    assert result1["num_tasks"] == 8
    assert result1["unique_hostnames"] >= 2

    # KEY TEST: Verify beautifulsoup4 is NOT available initially
    assert result1["bs4_available"] is False, "beautifulsoup4 should not be available in base rayproject/ray image"

    # Verify basic calculations work
    expected_sum = 0 + 10 + 20 + 30 + 40 + 50 + 60 + 70  # 0*10 + 1*10 + 2*10 + 3*10 + 4*10 + 5*10 + 6*10 + 7*10
    assert result1["sum_calculations"] == expected_sum

    # Step 3: Update the image to include beautifulsoup4 package
    ray_compute_with_bs4 = kt.Compute(
        cpus="1",
        memory="2Gi",
        replicas=2,
        image=kt.Image(image_id="rayproject/ray").pip_install(["beautifulsoup4"]),
        call_config=kt.CallConfig(call_mode="ray"),
    )

    # Step 4: Deploy the SAME function with SAME service name but updated image
    updated_remote_fn = await kt.fn(adaptive_ray_fn_with_bs4, name=name).to_async(ray_compute_with_bs4)

    # Step 5: Test the updated function (should now have beautifulsoup4 package)
    result2 = updated_remote_fn()

    # Verify the updated function works
    assert result2["num_tasks"] == 8
    assert result2["unique_hostnames"] >= 2

    # Handle any execution errors
    if "error" in result2:
        assert False, f"Function execution failed: {result2['error']}"

    # KEY TEST: Verify beautifulsoup4 is NOW available on all workers
    assert result2["bs4_available"] is True, "beautifulsoup4 should be available after image update"

    # Verify all workers report having beautifulsoup4
    for task_result in result2["task_results"]:
        assert task_result["bs4_available"] is True, f"Worker {task_result['worker_id']} missing beautifulsoup4"
        assert task_result["bs4_version"] is not None, f"Worker {task_result['worker_id']} missing bs4 version"

    # Verify calculations still work
    assert result2["sum_calculations"] == expected_sum
