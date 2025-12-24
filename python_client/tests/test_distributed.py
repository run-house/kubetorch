import asyncio

import kubetorch as kt
import pytest

from kubetorch.utils import capture_stdout

from .assets.test_distributed.distributed_test_class import DistributedTestClass

# Import test functions and classes from assets
from .assets.test_distributed.distributed_test_functions import (
    adaptive_ray_fn_with_bs4,
    jax_distributed_fn,
    pytorch_distributed_fn,
    raise_test_exception,
    tensorflow_distributed_fn,
    verify_distributed_env,
)
from .utils import get_test_fn_name


# ============================================================================
# Tests for generic SPMD (no specific framework)
# ============================================================================


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_spmd_distributed_fn():
    """Test generic SPMD distributed with function."""
    compute = kt.Compute(cpus="0.5", memory="512Mi").distribute(workers=2, num_proc=2)
    remote_fn = kt.fn(verify_distributed_env, name=get_test_fn_name()).to(compute)

    # Test with stream_logs=True and verify logs from subprocesses appear in stdout
    out = ""
    with capture_stdout() as stdout:
        results = remote_fn(stream_logs=True)
        await asyncio.sleep(4)  # wait for logs to stream from subprocesses via queue
        out = out + str(stdout)

    assert len(results) == 4  # 2 workers * 2 processes

    # Verify log streaming works for distributed functions
    # We should see output from all 4 ranks (2 workers * 2 processes)
    for rank in range(4):
        assert f"DISTRIBUTED_PRINT rank={rank}" in out, f"Missing print output for rank {rank}"

    # Sort results by rank since they may come back in any order
    results = sorted(results, key=lambda r: int(r["rank"]))

    # Verify environment variables are set correctly
    for i, result in enumerate(results):
        assert result["rank"] == str(i)
        assert result["world_size"] == "4"
        assert result["local_rank"] in ["0", "1"]
        assert result["node_rank"] in ["0", "1"]
        assert result["pod_ips"] is not None

    # Test workers="any" - returns results from all local processes on coordinator only
    results_any = remote_fn(workers="any", stream_logs=True)
    assert len(results_any) == 2  # num_proc=2 on coordinator
    for result in results_any:
        assert result["rank"] in ["0", "1"]  # Only coordinator's processes
        assert result["world_size"] == "4"
        assert result["local_rank"] in ["0", "1"]
        assert result["node_rank"] == "0"  # Coordinator is always node 0
        assert result["pod_ips"] is not None

    # Test with worker indices (indices refer to worker nodes, not ranks)
    results_idx = remote_fn(workers=[0], stream_logs=True)  # Select first worker node only
    assert len(results_idx) == 2  # 2 processes on worker 0
    for result in results_idx:
        assert int(result["rank"]) in [0, 1]  # Worker 0 has ranks 0 and 1
        assert result["node_rank"] == "0"

    # Test with string indices
    results_str_idx = remote_fn(workers=["1"], stream_logs=True)  # Select second worker node
    assert len(results_str_idx) == 2  # 2 processes on worker 1
    for result in results_str_idx:
        assert int(result["rank"]) in [2, 3]  # Worker 1 has ranks 2 and 3
        assert result["node_rank"] == "1"

    # Test with both worker nodes
    results_both = remote_fn(workers=[0, 1], stream_logs=True)  # Select both worker nodes
    assert len(results_both) == 4  # All 4 processes
    ranks = [int(r["rank"]) for r in results_both]
    assert sorted(ranks) == [0, 1, 2, 3]


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_spmd_distributed_cls():
    """Test generic SPMD distributed with class."""
    remote_cls = kt.cls(DistributedTestClass, name=get_test_fn_name()).to(
        kt.Compute(cpus="0.5", memory="512Mi").distribute(workers=2, num_proc=2)
    )

    # Test with stream_logs=True and verify logs from subprocesses appear in stdout
    out = ""
    with capture_stdout() as stdout:
        first_call = remote_cls.increment_and_return(stream_logs=True)
        await asyncio.sleep(4)  # wait for logs to stream from subprocesses via queue
        out = out + str(stdout)

    assert len(first_call) == 4
    for result in first_call:
        assert result["call_count"] == 1
        assert result["world_size"] == "4"

    # Verify log streaming works for distributed class methods
    # We should see output from all 4 ranks (2 workers * 2 processes)
    for rank in range(4):
        assert f"DISTRIBUTED_CLS_PRINT rank={rank}" in out, f"Missing print output for rank {rank}"

    second_call = remote_cls.increment_and_return(stream_logs=True)
    assert len(second_call) == 4
    for result in second_call:
        assert result["call_count"] == 2

    # Test that redeploying with .to() resets state
    compute = kt.Compute(cpus="0.5", memory="512Mi").distribute(workers=2, num_proc=2)
    remote_cls_redeployed = kt.cls(DistributedTestClass, name=get_test_fn_name()).to(compute)

    # State should be reset after redeployment
    reset_call = remote_cls_redeployed.increment_and_return(stream_logs=True)
    assert len(reset_call) == 4
    for result in reset_call:
        assert result["call_count"] == 1  # Should be 1 again, not 3

    # Test parallel calls to verify multithreading within worker processes
    import concurrent.futures
    import time

    def make_concurrent_slow_call():
        return remote_cls_redeployed.slow_increment_with_timing(delay=2, stream_logs=True)

    # Make 3 concurrent calls with 2-second delays - this should test multithreading
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        test_start_time = time.time()
        futures = [executor.submit(make_concurrent_slow_call) for _ in range(3)]
        parallel_results = [future.result() for future in concurrent.futures.as_completed(futures)]
        test_end_time = time.time()

    # All 3 calls should have completed
    assert len(parallel_results) == 3

    # Each call should return 4 results (2 workers * 2 processes each)
    for call_result in parallel_results:
        assert len(call_result) == 4

    # Collect all execution times to verify overlap
    all_executions = []
    for call_result in parallel_results:
        for process_result in call_result:
            all_executions.append(
                {
                    "rank": process_result["rank"],
                    "thread_id": process_result["thread_id"],
                    "start_time": process_result["start_time"],
                    "end_time": process_result["end_time"],
                    "call_count": process_result["call_count"],
                }
            )

    # Sort executions by start time to analyze overlap
    all_executions.sort(key=lambda x: x["start_time"])

    # Verify that some executions overlapped (started before others finished)
    overlaps_found = 0
    for i in range(len(all_executions)):
        for j in range(i + 1, len(all_executions)):
            exec_a = all_executions[i]
            exec_b = all_executions[j]
            # Check if exec_b started before exec_a finished (overlap)
            if exec_b["start_time"] < exec_a["end_time"]:
                overlaps_found += 1
        # Print the times, threads, and ranks because helps with debugging
        print(
            f"Start: {all_executions[i]['start_time']}, end: {all_executions[i]['end_time']}, duration: {all_executions[i]['end_time'] - all_executions[i]['start_time']}, rank: {all_executions[i]['rank']}, thread: {all_executions[i]['thread_id']}"
        )

    # We should find multiple overlapping executions if multithreading is working
    assert (
        overlaps_found > 0
    ), f"No overlapping executions found - multithreading may not be working. All executions: {all_executions}"

    # Total execution time should be much less than 3 * 2 = 6 seconds if running in parallel
    total_execution_time = test_end_time - test_start_time
    assert total_execution_time < 4, f"Execution took {total_execution_time}s, expected < 4s for parallel execution"

    # Verify different thread IDs were used (proves multithreading)
    thread_ids = {exec["thread_id"] for exec in all_executions}
    assert len(thread_ids) > 1, f"Expected multiple thread IDs, only got: {thread_ids}"

    # Test worker selection with indices (selecting worker node 1 only)
    results_subset = remote_cls_redeployed.increment_and_return(workers=[1], stream_logs=True)
    assert len(results_subset) == 2  # 2 processes on worker 1
    ranks = [int(r["rank"]) for r in results_subset]
    assert all(rank in [2, 3] for rank in ranks)  # Worker 1 has ranks 2 and 3

    # Test workers="any" - returns results from all local processes on coordinator only
    results_any = remote_cls_redeployed.increment_and_return(workers="any", stream_logs=True)
    assert len(results_any) == 2  # num_proc=2 on coordinator
    for result in results_any:
        assert result["rank"] in ["0", "1"]  # Only coordinator's processes
        assert result["call_count"] >= 1

    # Test invalid worker index (should raise error)
    with pytest.raises(Exception, match="Worker index 10 out of range"):
        remote_cls_redeployed.increment_and_return(workers=[10], stream_logs=True)  # Only have workers 0 and 1

    # Test invalid worker specification
    with pytest.raises(Exception, match="Invalid worker specification"):
        remote_cls_redeployed.increment_and_return(workers=["not-an-index-or-ip"], stream_logs=True)

    # Test exception handling
    with pytest.raises(Exception, match="Test exception") as exc_info:
        remote_cls.raise_exception("Test exception from SPMD", stream_logs=True)

    assert exc_info.value.remote_traceback is not None


# ============================================================================
# Tests for PyTorch distributed
# ============================================================================


@pytest.mark.level("minimal")
def test_pytorch_distributed_fn():
    """Test PyTorch distributed with function."""
    remote_fn = kt.fn(pytorch_distributed_fn, name=get_test_fn_name()).to(
        kt.Compute(
            cpus="0.5",
            memory="1Gi",
            image=kt.images.Debian().run_bash(
                "uv pip install --system numpy torch --index-url https://download.pytorch.org/whl/cpu"
            ),
        ).distribute("pytorch", workers=2, num_proc=2)
    )

    results = remote_fn()
    assert len(results) == 4  # 2 workers * 2 processes

    # Sort results by rank since they may come back in any order
    results = sorted(results, key=lambda r: int(r["rank"]))

    # Verify PyTorch-specific environment and initialization
    for i, result in enumerate(results):
        assert result["master_addr"] is not None
        assert result["master_port"] == "12345"  # Default port
        assert result["rank"] == str(i)
        assert result["world_size"] == "4"

        # Check if PyTorch distributed was initialized successfully
        if "pytorch_initialized" in result:
            if result["pytorch_initialized"]:
                assert result["backend"] == "gloo"  # CPU backend
                # all_reduce should sum all ranks: 0+1+2+3 = 6
                assert result["all_reduce_result"] == 6.0


@pytest.mark.level("minimal")
def test_pytorch_distributed_cls():
    """Test PyTorch distributed with class."""
    remote_cls = kt.cls(DistributedTestClass, name=get_test_fn_name()).to(
        kt.Compute(
            cpus="0.5",
            memory="1Gi",
            image=kt.images.Debian().run_bash(
                "uv pip install --system numpy torch --index-url https://download.pytorch.org/whl/cpu"
            ),
        ).distribute("pytorch", workers=2, num_proc=2)
    )

    # Verify framework-specific setup
    results = remote_cls.verify_framework("pytorch")
    assert len(results) == 4

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
def test_jax_distributed_fn():
    """Test JAX distributed with function."""
    remote_fn = kt.fn(jax_distributed_fn, name=get_test_fn_name()).to(
        kt.Compute(
            cpus="0.5",
            memory="1Gi",
            image=kt.images.Debian().run_bash("uv pip install --system jax jaxlib"),
        ).distribute("jax", workers=2, num_proc=2)
    )

    results = remote_fn()
    assert len(results) == 4  # 2 workers * 2 processes

    # Sort results by process_id since they may come back in any order
    results = sorted(results, key=lambda r: int(r["process_id"]))

    # Verify JAX-specific environment
    for i, result in enumerate(results):
        assert result["coordinator_address"] is not None
        assert result["process_id"] == str(i)
        assert result["num_processes"] == "4"
        assert result["local_device_ids"] in ["0", "1"]

        # Check if JAX was initialized successfully
        if result.get("jax_initialized"):
            assert result["process_index"] == i
            assert result["process_count"] == 4


@pytest.mark.level("minimal")
def test_jax_distributed_cls():
    """Test JAX distributed with class."""
    remote_cls = kt.cls(DistributedTestClass, name=get_test_fn_name()).to(
        kt.Compute(
            cpus="0.5",
            memory="1Gi",
            image=kt.images.Debian().run_bash("uv pip install --system jax jaxlib"),
        ).distribute("jax", workers=2, num_proc=2)
    )

    # Verify framework-specific setup
    results = remote_cls.verify_framework("jax")
    assert len(results) == 4

    for result in results:
        assert result["coordinator_address"] is not None
        assert result["num_processes"] == "4"

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
def test_tensorflow_distributed_fn():
    """Test TensorFlow distributed with function."""
    remote_fn = kt.fn(tensorflow_distributed_fn, name=get_test_fn_name()).to(
        kt.Compute(
            cpus="0.5",
            memory="1Gi",
            image=kt.images.Debian().run_bash("uv pip install --system tensorflow-cpu"),
        ).distribute("tensorflow", workers=2, num_proc=2)
    )

    results = remote_fn()
    assert len(results) == 4  # 2 workers * 2 processes

    # Verify TensorFlow-specific environment
    for result in results:
        assert result["tf_config"] is not None
        tf_config = result["tf_config"]
        assert "cluster" in tf_config
        assert "task" in tf_config
        assert tf_config["task"]["type"] == "worker"

        # Verify cluster configuration
        workers = tf_config["cluster"]["worker"]
        assert len(workers) == 2  # 2 worker nodes
        for worker in workers:
            assert ":2222" in worker  # Default TF port


@pytest.mark.level("minimal")
def test_tensorflow_distributed_cls():
    """Test TensorFlow distributed with class."""
    remote_cls = kt.cls(DistributedTestClass, name=get_test_fn_name()).to(
        kt.Compute(
            cpus="1",
            memory="2Gi",
            image=kt.images.Debian().run_bash("uv pip install --system tensorflow-cpu"),
            launch_timeout=600,
        ).distribute(
            "tf", workers=2, num_proc=2
        )  # Test "tf" alias
    )

    # Verify framework-specific setup
    results = remote_cls.verify_framework("tensorflow")
    assert len(results) == 4

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
def test_distributed_exception_handling():
    """Test that exceptions are properly propagated from all workers."""
    remote_fn = kt.fn(raise_test_exception, name=get_test_fn_name()).to(
        kt.Compute(cpus="1", memory="1Gi", launch_timeout=450).distribute(workers=2, num_proc=2)
    )

    with pytest.raises(ValueError, match="Test exception from distributed worker") as exc:
        remote_fn()

    # Verify we get proper traceback
    assert exc.value.remote_traceback is not None
    assert "ValueError" in exc.value.remote_traceback


@pytest.mark.asyncio
@pytest.mark.level("minimal")
async def test_mixed_distribution_types():
    """Test that we can use different distribution types in the same test."""

    # Create different distributed configurations
    configs = [
        (None, "spmd"),  # Generic SPMD
        ("pytorch", "pytorch"),
        ("jax", "jax"),
        ("tensorflow", "tensorflow"),
    ]

    # Run all distributed configs concurrently
    async def launch_distributed_fns(dist_type, expected_name):
        compute = kt.Compute(cpus="1", memory="1Gi", launch_timeout=450)

        if dist_type:
            compute = compute.distribute(dist_type, workers=1, num_proc=2)
        else:
            compute = compute.distribute(workers=1, num_proc=2)

        remote_fn = await kt.fn(verify_distributed_env, name=f"{get_test_fn_name()}_{expected_name}").to_async(compute)
        return remote_fn

    import asyncio

    # Use asyncio.gather directly instead of asyncio.run
    remote_fns = await asyncio.gather(
        *[launch_distributed_fns(dist_type, expected_name) for dist_type, expected_name in configs]
    )

    for remote_fn in remote_fns:
        results = remote_fn()
        assert len(results) == 2  # 1 worker * 2 processes

        # Sort results by rank for consistent ordering
        results = sorted(results, key=lambda r: int(r["rank"]))

        # All should have basic env vars set
        for result in results:
            assert result["rank"] is not None
            assert result["world_size"] == "2"


# ============================================================================
# Tests for Ray distributed (existing, keep for compatibility)
# ============================================================================


@pytest.mark.skip("Compute requirements are too high for CI.")
@pytest.mark.level("minimal")
def test_ray_distributed_fn():
    """Test Ray distributed functionality."""
    from .assets.ray_tune.ray_tune_hpo import ray_tune_hpo

    ray_compute = kt.Compute(cpus="2", memory="3Gi", image=kt.Image(image_id="rayproject/ray")).distribute(
        "ray", workers=2
    )

    remote_fn = kt.fn(ray_tune_hpo, name=get_test_fn_name()).to(ray_compute)

    # Run Ray Tune HPO with small parameters for testing
    result = remote_fn(num_samples=2, max_concurrent_trials=1)

    # Verify the HPO completed successfully
    assert result["status"] == "completed"
    assert result["num_trials"] == 2
    assert "best_score" in result
    assert "best_config" in result


@pytest.mark.level("minimal")
def test_ray_package_iteration_flow():
    """Test Ray iteration flow - updating package dependencies.

    This test validates the end-to-end iteration workflow:
    1. Create Ray cluster with 2 workers, no additional dependencies
    2. Send function that tests for 'beautifulsoup4' package (should not be available)
    3. Update the image to include 'beautifulsoup4' package
    4. Send same function again with same service name
    5. Verify the new package is installed and available on all workers
    """

    # Step 1: Create Ray compute with 2 workers, no additional dependencies
    ray_compute = kt.Compute(cpus="1", memory="2Gi", image=kt.Image(image_id="rayproject/ray")).distribute(
        "ray", workers=2
    )

    name = f"{get_test_fn_name()}"

    # Deploy the initial function
    remote_fn = kt.fn(adaptive_ray_fn_with_bs4, name=name).to(ray_compute)

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
        image=kt.Image(image_id="rayproject/ray").pip_install(["beautifulsoup4"]),
    ).distribute("ray", workers=2)

    # Step 4: Deploy the SAME function with SAME service name but updated image
    updated_remote_fn = kt.fn(adaptive_ray_fn_with_bs4, name=name).to(ray_compute_with_bs4)

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
