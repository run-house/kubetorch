"""
Tests for GPU tensor transfer via kt.put(src=tensor) and kt.get(dest=tensor).

These tests verify:
- GPU tensor publishing via put with GPU data
- GPU tensor retrieval via get with pre-allocated tensor destination (NCCL broadcast)
- Tensor value correctness after transfer

The GPU Data Server architecture:
- kt.put(src=tensor) registers tensor IPC handles with a per-node GPU server
- kt.get(dest=tensor) triggers automatic server-to-server NCCL broadcast
- No explicit "serve" step needed - transfers are automatic
"""
import asyncio

import kubetorch as kt
import pytest

from tests.assets.kv_store.gpu_helper import GPUTestHelper


@pytest.fixture(scope="session")
async def gpu_source():
    """Fixture that provides a GPU helper instance for publishing tensors (source/training worker)."""
    gpu = kt.Compute(gpus=1, memory="4Gi", image=kt.images.pytorch("23.10-py3"))
    helper_cls = await kt.cls(GPUTestHelper, name="gpu-source").to_async(gpu)

    result = helper_cls.check_gpu_available()
    # If we're rerunning with SPMD
    if isinstance(result, list):
        result = result[0]
    assert result["cuda_available"], f"GPU not available on source: {result}"
    assert result["device_count"] > 0, "No GPU devices found on source"
    return helper_cls


@pytest.fixture(scope="session")
async def gpu_consumer():
    """Fixture that provides a GPU helper instance for consuming tensors (inference worker)."""
    gpu = kt.Compute(gpus=1, memory="4Gi", image=kt.images.pytorch("23.10-py3"))
    helper_cls = await kt.cls(GPUTestHelper, name="gpu-consumer").to_async(gpu)

    result = helper_cls.check_gpu_available()
    # If we're rerunning with SPMD
    if isinstance(result, list):
        result = result[0]
    assert result["cuda_available"], f"GPU not available on consumer: {result}"
    assert result["device_count"] > 0, "No GPU devices found on consumer"

    return helper_cls


# ==================== Basic GPU Transfer Tests ====================


@pytest.mark.level("gpu")
async def test_gpu_put_registers_key(gpu_source):
    """Test that put with GPU data correctly registers a key with the metadata server."""
    service_name = gpu_source.service_name
    key = f"{service_name}/gpu-test/basic"
    shape = [1024, 1024]
    fill_value = 2.5

    result = gpu_source.publish_tensor(key=key, shape=shape, fill_value=fill_value)

    assert result["success"], f"put failed: {result.get('error')}"
    assert result["shape"] == shape
    assert result["fill_value"] == fill_value


@pytest.mark.level("gpu")
async def test_gpu_transfer_single_consumer(gpu_source, gpu_consumer):
    """
    Test GPU tensor transfer from source to a single consumer.

    Flow:
    1. Source publishes tensor via put(src=tensor)
    2. Consumer requests tensor via get(dest=tensor)
    3. GPU servers handle the NCCL broadcast automatically
    4. Verify tensor values match
    """
    service_name = gpu_source.service_name
    key = f"{service_name}/gpu-test/single-consumer"
    shape = [512, 512]
    fill_value = 3.14
    expected_sum = fill_value * shape[0] * shape[1]

    # Step 1: Source publishes tensor
    pub_result = gpu_source.publish_tensor(key=key, shape=shape, fill_value=fill_value)
    assert pub_result["success"], f"put failed: {pub_result.get('error')}"

    # Step 2: Consumer retrieves tensor - GPU servers handle NCCL automatically
    consumer_result = gpu_consumer.verify_tensor_values(
        key=key,
        expected_sum=expected_sum,
        expected_shape=shape,
    )

    # Step 3: Verify results
    assert consumer_result["success"], f"Consumer get failed: {consumer_result.get('error')}"
    assert consumer_result["all_correct"], (
        f"Tensor values don't match: shape={consumer_result['actual_shape']} "
        f"(expected {consumer_result['expected_shape']}), "
        f"sum={consumer_result['actual_sum']} (expected {consumer_result['expected_sum']})"
    )


# ==================== Edge Cases ====================


@pytest.mark.level("gpu")
async def test_gpu_transfer_different_dtypes(gpu_source, gpu_consumer):
    """Test GPU transfer with different tensor dtypes."""
    service_name = gpu_source.service_name

    for dtype in ["float32", "float16"]:
        key = f"{service_name}/gpu-test/dtype-{dtype}"
        shape = [128, 128]
        fill_value = 1.0

        pub_result = gpu_source.publish_tensor(key=key, shape=shape, fill_value=fill_value, dtype=dtype)
        assert pub_result["success"], f"put failed for {dtype}: {pub_result.get('error')}"

        consumer_result = gpu_consumer.get_tensor(key=key, shape=shape, dtype=dtype)

        assert consumer_result["success"], f"Get failed for {dtype}: {consumer_result.get('error')}"
        assert dtype in consumer_result["dtype"], f"Wrong dtype: {consumer_result['dtype']}"


@pytest.mark.level("gpu")
async def test_gpu_transfer_large_tensor(gpu_source, gpu_consumer):
    """Test GPU transfer with a larger tensor (simulating model weights)."""
    service_name = gpu_source.service_name
    key = f"{service_name}/gpu-test/large-tensor"
    # ~100MB tensor (25M floats * 4 bytes)
    shape = [5000, 5000]
    fill_value = 0.01
    expected_sum = fill_value * shape[0] * shape[1]

    pub_result = gpu_source.publish_tensor(key=key, shape=shape, fill_value=fill_value)
    assert pub_result["success"], f"put failed: {pub_result.get('error')}"

    consumer_result = gpu_consumer.get_tensor(key=key, shape=shape)

    assert consumer_result["success"], f"Get failed: {consumer_result.get('error')}"
    assert consumer_result["shape"] == shape
    # More tolerance for large tensor sum due to float precision
    assert abs(consumer_result["sum"] - expected_sum) < expected_sum * 0.01  # 1% tolerance


# ==================== GPU Server Resilience Tests ====================


@pytest.mark.level("gpu")
async def test_gpu_server_resilience_after_failure(gpu_source, gpu_consumer):
    """
    Test that the GPU Data Server remains usable after a failed transfer attempt.

    This test verifies:
    1. A successful transfer works (baseline)
    2. A failed transfer (to non-existent IP) returns an error
    3. The server can still handle valid requests after the failure
    4. If the server crashes, it can be restarted and used again

    This is important for production reliability - a single bad transfer
    should not break the server for subsequent valid requests.
    """
    service_name = gpu_source.service_name

    # Step 1: Baseline - verify normal transfer works
    key = f"{service_name}/gpu-test/resilience-baseline"
    shape = [256, 256]
    fill_value = 1.5
    expected_sum = fill_value * shape[0] * shape[1]

    pub_result = gpu_source.publish_tensor(key=key, shape=shape, fill_value=fill_value)
    assert pub_result["success"], f"Baseline put failed: {pub_result.get('error')}"

    consumer_result = gpu_consumer.get_tensor(key=key, shape=shape)
    assert consumer_result["success"], f"Baseline get failed: {consumer_result.get('error')}"
    assert abs(consumer_result["sum"] - expected_sum) < 1.0, "Baseline tensor values incorrect"

    # Step 2: Check server health before failure test
    health_before = gpu_consumer.check_gpu_server_health()
    assert health_before["healthy"], f"Server unhealthy before test: {health_before}"

    # Step 3: Trigger a failure by attempting to connect to non-existent source
    # Uses 2-second timeout for fast failure
    failure_result = gpu_consumer.trigger_nccl_timeout(nccl_timeout=1)

    # The request should fail (expected_failure=True means we expect it to fail)
    assert failure_result.get("expected_failure"), "Failure test should have failed"
    assert "error" in failure_result, "Failure should include error message"

    # Step 4: Check if server is still healthy
    health_after = gpu_consumer.check_gpu_server_health()

    if not health_after["healthy"]:
        # Server may have crashed due to NCCL corruption - restart it
        restart_result = gpu_consumer.restart_gpu_server()
        assert restart_result["success"], f"Failed to restart server: {restart_result.get('error')}"
        assert restart_result["server_running"], "Server not running after restart"

    # Step 5: Verify the server can still handle valid requests
    # Publish a new tensor (source needs to re-publish after potential restart)
    key2 = f"{service_name}/gpu-test/resilience-after"
    fill_value2 = 2.5
    expected_sum2 = fill_value2 * shape[0] * shape[1]

    pub_result2 = gpu_source.publish_tensor(key=key2, shape=shape, fill_value=fill_value2)
    assert pub_result2["success"], f"Post-failure put failed: {pub_result2.get('error')}"

    consumer_result2 = gpu_consumer.get_tensor(key=key2, shape=shape)
    assert consumer_result2["success"], f"Post-failure get failed: {consumer_result2.get('error')}"
    assert abs(consumer_result2["sum"] - expected_sum2) < 1.0, "Post-failure tensor values incorrect"


@pytest.mark.level("gpu")
async def test_gpu_transfer_state_dict(gpu_source, gpu_consumer):
    """
    Test GPU state_dict (dictionary of tensors) transfer in both modes:
    1. Point-to-point (no broadcast) - each tensor registered with full key
    2. Broadcast mode - coordinated multi-party transfer

    This test verifies:
    - Multiple tensors can be transferred as a dict (like model.state_dict())
    - Point-to-point: each tensor stored individually, batched retrieval
    - Broadcast: all tensors flow through the same NCCL session
    - Tensor values are preserved correctly in both modes
    """
    import uuid

    service_name = gpu_source.service_name

    # Define a realistic "model" state_dict (simulating a small transformer/MLP)
    # ~40 tensors, ~10MB total - similar to a small production model
    state_dict_spec = {}

    # Embedding layer: 8192 vocab x 256 dim = 8MB
    state_dict_spec["embedding.weight"] = {"shape": [8192, 256], "fill_value": 0.02}

    # 8 transformer-like layers, each with:
    # - attention weights (256x256) + bias
    # - feedforward weights (256x512, 512x256) + biases
    # - layer norm weights + biases
    for i in range(8):
        prefix = f"layers.{i}"
        state_dict_spec[f"{prefix}.attention.query.weight"] = {"shape": [256, 256], "fill_value": 0.01}
        state_dict_spec[f"{prefix}.attention.query.bias"] = {"shape": [256], "fill_value": 0.001}
        state_dict_spec[f"{prefix}.attention.key.weight"] = {"shape": [256, 256], "fill_value": 0.01}
        state_dict_spec[f"{prefix}.attention.key.bias"] = {"shape": [256], "fill_value": 0.001}
        state_dict_spec[f"{prefix}.attention.value.weight"] = {"shape": [256, 256], "fill_value": 0.01}
        state_dict_spec[f"{prefix}.attention.value.bias"] = {"shape": [256], "fill_value": 0.001}
        state_dict_spec[f"{prefix}.attention.out.weight"] = {"shape": [256, 256], "fill_value": 0.01}
        state_dict_spec[f"{prefix}.attention.out.bias"] = {"shape": [256], "fill_value": 0.001}
        state_dict_spec[f"{prefix}.ffn.up.weight"] = {"shape": [512, 256], "fill_value": 0.01}
        state_dict_spec[f"{prefix}.ffn.up.bias"] = {"shape": [512], "fill_value": 0.001}
        state_dict_spec[f"{prefix}.ffn.down.weight"] = {"shape": [256, 512], "fill_value": 0.01}
        state_dict_spec[f"{prefix}.ffn.down.bias"] = {"shape": [256], "fill_value": 0.001}
        state_dict_spec[f"{prefix}.norm1.weight"] = {"shape": [256], "fill_value": 1.0}
        state_dict_spec[f"{prefix}.norm1.bias"] = {"shape": [256], "fill_value": 0.0}
        state_dict_spec[f"{prefix}.norm2.weight"] = {"shape": [256], "fill_value": 1.0}
        state_dict_spec[f"{prefix}.norm2.bias"] = {"shape": [256], "fill_value": 0.0}

    # Output head
    state_dict_spec["output.weight"] = {"shape": [1000, 256], "fill_value": 0.01}
    state_dict_spec["output.bias"] = {"shape": [1000], "fill_value": 0.0}

    # Calculate total size for logging
    total_params = sum(
        spec["shape"][0] * (spec["shape"][1] if len(spec["shape"]) > 1 else 1) for spec in state_dict_spec.values()
    )
    total_mb = total_params * 4 / (1024 * 1024)  # float32 = 4 bytes
    print(f"State dict: {len(state_dict_spec)} tensors, {total_params:,} params, {total_mb:.1f} MB")

    # Calculate expected sums for verification
    for name, spec in state_dict_spec.items():
        numel = 1
        for dim in spec["shape"]:
            numel *= dim
        spec["expected_sum"] = spec["fill_value"] * numel
        spec["tolerance"] = abs(spec["expected_sum"]) * 0.01 + 0.1  # 1% + small absolute tolerance

    # ==================== Test 1: Point-to-point (no broadcast) ====================
    print("\n=== Testing point-to-point state dict transfer ===")
    p2p_key = f"{service_name}/state-dict-p2p-{uuid.uuid4().hex[:8]}"

    # Source publishes state dict (each tensor gets its own key: key/tensor_name)
    put_result = gpu_source.publish_state_dict(
        key=p2p_key,
        state_dict_spec=state_dict_spec,
    )
    assert put_result.get("success"), f"Point-to-point put failed: {put_result.get('error')}"
    assert put_result["num_tensors"] == len(state_dict_spec), "Wrong number of tensors published"
    print(f"Published {put_result['num_tensors']} tensors via point-to-point")

    # Consumer retrieves state dict (batched retrieval, grouped by source)
    get_result = gpu_consumer.get_state_dict(
        key=p2p_key,
        state_dict_spec=state_dict_spec,
    )
    assert get_result.get("success"), f"Point-to-point get failed: {get_result.get('error')}"
    assert get_result["num_tensors"] == len(state_dict_spec), "Wrong number of tensors received"
    assert get_result["all_correct"], f"Point-to-point values mismatch: {get_result.get('verification')}"
    print(f"Retrieved {get_result['num_tensors']} tensors via point-to-point - all values correct")

    # ==================== Test 2: Broadcast mode ====================
    print("\n=== Testing broadcast state dict transfer ===")
    group_id = f"{service_name}/state-dict-broadcast-{uuid.uuid4().hex[:8]}"

    # Create BroadcastWindow - 1 putter + 1 getter = world_size 2
    broadcast_window = kt.BroadcastWindow(
        group_id=group_id,
        world_size=2,
        timeout=30.0,
    )

    # Launch putter and getter concurrently
    put_task = gpu_source.publish_state_dict_with_broadcast(
        key=group_id,
        state_dict_spec=state_dict_spec,
        broadcast_window=broadcast_window.to_dict(),
        async_=True,
    )

    get_task = gpu_consumer.get_state_dict_with_broadcast(
        key=group_id,
        state_dict_spec=state_dict_spec,
        broadcast_window=broadcast_window.to_dict(),
        async_=True,
    )

    put_result, get_result = await asyncio.gather(put_task, get_task)

    # Verify put succeeded
    assert put_result.get("success"), f"Broadcast put failed: {put_result.get('error')}"
    assert put_result["num_tensors"] == len(state_dict_spec), "Wrong number of tensors published"
    print(f"Published {put_result['num_tensors']} tensors via broadcast")

    # Verify get succeeded
    assert get_result.get("success"), f"Broadcast get failed: {get_result.get('error')}"
    assert get_result["num_tensors"] == len(state_dict_spec), "Wrong number of tensors received"
    assert get_result["all_correct"], f"Broadcast values mismatch: {get_result.get('verification')}"
    print(f"Retrieved {get_result['num_tensors']} tensors via broadcast - all values correct")


@pytest.mark.level("gpu")
async def test_gpu_transfer_many_to_many(gpu_source, gpu_consumer):
    """
    Test many-to-many GPU tensor transfer with coordinated BroadcastWindow.

    This test verifies:
    - Multiple putters and getters can coordinate through a unified NCCL process group
    - All participants join the same BroadcastWindow group and transfer completes atomically
    - Tensors flow correctly from putters to getters based on key matching

    Architecture:
    - 2 source ranks (putters): rank 0 puts t0, rank 1 puts t1
    - 2 consumer ranks (getters): rank 0 gets t0, rank 1 gets t1
    - All 4 participants join the same BroadcastWindow group (world_size=4)
    """
    # Create SPMD-distributed source with 2 worker processes
    gpu_source.compute.distribute("pytorch", num_proc=2)
    gpu_source_task = gpu_source.to_async(gpu_source.compute)

    # Create SPMD-distributed consumer with 2 worker processes
    gpu_consumer.compute.distribute("pytorch", num_proc=2)
    gpu_consumer_task = gpu_consumer.to_async(gpu_consumer.compute)
    gpu_source, gpu_consumer = await asyncio.gather(gpu_source_task, gpu_consumer_task)

    import uuid

    service_name = gpu_source.service_name
    # Use unique group_id to avoid conflicts with previous test runs
    group_id = f"{service_name}/broadcast-group-{uuid.uuid4().hex[:8]}"

    # Define 2 tensors to transfer - each rank will extract its own based on LOCAL_RANK
    keys = [f"{group_id}/t0", f"{group_id}/t1"]
    shapes = [[128, 128], [256, 128]]
    fill_values = [1.0, 2.0]

    # Create BroadcastWindow - all 4 participants (2 putters + 2 getters) will join
    broadcast_window = kt.BroadcastWindow(
        group_id=group_id,
        world_size=4,  # 2 putters + 2 getters
        timeout=30.0,  # 30 second timeout as fallback
    )

    # Launch all 4 participants concurrently
    # Each participant calls kt.put or kt.get with the BroadcastWindow
    # Each rank extracts its own key/shape/fill_value based on LOCAL_RANK

    # Source ranks publish tensors (each rank picks its own from the lists)
    put_task = gpu_source.publish_tensor_with_broadcast(
        keys=keys,
        shapes=shapes,
        fill_values=fill_values,
        broadcast_window=broadcast_window.to_dict(),
        async_=True,
    )

    # Consumer ranks get tensors (each rank picks its own from the lists)
    get_task = gpu_consumer.get_tensor_with_broadcast(
        keys=keys,
        shapes=shapes,
        broadcast_window=broadcast_window.to_dict(),
        async_=True,
    )

    # Wait for all participants to complete
    # Results come back as lists (one per rank)
    put_results, get_results = await asyncio.gather(put_task, get_task)

    # Verify all operations succeeded
    for i, result in enumerate(put_results):
        assert result.get("success"), f"Putter rank {i} failed: {result}"

    for i, result in enumerate(get_results):
        assert result.get("success"), f"Getter rank {i} failed: {result}"

    # Verify tensor values for each rank
    for i, get_result in enumerate(get_results):
        expected_sum = fill_values[i] * shapes[i][0] * shapes[i][1]
        assert (
            abs(get_result["sum"] - expected_sum) < 1e-3
        ), f"Getter rank {i} sum mismatch: {get_result['sum']} vs {expected_sum}"

    print("=== Part 1 (BroadcastWindow) passed ===")

    # ==================== Part 2: Complex point-to-point without BroadcastWindow ====================
    # Test a complex pattern of transfers between all 4 processes:
    #   gpu_source: S0 (rank 0), S1 (rank 1)
    #   gpu_consumer: C0 (rank 0), C1 (rank 1)
    #
    # Round 1: Sources publish (S0: 2 tensors, S1: 1 tensor)
    # Round 2: Consumers get from sources, then publish their own
    #          C0 gets 2 from S0, publishes 3

    def check_parallel_execution(results: list, operation_name: str) -> bool:
        """Check if SPMD ranks executed in parallel by verifying time overlap."""
        # Filter out skipped results and extract timing
        timed_results = [r for r in results if not r.get("skipped") and "start_time" in r]
        if len(timed_results) < 2:
            return True  # Can't verify parallelism with < 2 results

        # Check for time overlap between any pair of ranks
        for i, r1 in enumerate(timed_results):
            for r2 in timed_results[i + 1 :]:
                # Overlap exists if one starts before the other ends
                overlap = not (r1["end_time"] < r2["start_time"] or r2["end_time"] < r1["start_time"])
                if overlap:
                    overlap_start = max(r1["start_time"], r2["start_time"])
                    overlap_end = min(r1["end_time"], r2["end_time"])
                    overlap_duration = overlap_end - overlap_start
                    print(
                        f"  ✓ {operation_name}: rank {r1['local_rank']} and {r2['local_rank']} "
                        f"overlapped for {overlap_duration:.3f}s"
                    )
                    return True

        # No overlap found - sequential execution
        print(f"  ⚠ {operation_name}: ranks executed sequentially (no time overlap)")
        return False

    #          C1 gets 1 from S1, publishes 4
    # Round 3: Sources get from consumers (cross-pattern)
    #          S0 gets 4 from C1
    #          S1 gets 3 from C0
    #
    # This creates bidirectional flow with different tensor counts per rank.

    prefix = f"{group_id}/p2p"

    # === Round 1: Sources publish different tensors per rank ===
    print("Round 1: Sources publishing...")
    source_pub_results = gpu_source.publish_tensors_by_rank(
        rank_specs={
            0: {  # S0 publishes 2 tensors
                "keys": [f"{prefix}/s0-a", f"{prefix}/s0-b"],
                "shapes": [[64, 64], [128, 64]],
                "fill_values": [1.0, 2.0],
            },
            1: {  # S1 publishes 1 tensor
                "keys": [f"{prefix}/s1-a"],
                "shapes": [[96, 96]],
                "fill_values": [3.0],
            },
        }
    )

    # Verify source publish results
    source_pub_list = source_pub_results if isinstance(source_pub_results, list) else [source_pub_results]
    for result in source_pub_list:
        if not result.get("skipped"):
            assert result.get("success"), f"Source publish failed: {result}"
            print(
                f"  S{result['local_rank']} published {len(result['results'])} tensor(s) in {result.get('duration', 0):.3f}s"
            )
    check_parallel_execution(source_pub_list, "Source publish")

    # === Round 2: Consumers get from sources, then publish their own ===
    print("Round 2: Consumers getting and publishing...")

    # First, consumers get from sources
    consumer_get_results = gpu_consumer.get_tensors_by_rank(
        rank_specs={
            0: {  # C0 gets 2 tensors from S0
                "keys": [f"{prefix}/s0-a", f"{prefix}/s0-b"],
                "shapes": [[64, 64], [128, 64]],
                "expected_fill_values": [1.0, 2.0],
            },
            1: {  # C1 gets 1 tensor from S1
                "keys": [f"{prefix}/s1-a"],
                "shapes": [[96, 96]],
                "expected_fill_values": [3.0],
            },
        }
    )

    # Verify consumer get results
    consumer_get_list = consumer_get_results if isinstance(consumer_get_results, list) else [consumer_get_results]
    for result in consumer_get_list:
        if not result.get("skipped"):
            assert result.get("success"), f"Consumer get failed: {result}"
            for r in result.get("results", []):
                assert r.get("correct", True), f"Value mismatch: {r}"
            print(
                f"  C{result['local_rank']} retrieved {len(result['results'])} tensor(s) in {result.get('duration', 0):.3f}s"
            )
    check_parallel_execution(consumer_get_list, "Consumer get")

    # Then, consumers publish their own tensors
    consumer_pub_results = gpu_consumer.publish_tensors_by_rank(
        rank_specs={
            0: {  # C0 publishes 3 tensors
                "keys": [f"{prefix}/c0-a", f"{prefix}/c0-b", f"{prefix}/c0-c"],
                "shapes": [[32, 32], [48, 48], [64, 32]],
                "fill_values": [4.0, 5.0, 6.0],
            },
            1: {  # C1 publishes 4 tensors
                "keys": [f"{prefix}/c1-a", f"{prefix}/c1-b", f"{prefix}/c1-c", f"{prefix}/c1-d"],
                "shapes": [[40, 40], [50, 50], [60, 60], [70, 70]],
                "fill_values": [7.0, 8.0, 9.0, 10.0],
            },
        }
    )

    # Verify consumer publish results
    consumer_pub_list = consumer_pub_results if isinstance(consumer_pub_results, list) else [consumer_pub_results]
    for result in consumer_pub_list:
        if not result.get("skipped"):
            assert result.get("success"), f"Consumer publish failed: {result}"
            print(
                f"  C{result['local_rank']} published {len(result['results'])} tensor(s) in {result.get('duration', 0):.3f}s"
            )
    check_parallel_execution(consumer_pub_list, "Consumer publish")

    # === Round 3: Sources get from consumers (cross-pattern) ===
    print("Round 3: Sources getting from consumers (cross-pattern)...")

    source_get_results = gpu_source.get_tensors_by_rank(
        rank_specs={
            0: {  # S0 gets 4 tensors from C1
                "keys": [f"{prefix}/c1-a", f"{prefix}/c1-b", f"{prefix}/c1-c", f"{prefix}/c1-d"],
                "shapes": [[40, 40], [50, 50], [60, 60], [70, 70]],
                "expected_fill_values": [7.0, 8.0, 9.0, 10.0],
            },
            1: {  # S1 gets 3 tensors from C0
                "keys": [f"{prefix}/c0-a", f"{prefix}/c0-b", f"{prefix}/c0-c"],
                "shapes": [[32, 32], [48, 48], [64, 32]],
                "expected_fill_values": [4.0, 5.0, 6.0],
            },
        }
    )

    # Verify source get results
    source_get_list = source_get_results if isinstance(source_get_results, list) else [source_get_results]
    for result in source_get_list:
        if not result.get("skipped"):
            assert result.get("success"), f"Source get failed: {result}"
            for r in result.get("results", []):
                assert r.get("correct", True), f"Value mismatch: {r}"
            print(
                f"  S{result['local_rank']} retrieved {len(result['results'])} tensor(s) in {result.get('duration', 0):.3f}s"
            )
    check_parallel_execution(source_get_list, "Source get")

    print("=== Part 2 (Complex point-to-point transfers) passed ===")
