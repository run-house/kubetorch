"""
Tests for GPU tensor transfer via kt.put(data=tensor) and kt.get(dest=tensor).

These tests verify:
- GPU tensor publishing via put with GPU data
- GPU tensor retrieval via get with pre-allocated tensor destination (NCCL broadcast)
- Tensor value correctness after transfer

The GPU Data Server architecture:
- kt.put(data=tensor) registers tensor IPC handles with a per-node GPU server
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
    1. Source publishes tensor via put(data=tensor)
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
    Test GPU state_dict (dictionary of tensors) transfer with BroadcastWindow.

    This test verifies:
    - Multiple tensors can be transferred as a dict (like model.state_dict())
    - All tensors flow through the same NCCL session
    - Tensor values are preserved correctly

    Architecture:
    - Source publishes a state_dict with multiple tensors
    - Consumer receives into a pre-allocated state_dict with same structure
    - All tensors are broadcast in a single coordinated transfer
    """
    import uuid

    service_name = gpu_source.service_name
    group_id = f"{service_name}/state-dict-test-{uuid.uuid4().hex[:8]}"

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
    assert put_result.get("success"), f"State dict put failed: {put_result.get('error')}"
    assert put_result["num_tensors"] == len(state_dict_spec), "Wrong number of tensors published"

    # Verify get succeeded
    assert get_result.get("success"), f"State dict get failed: {get_result.get('error')}"
    assert get_result["num_tensors"] == len(state_dict_spec), "Wrong number of tensors received"

    # Verify all tensor values are correct
    assert get_result["all_correct"], f"State dict values mismatch: {get_result.get('verification')}"


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
