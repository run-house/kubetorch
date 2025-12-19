"""
Tests for data storage and transfer APIs (kt.put and kt.get with key-value interface).

These tests verify:
- Basic put/get operations with keys
- Service-specific key handling
- Hierarchical key organization
- Force overwrite functionality
- Error handling
- ls (listing) operations
- locale="local" for peer-to-peer transfers
- Metadata server integration
"""

import tempfile
from pathlib import Path

import kubetorch as kt

import pytest

from tests.assets.kv_store.store_helper import StoreTestHelper


@pytest.fixture(scope="session")
async def store_helper():
    """Fixture that provides a StoreTestHelper instance for store testing."""
    helper_cls = await kt.cls(StoreTestHelper, name="store-test-helper").to_async(
        kt.Compute(cpus="0.1", memory="512Mi", logging_config=kt.LoggingConfig(level="debug"))
    )
    return helper_cls


@pytest.fixture(scope="session")
async def store_peer():
    """Fixture that provides a second StoreTestHelper instance for peer-to-peer testing."""
    helper_cls = await kt.cls(StoreTestHelper, name="store-test-peer").to_async(
        kt.Compute(cpus="0.1", memory="512Mi", logging_config=kt.LoggingConfig(level="debug"))
    )
    return helper_cls


# ==================== Basic KV Operations ====================


@pytest.mark.level("minimal")
async def test_store_kv_interface(store_helper):
    """Test kt.put and kt.get with key-value interface."""
    service_name = store_helper.service_name

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test files
        single_file = tmpdir / "single.txt"
        single_file.write_text("single file content")

        test_dir = tmpdir / "test_dir"
        test_dir.mkdir()
        (test_dir / "file1.txt").write_text("file 1 content")
        (test_dir / "file2.txt").write_text("file 2 content")
        nested = test_dir / "nested"
        nested.mkdir()
        (nested / "deep.txt").write_text("deep file content")

        file1 = tmpdir / "file1.txt"
        file2 = tmpdir / "file2.txt"
        file3 = tmpdir / "file3.txt"
        file1.write_text("content 1")
        file2.write_text("content 2")
        file3.write_text("content 3")

        # Test uploads
        kt.put(key=f"{service_name}/test-files/single.txt", src=str(single_file))
        kt.put(key=f"{service_name}/test-dir", src=str(test_dir))
        kt.put(key=f"{service_name}/multi-files", src=[str(file1), str(file2), str(file3)])
        kt.put(key=f"{service_name}/contents-test", src=str(test_dir), contents=True)

        # Verify uploads on remote
        results = store_helper.verify_uploaded_files()

        assert results["single_file"]["exists"], "Single file not stored"
        assert results["single_file"]["correct"], "Single file content incorrect"
        assert results["directory"]["exists"], "Directory not stored"
        assert results["directory"]["file_count"] >= 3, "Not all directory files stored"
        assert all(results["multiple_files"].values()), "Not all multiple files stored"
        assert results["contents_flag"]["exists"], "Contents flag upload failed"

        # Prepare and test downloads
        store_helper.prepare_download_files()

        download_dir = tmpdir / "downloads"
        download_dir.mkdir()

        kt.get(
            key=f"{service_name}/downloads/result.csv",
            dest=str(download_dir),
            filter_options="--include='*.csv'",
        )
        assert (download_dir / "result.csv").exists(), "Single file not retrieved"


@pytest.mark.level("minimal")
async def test_store_service_keys(store_helper):
    """Test kt.put and kt.get with service-specific keys."""
    service_name = store_helper.service_name

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        test_file = tmpdir / "service_test.txt"
        test_file.write_text("service specific content")

        kt.put(key=f"{service_name}/config/test.txt", src=str(test_file))

        result = store_helper.check_file_exists("config/test.txt")
        assert result["exists"], "File not stored with service key"
        assert "service specific content" in result["content"], "Content incorrect"

        # Test download
        store_helper.create_output_file("output/service_result.log", "output from service")
        download_dir = tmpdir / "service_downloads"
        download_dir.mkdir()

        kt.get(
            key=f"{service_name}/output/service_result.log", dest=str(download_dir), filter_options="--include='*.log'"
        )
        downloaded = download_dir / "service_result.log"
        assert downloaded.exists(), "File not retrieved with service key"


@pytest.mark.level("minimal")
async def test_store_hierarchical_keys(store_helper):
    """Test hierarchical key organization."""
    service_name = store_helper.service_name

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create model and dataset files
        model_v1 = tmpdir / "model_v1.pkl"
        model_v1.write_text("model version 1")
        model_v2 = tmpdir / "model_v2.pkl"
        model_v2.write_text("model version 2")
        train_data = tmpdir / "train.csv"
        train_data.write_text("train data")
        test_data = tmpdir / "test.csv"
        test_data.write_text("test data")

        # Store with hierarchical keys
        kt.put(key=f"{service_name}/ml-project/models/v1.pkl", src=str(model_v1), filter_options="--include='*.pkl'")
        kt.put(key=f"{service_name}/ml-project/models/v2.pkl", src=str(model_v2), filter_options="--include='*.pkl'")
        kt.put(
            key=f"{service_name}/ml-project/datasets/train.csv", src=str(train_data), filter_options="--include='*.csv'"
        )
        kt.put(
            key=f"{service_name}/ml-project/datasets/test.csv", src=str(test_data), filter_options="--include='*.csv'"
        )

        result = store_helper.check_hierarchy()
        assert result["models_v1"], "Model v1 not stored"
        assert result["models_v2"], "Model v2 not stored"
        assert result["datasets_train"], "Train dataset not stored"
        assert result["datasets_test"], "Test dataset not stored"


@pytest.mark.level("minimal")
async def test_store_overwrite(store_helper):
    """Test force overwrite option with keys."""
    service_name = store_helper.service_name

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        file_v1 = tmpdir / "config.yaml"
        file_v1.write_text("version: 1\ndata: initial")

        kt.put(key=f"{service_name}/config/main/config.yaml", src=str(file_v1), filter_options="--include='*.yaml'")

        initial_content = store_helper.read_file("config/main/config.yaml")
        assert "version: 1" in initial_content, "Initial storage failed"

        # Update and overwrite
        file_v1.write_text("version: 2\ndata: updated")
        kt.put(
            key=f"{service_name}/config/main/config.yaml",
            src=str(file_v1),
            force=True,
            filter_options="--include='*.yaml'",
        )

        updated_content = store_helper.read_file("config/main/config.yaml")
        assert "version: 2" in updated_content, "Force overwrite failed"


# ==================== Error Handling ====================


@pytest.mark.level("minimal")
def test_store_error_handling():
    """Test error handling for key-value operations."""
    DataStoreError = kt.DataStoreError

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Non-existent source file
        with pytest.raises(ValueError, match="Could not locate path to sync up"):
            kt.put(key="test/nonexistent", src="/non/existent/file.txt")

        # Non-existent key
        with pytest.raises(DataStoreError):
            kt.get(key="nonexistent/key", dest=str(tmpdir))

        # These should work - keys are flexible
        test_file = tmpdir / "test.txt"
        test_file.write_text("test")
        kt.put(key="../test", src=str(test_file))
        kt.put(key="/absolute/path", src=str(test_file))

        kt.rm(key="../test", recursive=True)
        kt.rm(key="/absolute/path", recursive=True)


# ==================== Listing Operations ====================


@pytest.mark.level("minimal")
async def test_store_ls(store_helper):
    """Test kt.ls for listing store contents."""
    service_name = store_helper.service_name

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create and upload test files
        test_file1 = tmpdir / "file1.txt"
        test_file1.write_text("content 1")
        test_file2 = tmpdir / "file2.txt"
        test_file2.write_text("content 2")
        test_dir = tmpdir / "test_dir"
        test_dir.mkdir()
        (test_dir / "nested.txt").write_text("nested content")

        kt.put(key=f"{service_name}/ls-test/file1.txt", src=str(test_file1))
        kt.put(key=f"{service_name}/ls-test/file2.txt", src=str(test_file2))
        kt.put(key=f"{service_name}/ls-test/subdir", src=str(test_dir))

        # Test listing from outside the cluster
        items = kt.ls(f"{service_name}/ls-test", verbose=True)
        assert len(items) >= 3, f"Should list uploaded files, got: {items}"

        item_names = [item["name"] if isinstance(item, dict) else item for item in items]
        assert any("file1.txt" in name for name in item_names), "file1.txt should be listed"
        assert any("file2.txt" in name for name in item_names), "file2.txt should be listed"

        # Test listing from inside the cluster
        items = store_helper.list_store_contents("ls-test")
        assert len(items) >= 3, "Helper should list uploaded files"


# ==================== locale="local" / Peer-to-Peer Operations ====================


@pytest.mark.level("minimal")
async def test_put_locale_local_get_external(store_helper):
    """Test kt.put(locale="local") for zero-copy publishing of data."""
    service_name = store_helper.service_name

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        test_data_path = "local_test/data.txt"
        test_data_content = "Published via locale=local\nZero-copy content"

        publish_key = f"{service_name}/local-published/data"
        result = store_helper.publish_data_local(key=publish_key, local_path=test_data_path, content=test_data_content)

        assert result["success"], f"put with locale=local failed: {result.get('error')}"
        assert result["pod_ip"] != "unknown", "POD_IP should be set"

        # Test external client retrieval
        download_dir = tmpdir / "external_download"
        download_dir.mkdir()

        kt.get(key=publish_key, dest=str(download_dir), verbose=True)

        downloaded_files = list(download_dir.rglob("*.txt"))
        assert len(downloaded_files) > 0, "Should have downloaded file via external client"


@pytest.mark.level("minimal")
async def test_store_peer_to_peer_transfer(store_helper, store_peer):
    """Test peer-to-peer data transfer using locale='local' and get."""
    publisher_data_path = "peer_pub/shared_model.pkl"
    publisher_data_content = "Peer-to-peer model data\nVersion 1.0"
    publish_key = f"/{store_helper.service_name}/peer-shared/model"

    pub_result = store_helper.publish_data_local(
        key=publish_key, local_path=publisher_data_path, content=publisher_data_content
    )
    assert pub_result["success"], f"Publisher put with locale=local failed: {pub_result.get('error')}"

    # Retrieve from second service
    get_result = store_peer.get_data_from_store(
        key=publish_key,
        dest_path="peer_download",
        filter_options="--include='*.pkl'",
    )

    assert get_result["success"], f"Peer get failed: {get_result.get('error')}"
    assert get_result["file_count"] > 0, "Should have downloaded from peer"


@pytest.mark.level("minimal")
async def test_store_metadata_server_integration(store_helper):
    """Test that metadata server integration works for tracking published keys."""
    service_name = store_helper.service_name

    metadata_test_path = "metadata_test/tracked.txt"
    metadata_key = f"{service_name}/metadata-tracked/data"
    result = store_helper.publish_data_local(key=metadata_key, local_path=metadata_test_path)

    assert result["success"], "put with locale=local should succeed"

    check_result = store_helper.check_metadata_server(metadata_key)
    assert check_result.get("registered") or check_result.get("retrievable"), f"Key should be tracked: {check_result}"


@pytest.mark.level("minimal")
async def test_store_external_client_metadata_api(store_helper):
    """Test that metadata server returns pod info for external clients."""
    service_name = store_helper.service_name

    metadata_api_test_path = "metadata_api_test/data.txt"
    metadata_api_key = f"{service_name}/metadata-api-test/data"
    result = store_helper.publish_data_local(key=metadata_api_key, local_path=metadata_api_test_path)

    assert result["success"], "put with locale=local should succeed"

    from kubetorch.data_store.metadata_client import MetadataClient

    metadata_client = MetadataClient(namespace=store_helper.compute.namespace)

    pod_info = metadata_client.get_source_ip(metadata_api_key, external=True)

    assert pod_info is not None, "Should return pod info"
    assert isinstance(pod_info, dict), "Should return dict"
    assert "pod_name" in pod_info, "Should include pod_name"
    assert "namespace" in pod_info, "Should include namespace"


# ==================== Seeding Verification ====================


def _delete_from_store_filesystem(key: str, namespace: str = "default") -> dict:
    """
    Delete a key directly from the store pod's filesystem (bypassing metadata server).

    This is useful for testing failover - delete from store but keep metadata entries
    so we can verify retrieval from seeded peers.
    """
    import subprocess

    # Build the path on the store pod
    store_path = f"/data/{namespace}/{key}"

    # Get the store pod name
    result = subprocess.run(
        [
            "kubectl",
            "get",
            "pods",
            "-n",
            namespace,
            "-l",
            "app=kubetorch-data-store",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    pod_name = result.stdout.strip()

    if not pod_name:
        raise RuntimeError("No store pod found")

    # Delete the file/directory from the store pod
    result = subprocess.run(
        ["kubectl", "exec", pod_name, "-n", namespace, "--", "rm", "-rf", store_path],
        capture_output=True,
        text=True,
        check=True,
    )

    return {
        "success": True,
        "pod_name": pod_name,
        "path": store_path,
    }


# ==================== Filesystem Broadcast Operations ====================


@pytest.mark.level("minimal")
async def test_filesystem_broadcast_p2p(store_helper, store_peer):
    """Test peer-to-peer filesystem broadcast: helper publishes, peer gets via broadcast.

    This tests p2p broadcast (locale="local") with a single getter:
    1. store_helper publishes data locally (rank 0)
    2. store_peer gets via broadcast (rank 1, parent = helper)
    """
    import uuid

    test_id = uuid.uuid4().hex[:8]
    publisher_data_path = f"fs_broadcast_p2p_{test_id}/model.bin"
    publisher_data_content = "Peer-to-peer broadcast test data\nVersion 1.0"
    publish_key = f"/{store_helper.service_name}/fs-broadcast-p2p-{test_id}/model"
    group_id = f"fs-broadcast-p2p-{test_id}"

    # Step 1: Publisher puts data with locale="local" (starts rsync daemon)
    pub_result = store_helper.publish_data_local(
        key=publish_key, local_path=publisher_data_path, content=publisher_data_content
    )
    assert pub_result["success"], f"Publisher put with locale=local failed: {pub_result.get('error')}"
    publisher_ip = pub_result["pod_ip"]
    assert publisher_ip != "unknown", "Publisher should have POD_IP set"

    # Step 2: Getter retrieves using filesystem broadcast
    # With fanout=1, getter is rank 1 with parent = source (rank 0)
    get_result = store_peer.get_with_fs_broadcast(
        key=publish_key,
        dest_path=f"fs_broadcast_p2p_download_{test_id}",
        group_id=group_id,
        fanout=1,
        timeout=60.0,
    )

    assert get_result[
        "success"
    ], f"Filesystem broadcast get failed: {get_result.get('error')}\n{get_result.get('traceback', '')}"
    assert get_result["file_count"] > 0, "Should have downloaded files via broadcast"
    assert publisher_data_content in get_result.get("content", ""), "Content should match published data"


@pytest.mark.level("minimal")
async def test_filesystem_broadcast_chain(store_helper, store_peer):
    """Test filesystem broadcast chain: store -> helper -> peer (fanout=1).

    This tests the full 3-node chain:
    1. Local client uploads to store pod (rank 0)
    2. store_helper gets via broadcast (rank 1, parent = store)
    3. store_peer gets via broadcast (rank 2, parent = helper)

    With fanout=1, each node can only have 1 child, creating a linear chain.
    """
    import uuid

    test_id = uuid.uuid4().hex[:8]
    service_name = store_helper.service_name
    test_content = "Filesystem broadcast chain test data\nVersion 1.0\n" + ("x" * 1000)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test file locally
        test_file = tmpdir / "chain_test.bin"
        test_file.write_text(test_content)

        broadcast_key = f"{service_name}/fs-broadcast-chain-{test_id}/model"
        group_id = f"fs-broadcast-chain-{test_id}"

        # Step 1: Upload from local client to store pod
        kt.put(key=broadcast_key, src=str(test_file))

        # Step 2: store_helper gets via broadcast (rank 1, parent = store at rank 0)
        get_result_helper = store_helper.get_with_fs_broadcast(
            key=broadcast_key,
            dest_path=f"fs_chain_helper_{test_id}",
            group_id=group_id,
            fanout=1,
            timeout=60.0,
        )

        assert get_result_helper["success"], (
            f"Helper get failed: {get_result_helper.get('error')}\n" f"{get_result_helper.get('traceback', '')}"
        )
        assert get_result_helper["file_count"] > 0, "Helper should have downloaded files"
        assert test_content in get_result_helper.get("content", ""), "Helper content should match"

        # Step 3: store_peer gets via broadcast (rank 2, parent = helper at rank 1)
        get_result_peer = store_peer.get_with_fs_broadcast(
            key=broadcast_key,
            dest_path=f"fs_chain_peer_{test_id}",
            group_id=group_id,
            fanout=1,
            timeout=60.0,
        )

        assert get_result_peer["success"], (
            f"Peer get failed: {get_result_peer.get('error')}\n" f"{get_result_peer.get('traceback', '')}"
        )
        assert get_result_peer["file_count"] > 0, "Peer should have downloaded files"
        assert test_content in get_result_peer.get("content", ""), "Peer content should match"

        # Verify the chain: peer should have gotten data from helper (rank 1), not store (rank 0)
        # With fanout=1: rank 2's parent = (2-1) // 1 = 1 (helper)


# ==================== Edge Cases ====================


@pytest.mark.level("minimal")
async def test_store_single_file_contents_flag(store_helper):
    """Test single file upload with contents=True flag."""
    service_name = store_helper.service_name

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        test_file = tmpdir / "source.txt"
        test_file.write_text("source content")

        kt.put(key=f"{service_name}/single-contents/dir", src=str(test_file), contents=True)

        download_dir = tmpdir / "verify"
        download_dir.mkdir()
        kt.get(key=f"{service_name}/single-contents/dir", dest=str(download_dir))

        assert (download_dir / "dir" / "source.txt").exists(), "File should be inside directory"


@pytest.mark.level("minimal")
async def test_store_file_renaming(store_helper):
    """Test that files are renamed to match the key when uploaded."""
    service_name = store_helper.service_name

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        source_file = tmpdir / "original_name.txt"
        source_file.write_text("renamed content")

        kt.put(key=f"{service_name}/renamed/new_name.txt", src=str(source_file))

        download_dir = tmpdir / "download"
        download_dir.mkdir()
        kt.get(key=f"{service_name}/renamed/new_name.txt", dest=str(download_dir))

        assert (download_dir / "new_name.txt").exists(), "File should be renamed"
        assert not (download_dir / "original_name.txt").exists(), "Original name should not exist"


# ==================== Queue/Stream Operations ====================


@pytest.mark.level("minimal")
async def test_queue_basic_produce_consume(store_helper):
    """Test basic queue produce and consume with kt.put/kt.get using Queue objects."""
    import uuid

    test_id = uuid.uuid4().hex[:8]
    service_name = store_helper.service_name
    queue_key = f"{service_name}/queue-test-{test_id}/logs"

    # Test items to send
    test_items = [f"log line {i}" for i in range(5)]

    # Produce items from the remote pod
    produce_result = store_helper.queue_produce(key=queue_key, items=test_items, timeout=30.0)
    assert produce_result["success"], f"Queue produce failed: {produce_result.get('error')}"
    assert produce_result["items_sent"] == len(test_items), "Should send all items"

    # Consume items from the remote pod
    consume_result = store_helper.queue_consume(key=queue_key, max_items=10, timeout=30.0)
    assert consume_result["success"], f"Queue consume failed: {consume_result.get('error')}"
    assert consume_result["items_received"] == len(
        test_items
    ), f"Should receive all items, got {consume_result['items_received']}"
    assert consume_result["items"] == test_items, "Items should match"


@pytest.mark.level("minimal")
async def test_queue_cross_pod_transfer(store_helper, store_peer):
    """Test queue transfer between two pods: one produces, other consumes."""
    import uuid

    test_id = uuid.uuid4().hex[:8]
    service_name = store_helper.service_name
    queue_key = f"{service_name}/queue-cross-{test_id}/stream"

    # Test items
    test_items = [f"cross-pod message {i}" for i in range(3)]

    # Producer: store_helper produces items
    produce_result = store_helper.queue_produce(key=queue_key, items=test_items, timeout=30.0)
    assert produce_result["success"], f"Queue produce failed: {produce_result.get('error')}"

    # Consumer: store_peer consumes items
    consume_result = store_peer.queue_consume(key=queue_key, max_items=10, timeout=30.0)
    assert consume_result["success"], f"Queue consume failed: {consume_result.get('error')}"
    assert consume_result["items_received"] == len(
        test_items
    ), f"Should receive all items, got {consume_result['items_received']}"
    assert consume_result["items"] == test_items, "Items should match"


@pytest.mark.level("minimal")
async def test_queue_metadata_registration(store_helper):
    """Test that queue keys are properly registered with the metadata server."""
    import uuid

    test_id = uuid.uuid4().hex[:8]
    service_name = store_helper.service_name
    queue_key = f"{service_name}/queue-meta-{test_id}/data"

    # Produce a single item to create the queue
    produce_result = store_helper.queue_produce(key=queue_key, items=["test"], timeout=30.0)
    assert produce_result["success"], f"Queue produce failed: {produce_result.get('error')}"

    # Check that the queue was created in Redis
    check_result = store_helper.queue_check_redis(queue_key)
    assert check_result["success"], f"Queue check failed: {check_result.get('error')}"
    assert check_result["exists"], "Queue should exist in Redis"
    assert check_result["length"] >= 1, "Queue should have at least 1 item"

    # Verify metadata server knows about this key
    from kubetorch.data_store.metadata_client import MetadataClient

    metadata_client = MetadataClient(namespace=store_helper.compute.namespace)
    source_info = metadata_client.get_source_info(queue_key)

    assert source_info is not None, "Should return source info"
    assert source_info.get("found"), "Key should be found"
    assert source_info.get("data_type") == "queue", f"Data type should be 'queue', got {source_info.get('data_type')}"


@pytest.mark.level("minimal")
def test_queue_error_handling():
    """Test error handling for queue operations."""
    from queue import Queue

    # Non-existent queue key should raise error on get
    q = Queue()
    with pytest.raises(ValueError, match="not found"):
        kt.get(key="nonexistent/queue/key", dest=q)

    # Queue operations only support single keys
    with pytest.raises(ValueError, match="only supports a single key"):
        kt.put(key=["key1", "key2"], src=Queue())

    with pytest.raises(ValueError, match="only supports a single key"):
        kt.get(key=["key1", "key2"], dest=Queue())


# ==================== External Queue Tests ====================


@pytest.mark.level("minimal")
async def test_queue_external_consume(store_helper):
    """Test consuming queue items from outside the cluster via WebSocket tunnel."""
    import time
    import uuid
    from queue import Empty, Queue

    test_id = uuid.uuid4().hex[:8]
    service_name = store_helper.service_name
    queue_key = f"{service_name}/queue-external-consume-{test_id}/logs"

    # Produce items from inside the cluster
    test_items = [f"external consume test {i}" for i in range(5)]
    produce_result = store_helper.queue_produce(key=queue_key, items=test_items, timeout=30.0)
    assert produce_result["success"], f"Queue produce failed: {produce_result.get('error')}"

    # Consume from outside the cluster (this test runs externally)
    # This exercises the WebSocket tunnel for Redis access
    dest_queue = Queue()
    _thread = kt.get(key=queue_key, dest=dest_queue, verbose=True)  # noqa: F841

    # Collect items with timeout
    received_items = []
    start_time = time.time()
    while len(received_items) < len(test_items) and (time.time() - start_time) < 30:
        try:
            item = dest_queue.get(timeout=1.0)
            if item is None:
                break
            if isinstance(item, bytes):
                item = item.decode()
            received_items.append(item)
        except Empty:
            if received_items:  # Have some items, check if stream ended
                break
            continue

    assert len(received_items) == len(test_items), f"Should receive all items externally, got {len(received_items)}"
    assert received_items == test_items, "Items should match"


@pytest.mark.level("minimal")
async def test_queue_external_produce(store_helper):
    """Test producing queue items from outside the cluster via WebSocket tunnel."""
    import uuid
    from queue import Queue

    test_id = uuid.uuid4().hex[:8]
    service_name = store_helper.service_name
    queue_key = f"{service_name}/queue-external-produce-{test_id}/logs"

    # Produce from outside the cluster (this test runs externally)
    # This exercises the WebSocket tunnel for Redis access
    src_queue = Queue()
    test_items = [f"external produce test {i}" for i in range(5)]

    thread = kt.put(key=queue_key, src=src_queue, verbose=True)

    # Put items to the queue
    for item in test_items:
        src_queue.put(item)
    src_queue.put(None)  # Sentinel to stop streaming

    # Wait for thread to complete
    thread.join(timeout=10.0)

    # Consume from inside the cluster to verify items arrived
    consume_result = store_helper.queue_consume(key=queue_key, max_items=10, timeout=30.0)
    assert consume_result["success"], f"Queue consume failed: {consume_result.get('error')}"
    assert consume_result["items_received"] == len(
        test_items
    ), f"Should receive all items, got {consume_result['items_received']}"
    assert consume_result["items"] == test_items, "Items should match"


@pytest.mark.level("minimal")
async def test_queue_external_bidirectional(store_helper):
    """Test bidirectional queue access from outside the cluster."""
    import time
    import uuid
    from queue import Empty, Queue

    test_id = uuid.uuid4().hex[:8]
    service_name = store_helper.service_name
    queue_key = f"{service_name}/queue-external-bidir-{test_id}/stream"

    # Phase 1: Produce from external, consume from internal
    src_queue = Queue()
    external_items = [f"external to internal {i}" for i in range(3)]

    thread = kt.put(key=queue_key, src=src_queue, verbose=True)
    for item in external_items:
        src_queue.put(item)
    src_queue.put(None)
    thread.join(timeout=10.0)

    consume_result = store_helper.queue_consume(key=queue_key, max_items=10, timeout=30.0)
    assert consume_result["success"], f"Internal consume failed: {consume_result.get('error')}"
    assert consume_result["items"] == external_items, "Phase 1: External->Internal should work"

    # Phase 2: Produce from internal, consume from external
    queue_key_2 = f"{service_name}/queue-external-bidir2-{test_id}/stream"
    internal_items = [f"internal to external {i}" for i in range(3)]

    produce_result = store_helper.queue_produce(key=queue_key_2, items=internal_items, timeout=30.0)
    assert produce_result["success"], f"Internal produce failed: {produce_result.get('error')}"

    dest_queue = Queue()
    _thread = kt.get(key=queue_key_2, dest=dest_queue, verbose=True)  # noqa: F841

    received_items = []
    start_time = time.time()
    while len(received_items) < len(internal_items) and (time.time() - start_time) < 30:
        try:
            item = dest_queue.get(timeout=1.0)
            if item is None:
                break
            if isinstance(item, bytes):
                item = item.decode()
            received_items.append(item)
        except Empty:
            if received_items:
                break
            continue

    assert received_items == internal_items, "Phase 2: Internal->External should work"
