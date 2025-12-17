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
        kt.Compute(cpus="0.1", memory="512Mi")
    )
    return helper_cls


@pytest.fixture(scope="session")
async def store_peer():
    """Fixture that provides a second StoreTestHelper instance for peer-to-peer testing."""
    helper_cls = await kt.cls(StoreTestHelper, name="store-test-peer").to_async(kt.Compute(cpus="0.1", memory="512Mi"))
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


@pytest.mark.level("minimal")
async def test_store_seeding_with_broadcast(store_helper, store_peer):
    """
    Test that broadcast window works correctly for coordinated data transfer.

    Flow:
    1. Upload data to the central store via kt.put
    2. Pod B (store_peer) retrieves from store
    3. Delete the data from the store filesystem (but keep metadata entries)
    4. Pod A (store_helper) retrieves the same key - must get it from Pod B since store is empty
    """
    service_name = store_helper.service_name

    # Step 1: Upload data to the central store
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        seed_test_file = tmpdir / "seed_test.txt"
        seed_test_file.write_text("Seeding test data\nOriginal from central store")

        seeding_key = f"/{service_name}/seeding-test/data.txt"
        kt.put(key=seeding_key, src=str(seed_test_file))

    # Step 2: Pod B retrieves from store (uses locale="local" to seed)
    get_result_b = store_peer.get_data_from_store(
        key=seeding_key,
        dest_path="seeding_test_download",
    )
    assert get_result_b["success"], f"Pod B get failed: {get_result_b.get('error')}"
    assert get_result_b["file_count"] > 0, "Pod B should have downloaded files"

    # Pod B publishes the data locally so others can get it from Pod B
    store_peer.publish_data_local(
        key=seeding_key,
        local_path="seeding_test_download/data.txt",
    )

    # Step 3: Delete the data directly from the store pod's filesystem
    _delete_from_store_filesystem(seeding_key)  # Raises on failure

    # Step 4: Pod A retrieves the same key - must get it from Pod B since store is empty
    get_result_a = store_helper.get_data_from_store(
        key=seeding_key,
        dest_path="seeding_from_peer",
    )
    assert get_result_a["success"], f"Pod A get failed (failover didn't work): {get_result_a.get('error')}"
    assert get_result_a["file_count"] > 0, "Pod A should have downloaded from seeded Pod B"


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
