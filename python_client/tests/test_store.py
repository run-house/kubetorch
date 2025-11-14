"""
Tests for data storage and transfer APIs (kt.put and kt.get with key-value interface)
"""
import tempfile

import kubetorch as kt

import pytest

from tests.assets.kv_store.store_helper import StoreTestHelper


@pytest.fixture(scope="session")
async def store_test_helper():
    """Fixture that provides a StoreTestHelper instance for store testing."""
    helper_cls = await kt.cls(StoreTestHelper, name="store-test-helper").to_async(
        kt.Compute(cpus="0.1", memory="512Mi")
    )
    return helper_cls


@pytest.mark.level("minimal")
async def test_store_kv_interface(store_test_helper):
    """Test kt.put and kt.get with key-value interface."""
    from pathlib import Path

    # The fixture already provides the helper instance
    helper = store_test_helper

    # Create temporary test data locally
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test files
        single_file = tmpdir / "single.txt"
        single_file.write_text("single file content")

        # Create test directory with nested structure
        test_dir = tmpdir / "test_dir"
        test_dir.mkdir()
        (test_dir / "file1.txt").write_text("file 1 content")
        (test_dir / "file2.txt").write_text("file 2 content")
        nested = test_dir / "nested"
        nested.mkdir()
        (nested / "deep.txt").write_text("deep file content")

        # Create multiple files
        file1 = tmpdir / "file1.txt"
        file2 = tmpdir / "file2.txt"
        file3 = tmpdir / "file3.txt"
        file1.write_text("content 1")
        file2.write_text("content 2")
        file3.write_text("content 3")

        # Test 1: Upload single file with key
        kt.put(key=f"{store_test_helper.service_name}/test-files/single.txt", src=str(single_file))

        # Test 2: Upload directory with key
        kt.put(key=f"{store_test_helper.service_name}/test-dir", src=str(test_dir))

        # Test 3: Upload multiple files with key
        kt.put(key=f"{store_test_helper.service_name}/multi-files", src=[str(file1), str(file2), str(file3)])

        # Test 4: Upload directory contents only with contents flag
        kt.put(key=f"{store_test_helper.service_name}/contents-test", src=str(test_dir), contents=True)

        # Verify uploads on remote
        upload_results = helper.verify_uploaded_files()

        # Check upload results
        assert upload_results["single_file"]["exists"], "Single file not stored at key"
        assert upload_results["single_file"]["correct"], "Single file content incorrect"

        assert upload_results["directory"]["exists"], "Directory not stored at key"
        assert upload_results["directory"]["file_count"] >= 3, "Not all directory files stored"
        assert upload_results["directory"]["has_nested"], "Nested directory structure lost"

        assert all(upload_results["multiple_files"].values()), "Not all multiple files stored"

        assert upload_results["contents_flag"]["exists"], "Contents flag upload failed"
        assert upload_results["contents_flag"]["file1_direct"], "Contents not copied directly"
        assert upload_results["contents_flag"].get(
            "no_subdir", True
        ), "Subdirectory incorrectly created with contents flag"

        # Prepare download test files
        download_prep = helper.prepare_download_files()
        assert download_prep["prepared"], "Failed to prepare download files"

        # Create download directory
        download_dir = tmpdir / "downloads"
        download_dir.mkdir()

        # Test 5: Download single file with key
        # We need to bypass filters because our .gitignore might ignore .csv files
        kt.get(
            key=f"{store_test_helper.service_name}/downloads/result.csv",
            dest=str(download_dir),
            filter_options="--include='*.csv'",
        )
        assert (download_dir / "result.csv").exists(), "Single file not retrieved from key"
        assert "id,value" in (download_dir / "result.csv").read_text(), "Retrieved file content incorrect"

        # Test 6: Download directory with key
        models_dir = download_dir / "models_dl"
        models_dir.mkdir()
        # We need to bypass filters because our .gitignore ignores .pkl files
        kt.get(key=f"{store_test_helper.service_name}/models", dest=str(models_dir), filter_options="--include='*.pkl'")
        assert (models_dir / "models" / "model.pkl").exists(), "Directory not retrieved from key"
        assert (models_dir / "models" / "v1" / "weights.bin").exists(), "Nested files not retrieved"

        # Test 7: Download with contents flag
        logs_dir = download_dir / "logs_dl"
        logs_dir.mkdir()
        # We need to bypass filters because our .gitignore ignores .log files
        kt.get(
            key=f"{store_test_helper.service_name}/logs",
            dest=str(logs_dir),
            contents=True,
            filter_options="--include='*.log'",
        )
        # With contents=True, contents should be copied directly
        for i in range(3):
            log_file = logs_dir / f"output_{i}.log"
            assert log_file.exists(), f"Log file {i} not retrieved from key"
            assert f"Log file {i}" in log_file.read_text(), f"Log file {i} content incorrect"


@pytest.mark.level("minimal")
async def test_store_service_keys(store_test_helper):
    """Test kt.put and kt.get with service-specific keys."""
    from pathlib import Path

    # The fixture already provides the helper instance
    helper = store_test_helper
    service_name = helper.service_name

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test file
        test_file = tmpdir / "service_test.txt"
        test_file.write_text("service specific content")

        # Upload with service as part of the key
        # This should interpret the first part as service name
        kt.put(key=f"{service_name}/config/test.txt", src=str(test_file))

        # Verify via helper - check if file exists in the service's storage
        result = helper.check_file_exists("config/test.txt")

        assert result["exists"], "File not stored with service key"
        assert "service specific content" in result["content"], "Service key upload content incorrect"

        # Create a file on the service for download
        helper.create_output_file("output/service_result.log", "output from service\nline 2\nline 3")

        # Download with service key
        download_dir = tmpdir / "service_downloads"
        download_dir.mkdir()
        # We need to bypass filters because our .gitignore ignores .log files
        kt.get(
            key=f"{service_name}/output/service_result.log", dest=str(download_dir), filter_options="--include='*.log'"
        )

        downloaded = download_dir / "service_result.log"
        assert downloaded.exists(), "File not retrieved with service key"
        assert "output from service" in downloaded.read_text(), "Service key download content incorrect"


@pytest.mark.level("minimal")
async def test_store_hierarchical_keys(store_test_helper):
    """Test hierarchical key organization."""
    from pathlib import Path

    # The fixture already provides the helper instance
    helper = store_test_helper

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test data with hierarchical structure
        data_dir = tmpdir / "data"
        data_dir.mkdir()

        # Create model files
        model_v1 = data_dir / "model_v1.pkl"
        model_v1.write_text("model version 1")

        model_v2 = data_dir / "model_v2.pkl"
        model_v2.write_text("model version 2")

        # Create dataset files
        train_data = data_dir / "train.csv"
        train_data.write_text("train data")

        test_data = data_dir / "test.csv"
        test_data.write_text("test data")

        # Store with hierarchical keys - prepend service name
        service_name = store_test_helper.service_name
        # We need to bypass filters because our .gitignore ignores .pkl files
        # Keys include file extensions - files will be renamed to match the key
        kt.put(key=f"{service_name}/ml-project/models/v1.pkl", src=str(model_v1), filter_options="--include='*.pkl'")
        kt.put(key=f"{service_name}/ml-project/models/v2.pkl", src=str(model_v2), filter_options="--include='*.pkl'")
        # We need to bypass filters because our .gitignore might ignore .csv files
        kt.put(
            key=f"{service_name}/ml-project/datasets/train.csv", src=str(train_data), filter_options="--include='*.csv'"
        )
        kt.put(
            key=f"{service_name}/ml-project/datasets/test.csv", src=str(test_data), filter_options="--include='*.csv'"
        )

        # Verify hierarchical storage
        result = helper.check_hierarchy()

        assert result["models_v1"], "Model v1 not stored with hierarchical key"
        assert result["models_v2"], "Model v2 not stored with hierarchical key"
        assert result["datasets_train"], "Train dataset not stored with hierarchical key"
        assert result["datasets_test"], "Test dataset not stored with hierarchical key"

        # Test retrieval with hierarchical keys
        download_dir = tmpdir / "retrieved"
        download_dir.mkdir()

        # Create destination directories for single file downloads
        (download_dir / "model_v1_dir").mkdir()
        (download_dir / "datasets").mkdir()

        # Get specific versions - use service name
        # We need to bypass filters because our .gitignore ignores .pkl files
        kt.get(
            key=f"{service_name}/ml-project/models/v1.pkl",
            dest=str(download_dir / "model_v1_dir"),
            filter_options="--include='*.pkl'",
        )
        # We need to bypass filters because our .gitignore might ignore .csv files
        kt.get(
            key=f"{service_name}/ml-project/datasets/train.csv",
            dest=str(download_dir / "datasets"),
            filter_options="--include='*.csv'",
        )

        assert (download_dir / "model_v1_dir" / "v1.pkl").exists(), "Model v1 not retrieved"
        assert (download_dir / "datasets" / "train.csv").exists(), "Train dataset not retrieved"


@pytest.mark.level("minimal")
async def test_store_overwrite(store_test_helper):
    """Test force overwrite option with keys."""
    from pathlib import Path

    # The fixture already provides the helper instance
    helper = store_test_helper

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create initial file
        file_v1 = tmpdir / "config.yaml"
        file_v1.write_text("version: 1\ndata: initial")

        # Store initial version with key - prepend service name
        service_name = store_test_helper.service_name
        # We need to bypass filters because our .gitignore might ignore .yaml files
        # Key includes filename - file will be renamed to match the key
        kt.put(key=f"{service_name}/config/main/config.yaml", src=str(file_v1), filter_options="--include='*.yaml'")

        # Verify initial storage - check in service's storage area
        initial_content = helper.read_file("config/main/config.yaml")
        assert "version: 1" in initial_content, "Initial storage failed"

        # Update file locally
        file_v1.write_text("version: 2\ndata: updated")

        # Store with force=True to overwrite
        kt.put(
            key=f"{service_name}/config/main/config.yaml",
            src=str(file_v1),
            force=True,
            filter_options="--include='*.yaml'",
        )

        # Verify overwrite - check in service's storage area
        updated_content = helper.read_file("config/main/config.yaml")
        assert "version: 2" in updated_content, "Force overwrite failed"
        assert "data: updated" in updated_content, "Content not fully updated"


@pytest.mark.level("minimal")
def test_store_error_handling_kv():
    """Test error handling for key-value operations."""
    from pathlib import Path

    from kubetorch.resources.compute.utils import RsyncError

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Test 1: Try to store non-existent file
        # This raises ValueError before rsync is called (file validation)
        with pytest.raises(ValueError, match="Could not locate path to sync up"):
            kt.put(key="test/nonexistent", src="/non/existent/file.txt")

        # Test 2: Try to retrieve non-existent key
        # This should raise RsyncError when rsync fails to find the remote file
        with pytest.raises(RsyncError):
            kt.get(key="nonexistent/key", dest=str(tmpdir))

        # Test 3: Invalid key formats should still work (treated as paths)
        test_file = tmpdir / "test.txt"
        test_file.write_text("test")

        # These should work - keys are flexible
        kt.put(key="../test", src=str(test_file))  # Will store under /data/store/../test
        kt.put(key="/absolute/path", src=str(test_file))  # Will store under /data/store/absolute/path

        # Clean up test files
        try:
            kt.rm(key="../test", recursive=True)
            kt.rm(key="/absolute/path", recursive=True)
        except Exception:
            # Ignore cleanup errors - files may not exist or may have been cleaned up already
            pass


@pytest.mark.level("minimal")
async def test_store_ls(store_test_helper):
    """Test kt.ls for listing store contents."""
    from pathlib import Path

    helper = store_test_helper
    service_name = helper.service_name

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

        # Upload files to create a structure
        kt.put(key=f"{service_name}/ls-test/file1.txt", src=str(test_file1))
        kt.put(key=f"{service_name}/ls-test/file2.txt", src=str(test_file2))
        kt.put(key=f"{service_name}/ls-test/subdir", src=str(test_dir))

        # Test listing from outside the cluster - need to specify service name in key
        # List service-specific path (with verbose to see what's happening)
        items = kt.ls(f"{service_name}/ls-test", verbose=True)
        assert len(items) >= 3, f"Should list uploaded files and directories, got: {items}"
        assert any("file1.txt" in item for item in items), "file1.txt should be listed"
        assert any("file2.txt" in item for item in items), "file2.txt should be listed"
        assert any("subdir" in item for item in items), "subdir should be listed"

        # Test listing from inside the cluster (via helper)
        # The helper runs inside the cluster, so it can list relative paths
        items = helper.list_store_contents("ls-test")
        assert len(items) >= 3, "Helper should list uploaded files and directories"
        assert any("file1.txt" in item for item in items), "file1.txt should be listed via helper"
        assert any("file2.txt" in item for item in items), "file2.txt should be listed via helper"
        assert any("subdir" in item for item in items), "subdir should be listed via helper"


@pytest.mark.level("minimal")
async def test_store_single_file_contents_flag(store_test_helper):
    """Test single file upload with contents=True flag."""
    from pathlib import Path

    helper = store_test_helper
    service_name = helper.service_name

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a single file
        test_file = tmpdir / "source.txt"
        test_file.write_text("source content")

        # Upload with contents=True - should put file inside directory at key
        kt.put(key=f"{service_name}/single-contents/dir", src=str(test_file), contents=True)

        # Verify - file should be inside the directory
        download_dir = tmpdir / "verify"
        download_dir.mkdir()
        kt.get(key=f"{service_name}/single-contents/dir", dest=str(download_dir))

        # File should be at dir/source.txt (original filename preserved)
        assert (download_dir / "dir" / "source.txt").exists(), "File should be inside directory with original name"


@pytest.mark.level("minimal")
async def test_store_file_renaming(store_test_helper):
    """Test that files are renamed to match the key when uploaded."""
    from pathlib import Path

    helper = store_test_helper
    service_name = helper.service_name

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a file with one name
        source_file = tmpdir / "original_name.txt"
        source_file.write_text("renamed content")

        # Upload to a key with different name - file should be renamed
        kt.put(key=f"{service_name}/renamed/new_name.txt", src=str(source_file))

        # Download and verify it's renamed
        download_dir = tmpdir / "download"
        download_dir.mkdir()
        kt.get(key=f"{service_name}/renamed/new_name.txt", dest=str(download_dir))

        # Should be downloaded as new_name.txt (not original_name.txt)
        assert (download_dir / "new_name.txt").exists(), "File should be renamed to match key"
        assert not (download_dir / "original_name.txt").exists(), "Original filename should not exist"
        assert "renamed content" in (download_dir / "new_name.txt").read_text(), "Content should be preserved"
