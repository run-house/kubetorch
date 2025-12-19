"""
Helper class for store operations testing on remote cluster.

This helper runs inside a Kubernetes pod and provides methods for:
- Verifying file uploads
- Creating test files for downloads
- Testing locale="local" / peer-to-peer operations
- Checking metadata server state
"""

import os
from pathlib import Path
from typing import Optional

import kubetorch as kt


class StoreTestHelper:
    """Helper class for store operations testing on remote cluster."""

    @property
    def service_name(self) -> str:
        """Get the service name from KT_SERVICE_NAME environment variable."""
        service_name = os.getenv("KT_SERVICE_NAME")
        if not service_name:
            raise RuntimeError("KT_SERVICE_NAME environment variable not set")
        return service_name

    def _key(self, path: str) -> str:
        """Build a full key with service name prefix.

        Keys are now explicit - this helper prefixes with service name for convenience.
        """
        if not path:
            return self.service_name
        if path.startswith("/"):
            return path
        # Simple prefix with service name
        return f"{self.service_name}/{path}"

    # ==================== Upload Verification ====================

    def verify_uploaded_files(self) -> dict:
        """Verify files uploaded via kt.put by retrieving them from the store."""
        import tempfile

        results = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            results["single_file"] = self._verify_single_file(tmpdir)
            results["directory"] = self._verify_directory(tmpdir)
            results["multiple_files"] = self._verify_multiple_files(tmpdir)
            results["contents_flag"] = self._verify_contents_flag(tmpdir)

        return results

    def _verify_single_file(self, tmpdir: Path) -> dict:
        """Verify single file upload."""
        try:
            dest = tmpdir / "single"
            dest.mkdir()
            kt.get(self._key("test-files/single.txt"), dest=str(dest))
            single_file = dest / "single.txt"
            if single_file.exists():
                content = single_file.read_text()
                return {"exists": True, "content": content, "correct": "single file content" in content}
        except Exception:
            pass
        return {"exists": False, "correct": False}

    def _verify_directory(self, tmpdir: Path) -> dict:
        """Verify directory upload."""
        try:
            dest = tmpdir / "test-dir-check"
            dest.mkdir()
            kt.get(self._key("test-dir"), dest=str(dest))
            dir_path = dest / "test-dir"
            if dir_path.exists():
                files = list(dir_path.rglob("*"))
                return {
                    "exists": True,
                    "file_count": len([f for f in files if f.is_file()]),
                    "has_nested": (dir_path / "test_dir" / "nested" / "deep.txt").exists(),
                }
        except Exception:
            pass
        return {"exists": False}

    def _verify_multiple_files(self, tmpdir: Path) -> dict:
        """Verify multiple files upload."""
        try:
            dest = tmpdir / "multi"
            dest.mkdir()
            kt.get(self._key("multi-files"), dest=str(dest))
            multi_path = dest / "multi-files"
            return {
                "file1": (multi_path / "file1.txt").exists() if multi_path.exists() else False,
                "file2": (multi_path / "file2.txt").exists() if multi_path.exists() else False,
                "file3": (multi_path / "file3.txt").exists() if multi_path.exists() else False,
            }
        except Exception:
            return {"file1": False, "file2": False, "file3": False}

    def _verify_contents_flag(self, tmpdir: Path) -> dict:
        """Verify contents flag behavior."""
        try:
            dest = tmpdir / "contents"
            dest.mkdir()
            kt.get(self._key("contents-test"), dest=str(dest), contents=True)
            files_in_dir = list(dest.iterdir())
            if files_in_dir:
                return {
                    "exists": True,
                    "file1_direct": (dest / "file1.txt").exists(),
                    "no_subdir": not (dest / "test_dir").exists(),
                }
        except Exception:
            pass
        return {"exists": False}

    # ==================== Download Preparation ====================

    def prepare_download_files(self) -> dict:
        """Prepare files for download testing."""
        # Create test files on the remote
        os.makedirs("downloads", exist_ok=True)
        with open("downloads/result.csv", "w") as f:
            f.write("id,value\n1,100\n2,200\n")
        kt.put(self._key("downloads/result.csv"), src="downloads/result.csv", filter_options="--include='*.csv'")

        # Directory with nested structure
        os.makedirs("models/v1", exist_ok=True)
        with open("models/model.pkl", "w") as f:
            f.write("model data")
        with open("models/v1/weights.bin", "w") as f:
            f.write("weights data")
        kt.put(self._key("models"), src="models/", filter_options="--include='*.pkl'")

        # Log files
        os.makedirs("logs", exist_ok=True)
        for i in range(3):
            with open(f"logs/output_{i}.log", "w") as f:
                f.write(f"Log file {i} content\n")
        kt.put(self._key("logs"), src="logs/", filter_options="--include='*.log'")

        return {"prepared": True}

    # ==================== File Operations ====================

    def check_file_exists(self, key: str) -> dict:
        """Check if a file exists in the store and return its content."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                kt.get(key=self._key(key), dest=tmpdir)
                files = list(Path(tmpdir).rglob("*"))
                for f in files:
                    if f.is_file():
                        return {"exists": True, "content": f.read_text()}
            except Exception:
                pass
        return {"exists": False}

    def create_output_file(self, key: str, content: str) -> str:
        """Create an output file and put it in the store."""
        local_path = Path("temp_output.txt")
        local_path.write_text(content)
        filter_opts = "--include='*.log'" if key.endswith(".log") else None
        kt.put(key=self._key(key), src=str(local_path), filter_options=filter_opts)
        local_path.unlink()
        return "created"

    def read_file(self, key: str) -> str:
        """Read and return file contents from the store."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                kt.get(key=self._key(key), dest=tmpdir, filter_options="--include='*.yaml'")
                for f in Path(tmpdir).rglob("*"):
                    if f.is_file():
                        return f.read_text()
            except Exception:
                pass
        return "not found"

    def check_hierarchy(self) -> dict:
        """Check hierarchical storage structure by retrieving from store."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            try:
                kt.get(
                    self._key("ml-project"),
                    dest=str(tmpdir),
                    filter_options="--include='*.pkl' --include='*.csv'",
                )
                base = tmpdir / "ml-project"
                return {
                    "models_v1": (base / "models" / "v1.pkl").exists(),
                    "models_v2": (base / "models" / "v2.pkl").exists(),
                    "datasets_train": (base / "datasets" / "train.csv").exists(),
                    "datasets_test": (base / "datasets" / "test.csv").exists(),
                }
            except Exception:
                return {
                    "models_v1": False,
                    "models_v2": False,
                    "datasets_train": False,
                    "datasets_test": False,
                }

    def list_store_contents(self, key: str = "") -> list:
        """List contents of a key in the store (called from inside cluster)."""
        return kt.ls(self._key(key) if key else self.service_name)

    # ==================== Local Publish / Peer-to-Peer Operations ====================

    def publish_data_local(self, key: str, local_path: str, content: Optional[str] = None) -> dict:
        """
        Publish data using locale="local" (zero-copy).

        Creates the data locally on the pod and publishes it.
        """
        path = Path(local_path)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content or f"Published data for {key}\nTest content")

        try:
            kt.put(key=key, src=local_path, locale="local", verbose=True)
            return {
                "success": True,
                "pod_ip": os.getenv("POD_IP", "unknown"),
                "key": key,
                "content": path.read_text() if path.exists() else None,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_data_from_store(self, key: str, dest_path: str, filter_options: Optional[str] = None) -> dict:
        """Retrieve data from the store using kt.get() (called from inside the pod)."""
        try:
            kt.get(key=self._key(key), dest=dest_path, filter_options=filter_options, verbose=True)

            dest = Path(dest_path)
            if dest.exists():
                if dest.is_file():
                    return {"success": True, "file_count": 1, "content": dest.read_text()}
                else:
                    files = [str(f) for f in dest.rglob("*") if f.is_file()]
                    return {"success": True, "file_count": len(files), "files": files}
            return {"success": False, "error": "Destination path does not exist"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def check_metadata_server(self, key: str) -> dict:
        """Check if a key is registered with the metadata server.

        Note: This expects a full key (with service_name prefix) since publish_data_local
        also expects a full key.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                kt.get(key=key, dest=tmpdir, verbose=True)
                files = list(Path(tmpdir).rglob("*"))
                return {
                    "registered": True,
                    "retrievable": len(files) > 0,
                    "file_count": len([f for f in files if f.is_file()]),
                }
            except Exception as e:
                return {"registered": False, "error": str(e)}

    # ==================== Filesystem Broadcast Operations ====================

    def get_with_fs_broadcast(
        self,
        key: str,
        dest_path: str,
        group_id: str,
        fanout: int = 1,
        timeout: float = 30.0,
    ) -> dict:
        """
        Retrieve data using filesystem broadcast (tree-based p2p propagation).

        Args:
            key: Full key to retrieve
            dest_path: Local destination path
            group_id: Broadcast group identifier
            fanout: Tree fanout (1 = linear chain, 50 = wide tree)
            timeout: Max time to wait
        """
        from kubetorch.data_store.types import BroadcastWindow

        try:
            dest = Path(dest_path)
            dest.mkdir(parents=True, exist_ok=True)

            broadcast = BroadcastWindow(
                group_id=group_id,
                fanout=fanout,
                timeout=timeout,
                world_size=1,  # Rolling participation - just need 1 to proceed
            )

            kt.get(key=key, dest=str(dest), broadcast=broadcast, verbose=True)

            if dest.exists():
                files = [str(f) for f in dest.rglob("*") if f.is_file()]
                content = None
                if files:
                    content = Path(files[0]).read_text()
                return {
                    "success": True,
                    "file_count": len(files),
                    "files": files,
                    "content": content,
                    "pod_ip": os.getenv("POD_IP", "unknown"),
                }
            return {"success": False, "error": "Destination path does not exist"}
        except Exception as e:
            import traceback

            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    # ==================== Queue/Stream Operations ====================

    def queue_produce(self, key: str, items: list, timeout: float = 10.0) -> dict:
        """
        Produce items to a queue using kt.put with a Queue src.

        Args:
            key: Full key for the queue
            items: List of items to put (strings or bytes)
            timeout: Max time to wait for streaming to complete
        """
        from queue import Queue

        try:
            q = Queue()

            # Start producer thread
            thread = kt.put(key, src=q, verbose=True)

            # Put items
            for item in items:
                q.put(item)
            q.put(None)  # Sentinel to stop

            # Wait for thread to complete
            thread.join(timeout=timeout)

            return {
                "success": True,
                "items_sent": len(items),
                "key": key,
                "pod_ip": os.getenv("POD_IP", "unknown"),
            }
        except Exception as e:
            import traceback

            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def queue_consume(self, key: str, max_items: int = 100, timeout: float = 10.0) -> dict:
        """
        Consume items from a queue using kt.get with a Queue dest.

        Args:
            key: Full key for the queue
            max_items: Maximum number of items to consume
            timeout: Max time to wait for items
        """
        import time
        from queue import Empty, Queue

        try:
            q = Queue()

            # Start consumer thread
            _thread = kt.get(key, dest=q, verbose=True)  # noqa: F841

            # Collect items with timeout
            items = []
            start_time = time.time()

            while len(items) < max_items and (time.time() - start_time) < timeout:
                try:
                    item = q.get(timeout=1.0)
                    if item is None:
                        break
                    # Decode if bytes
                    if isinstance(item, bytes):
                        item = item.decode()
                    items.append(item)
                except Empty:
                    # Check if we have any items yet - if not, keep waiting
                    if items:
                        break
                    continue

            return {
                "success": True,
                "items_received": len(items),
                "items": items,
                "key": key,
                "pod_ip": os.getenv("POD_IP", "unknown"),
            }
        except Exception as e:
            import traceback

            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def queue_check_redis(self, key: str) -> dict:
        """
        Check if a queue exists in Redis and return its info.

        This directly queries Redis to verify the queue was created.
        """
        try:
            from kubetorch.data_store.queue_client import QueueClient

            namespace = os.getenv("KT_NAMESPACE", "default")
            client = QueueClient(namespace=namespace)

            length = client.length(key)

            return {
                "success": True,
                "exists": length >= 0,
                "length": length,
                "key": key,
            }
        except Exception as e:
            import traceback

            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
