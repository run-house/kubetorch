import os
from pathlib import Path

import kubetorch as kt


class StoreTestHelper:
    """Helper class for store operations testing on remote cluster."""

    def verify_uploaded_files(self) -> dict:
        """Verify files uploaded via kt.put by retrieving them from the store."""
        import tempfile

        results = {}

        # Create a temp directory for verification
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Check single file upload with key
            try:
                single_dir = tmpdir / "single"
                single_dir.mkdir()
                kt.get("test-files/single.txt", dest=str(single_dir))

                single_file = single_dir / "single.txt"
                if single_file.exists():
                    content = single_file.read_text()
                    results["single_file"] = {
                        "exists": True,
                        "content": content,
                        "correct": "single file content" in content,
                    }
                else:
                    results["single_file"] = {"exists": False, "correct": False}
            except Exception:
                results["single_file"] = {"exists": False, "correct": False}

            # Check directory upload with key
            try:
                dir_dest = tmpdir / "test-dir-check"
                dir_dest.mkdir()
                kt.get("test-dir", dest=str(dir_dest))

                dir_path = dir_dest / "test-dir"
                if dir_path.exists():
                    files = list(dir_path.rglob("*"))
                    results["directory"] = {
                        "exists": True,
                        "file_count": len([f for f in files if f.is_file()]),
                        "has_nested": (dir_path / "test_dir" / "nested" / "deep.txt").exists(),
                    }
                else:
                    results["directory"] = {"exists": False}
            except Exception:
                results["directory"] = {"exists": False}

            # Check multiple files upload
            try:
                multi_dir = tmpdir / "multi"
                multi_dir.mkdir()
                kt.get("multi-files", dest=str(multi_dir))

                multi_path = multi_dir / "multi-files"
                results["multiple_files"] = {
                    "file1": (multi_path / "file1.txt").exists() if multi_path.exists() else False,
                    "file2": (multi_path / "file2.txt").exists() if multi_path.exists() else False,
                    "file3": (multi_path / "file3.txt").exists() if multi_path.exists() else False,
                }
            except Exception:
                results["multiple_files"] = {"file1": False, "file2": False, "file3": False}

            # Check contents flag behavior
            try:
                contents_dir = tmpdir / "contents"
                contents_dir.mkdir()
                kt.get("contents-test", dest=str(contents_dir), contents=True)

                # With contents=True, files should be directly under contents_dir
                # Check if any files were downloaded (not just if directory exists)
                files_in_dir = list(contents_dir.iterdir())
                if files_in_dir:
                    # Should have files directly, not nested in directory
                    results["contents_flag"] = {
                        "exists": True,
                        "file1_direct": (contents_dir / "file1.txt").exists(),
                        "no_subdir": not (contents_dir / "test_dir").exists(),
                    }
                else:
                    results["contents_flag"] = {"exists": False}
            except Exception:
                results["contents_flag"] = {"exists": False}

        return results

    def prepare_download_files(self) -> dict:
        """Prepare files for download testing."""
        # Create test files on the remote under /data/store
        os.makedirs("downloads", exist_ok=True)

        # Single file with key
        with open("downloads/result.csv", "w") as f:
            f.write("id,value\n1,100\n2,200\n")
        # We need to bypass filters because our .gitignore might ignore .csv files
        kt.put("downloads/result.csv", src="downloads/result.csv", filter_options="--include='*.csv'")

        # Directory with nested structure
        os.makedirs("models/v1", exist_ok=True)
        with open("models/model.pkl", "w") as f:
            f.write("model data")
        with open("models/v1/weights.bin", "w") as f:
            f.write("weights data")
        # We need to bypass filters because our .gitignore ignores .pkl files
        kt.put("models", src="models/", filter_options="--include='*.pkl'")

        # Multiple individual files
        os.makedirs("logs", exist_ok=True)
        for i in range(3):
            with open(f"logs/output_{i}.log", "w") as f:
                f.write(f"Log file {i} content\n")
        # We need to bypass filters because our .gitignore ignores .log files
        kt.put("logs", src="logs/", filter_options="--include='*.log'")

        return {
            "prepared": True,
            "files": [
                "downloads/result.csv",
                "models/model.pkl",
                "models/v1/weights.bin",
                "logs/output_0.log",
                "logs/output_1.log",
                "logs/output_2.log",
            ],
        }

    def check_file_exists(self, key: str) -> dict:
        """Check if a file exists in the store and return its content."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # Try to get the file from store
                kt.get(key=key, dest=tmpdir)
                # Find the downloaded file
                files = list(Path(tmpdir).rglob("*"))
                if files and files[0].is_file():
                    return {"exists": True, "content": files[0].read_text()}
            except Exception:
                pass
        return {"exists": False}

    def create_output_file(self, key: str, content: str) -> str:
        """Create an output file and put it in the store."""
        # Create a local file first
        local_path = Path("temp_output.txt")
        local_path.write_text(content)

        # Put it into the store with the given key
        # Since we're in the service, it will auto-prepend our service name
        # We need to bypass filters because our .gitignore ignores .log files
        filter_opts = "--include='*.log'" if key.endswith(".log") else None
        kt.put(key=key, src=str(local_path), filter_options=filter_opts)

        # Clean up local file
        local_path.unlink()
        return "created"

    def check_hierarchy(self) -> dict:
        """Check hierarchical storage structure by retrieving from store."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            try:
                # Get the entire ml-project hierarchy
                # We need to bypass filters because our .gitignore ignores .pkl and .csv files
                kt.get("ml-project", dest=str(tmpdir), filter_options="--include='*.pkl' --include='*.csv'")

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

    def read_file(self, key: str) -> str:
        """Read and return file contents from the store."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # Try to get the file from store
                # We need to bypass filters because our .gitignore might ignore .yaml files
                kt.get(key=key, dest=tmpdir, filter_options="--include='*.yaml'")
                # Find the downloaded file
                files = list(Path(tmpdir).rglob("*"))
                if files and files[0].is_file():
                    return files[0].read_text()
            except Exception:
                pass
        return "not found"

    def list_store_contents(self, key: str = "") -> list:
        """List contents of a key in the store (called from inside cluster)."""
        # Since we're inside the cluster, kt.ls will auto-prepend service name
        return kt.ls(key)
