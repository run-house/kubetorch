"""Rsync verification utilities for testing."""
import os
from pathlib import Path
from typing import Dict, List


class RsyncVerifier:
    """Class to verify rsync operations on remote cluster."""

    def verify_single_file(self, file_path: str, expected_content: str) -> Dict:
        """Verify a single file exists and has expected content."""
        try:
            with open(file_path, "r") as f:
                actual_content = f.read()
                return {
                    "path": file_path,
                    "exists": True,
                    "content_match": expected_content in actual_content,
                    "actual_content": actual_content[:100],  # First 100 chars for debugging
                }
        except FileNotFoundError:
            return {
                "path": file_path,
                "exists": False,
                "content_match": False,
                "actual_content": None,
            }

    def verify_directory_structure(self, base_path: str, expected_files: List[str]) -> Dict:
        """Verify a directory structure exists with expected files."""
        base = Path(base_path)
        results = {"base_exists": base.exists() and base.is_dir(), "files": {}}

        for expected_file in expected_files:
            full_path = base / expected_file
            results["files"][expected_file] = {
                "exists": full_path.exists(),
                "is_file": full_path.is_file() if full_path.exists() else False,
            }

        # Also list what actually exists for debugging
        if base.exists():
            actual_files = []
            for root, dirs, files in os.walk(base):
                root_path = Path(root)
                for file in files:
                    rel_path = (root_path / file).relative_to(base)
                    actual_files.append(str(rel_path))
            results["actual_files"] = sorted(actual_files)
        else:
            results["actual_files"] = []

        return results

    def test_all_scenarios(self) -> Dict:
        """Test all rsync scenarios that were set up."""
        results = {}

        # Test 1: Single file with relative destination
        results["single_relative"] = self.verify_single_file(
            "relative_dest/single_file.txt", "This is a single test file"
        )

        # Test 2: Single file with absolute destination
        results["single_absolute"] = self.verify_single_file(
            "/data/absolute_dest/single_file.txt", "This is a single test file"
        )

        # Test 3: Directory with contents=False (directory itself is copied)
        results["dir_no_contents"] = self.verify_directory_structure(
            "copied_dir/test_dir",
            ["file1.txt", "file2.txt", "nested/deep.txt", "nested/another.txt"],
        )

        # Test 4: Directory with contents=True (only contents copied)
        results["dir_with_contents"] = self.verify_directory_structure(
            "contents_only",
            ["file1.txt", "file2.txt", "nested/deep.txt", "nested/another.txt"],
        )

        # Test 5: Absolute source and destination
        results["absolute_both"] = self.verify_directory_structure(
            "/data/rsync_test",
            ["single_file.txt", "test_dir/file1.txt", "test_dir/nested/deep.txt"],
        )

        # Test 6: Tilde in destination (treated as relative to home)
        results["tilde_home"] = self.verify_single_file("from_home/home_test.txt", "File from home test assets")

        # Test 7: No dest specified for a file (should be in cwd with basename)
        results["no_dest_file"] = self.verify_single_file("single_file.txt", "This is a single test file")

        # Test 8: No dest specified for a directory (should be in cwd with dir name)
        results["no_dest_dir"] = self.verify_directory_structure(
            "test_dir",
            ["file1.txt", "file2.txt", "nested/deep.txt", "nested/another.txt"],
        )

        # Test 9: No dest with absolute source (should use basename)
        results["no_dest_absolute"] = self.verify_single_file(
            "single_file.txt",  # Same as test 7, should end up in same place
            "This is a single test file",
        )

        return results
