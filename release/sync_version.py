#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = REPO_ROOT / "VERSION"
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync the Kubetorch repo version across release artifacts.")
    parser.add_argument("version", nargs="?", help="Version to write. Defaults to the current VERSION file.")
    return parser.parse_args()


def read_version() -> str:
    return VERSION_FILE.read_text().strip()


def write_version(version: str) -> None:
    VERSION_FILE.write_text(f"{version}\n")


def replace_in_file(path: Path, pattern: str, replacement: str, *, count: int = 0) -> None:
    contents = path.read_text()
    updated, replacements = re.subn(pattern, replacement, contents, count=count, flags=re.MULTILINE)
    if replacements == 0:
        raise RuntimeError(f"Pattern not found in {path}: {pattern}")
    path.write_text(updated)


def main() -> None:
    args = parse_args()
    version = args.version or read_version()
    if not SEMVER_RE.match(version):
        raise SystemExit(f"Invalid version '{version}'. Expected semver like 0.5.0")

    write_version(version)

    replace_in_file(
        REPO_ROOT / "python_client/pyproject.toml",
        r'(?m)^version = "[^"]+"$',
        f'version = "{version}"',
        count=1,
    )
    replace_in_file(
        REPO_ROOT / "python_client/kubetorch/_version.py",
        r'(?m)^__version__ = "[^"]+"$',
        f'__version__ = "{version}"',
        count=1,
    )
    replace_in_file(
        REPO_ROOT / "charts/kubetorch/Chart.yaml",
        r"(?m)^version: .+$",
        f"version: {version}",
        count=1,
    )
    replace_in_file(
        REPO_ROOT / "charts/kubetorch/Chart.yaml",
        r"(?m)^appVersion: .+$",
        f"appVersion: {version}",
        count=1,
    )
    replace_in_file(
        REPO_ROOT / "charts/kubetorch/values.yaml",
        r"(?m)^  tag: .+$",
        f"  tag: {version}",
        count=2,
    )
    replace_in_file(
        REPO_ROOT / "README.md",
        r"(?m)(--version )\S+",
        rf"\g<1>{version}",
    )
    replace_in_file(
        REPO_ROOT / "charts/kubetorch/README.md",
        r"(?m)Version-[0-9A-Za-z.+-]+-informational",
        f"Version-{version}-informational",
    )
    replace_in_file(
        REPO_ROOT / "charts/kubetorch/README.md",
        r"(?m)AppVersion-[0-9A-Za-z.+-]+-informational",
        f"AppVersion-{version}-informational",
    )
    replace_in_file(
        REPO_ROOT / "charts/kubetorch/README.md",
        r'(?m)^\| dataStore\.tag \| string \| `".*?"` \|  \|$',
        f'| dataStore.tag | string | `"{version}"` |  |',
    )
    replace_in_file(
        REPO_ROOT / "charts/kubetorch/README.md",
        r'(?m)^\| kubetorchController\.tag \| string \| `".*?"` \|  \|$',
        f'| kubetorchController.tag | string | `"{version}"` |  |',
    )
    replace_in_file(
        REPO_ROOT / "release/default_images/server-otel",
        r"(?m)^ARG BASE_IMAGE=.+$",
        f"ARG BASE_IMAGE=ghcr.io/run-house/server:{version}",
        count=1,
    )
    replace_in_file(
        REPO_ROOT / "release/default_images/ubuntu-otel",
        r"(?m)^ARG BASE_IMAGE=.+$",
        f"ARG BASE_IMAGE=ghcr.io/run-house/ubuntu:{version}",
        count=1,
    )

    print(version)


if __name__ == "__main__":
    main()
