"""
Utilities for parsing and handling storage keys.

Keys follow a consistent format:
- Service-specific: "service-name/path/to/file" -> stores under /data/{namespace}/{service-name}/path/to/file
- Store keys: "path/to/file" -> stores under /data/{namespace}/store/path/to/file
- Absolute paths: "/absolute/path" -> treated as path without service prefix

A key segment is considered a service name if it:
- Contains a hyphen ("-")
- Does not contain a dot (".")
- Does not start with "." or "/"
"""

import os
from dataclasses import dataclass
from typing import Optional

from kubetorch.servers.http.utils import is_running_in_kubernetes


@dataclass
class ParsedKey:
    """Result of parsing a storage key."""

    service_name: Optional[str]
    path: str
    original_key: str

    @property
    def storage_path(self) -> str:
        """Get the path portion for rsync operations."""
        if self.service_name:
            return self.path
        return f"store/{self.path}" if self.path else "store"

    @property
    def full_key(self) -> str:
        """Get the full key including service name if present."""
        if self.service_name:
            return f"{self.service_name}/{self.path}" if self.path else self.service_name
        return self.path


def is_service_name(segment: str) -> bool:
    """
    Check if a string segment looks like a service name.

    Service names typically:
    - Contain hyphens (e.g., "my-service", "store-test-helper")
    - Don't contain dots (to distinguish from filenames like "model.pkl")
    - Don't start with "." or "/"
    """
    if not segment:
        return False
    if segment.startswith(".") or segment.startswith("/"):
        return False
    if "." in segment:
        return False
    if "-" not in segment:
        return False
    return True


def parse_key(
    key: str,
    auto_prepend_service: bool = False,
    in_cluster_service_name: Optional[str] = None,
) -> ParsedKey:
    """
    Parse a storage key into its components.

    Args:
        key: The storage key to parse (trailing slashes are stripped)
        auto_prepend_service: If True and running in-cluster, auto-prepend service name
        in_cluster_service_name: Override for KT_SERVICE_NAME (for testing)

    Returns:
        ParsedKey with service_name (or None) and path

    Examples:
        >>> parse_key("my-service/models/v1.pkl")
        ParsedKey(service_name="my-service", path="models/v1.pkl")

        >>> parse_key("datasets/train.csv")
        ParsedKey(service_name=None, path="datasets/train.csv")

        >>> parse_key("/absolute/path")
        ParsedKey(service_name=None, path="absolute/path")

        >>> parse_key("my-service")
        ParsedKey(service_name="my-service", path="")
    """
    # Strip trailing slashes - keys are abstract, not directory paths
    key = key.rstrip("/")
    original_key = key

    # Handle absolute paths - strip leading slash, no service name
    if key.startswith("/"):
        return ParsedKey(service_name=None, path=key[1:], original_key=original_key)

    # Parse key to extract service name first (before auto-prepend check)
    parsed_service_name = None
    parsed_path = key

    if "/" in key:
        parts = key.split("/", 1)
        first_part = parts[0]
        rest = parts[1] if len(parts) > 1 else ""

        if is_service_name(first_part):
            parsed_service_name = first_part
            parsed_path = rest
    elif is_service_name(key):
        # Single segment that looks like a service name
        parsed_service_name = key
        parsed_path = ""

    # Check if we should auto-prepend service name for in-cluster relative keys
    if auto_prepend_service:
        kt_service_name = in_cluster_service_name or os.getenv("KT_SERVICE_NAME")
        in_cluster = is_running_in_kubernetes()
        if kt_service_name and in_cluster:
            # When auto_prepend_service=True, prepend the service name UNLESS:
            # 1. Key already starts with the current service name (e.g. "donny-my-svc/path")
            # 2. Key exactly equals the current service name
            # 3. Key already has a service-name-like prefix (for cross-service access)
            #
            # A "service-name-like prefix" means the key has a "/" and the first
            # segment contains multiple hyphens (typical k8s service naming pattern
            # like "user-service-name"). Simple paths like "test-files/foo.txt"
            # only have one hyphen so will be prepended.

            # Check if key already starts with current service name
            starts_with_own_service = key == kt_service_name or key.startswith(f"{kt_service_name}/")

            if starts_with_own_service:
                # Key already has the current service name prefix
                if key == kt_service_name:
                    return ParsedKey(
                        service_name=kt_service_name,
                        path="",
                        original_key=original_key,
                    )
                else:
                    # Strip the service name prefix
                    remaining_path = key[len(kt_service_name) + 1 :]  # +1 for the "/"
                    return ParsedKey(
                        service_name=kt_service_name,
                        path=remaining_path,
                        original_key=original_key,
                    )

            # Check if key has a different service name prefix (cross-service access)
            # Service names in kubetorch follow k8s naming and typically have
            # format "user-service-name" with multiple hyphens
            if "/" in key:
                first_segment = key.split("/", 1)[0]
                # Count hyphens - real service names typically have 2+ hyphens
                # (e.g., "donny-store-test-helper" has 3, "donny-my-svc" has 2)
                # while paths like "test-files" or "ls-test" have only 1
                hyphen_count = first_segment.count("-")
                if hyphen_count >= 2 and is_service_name(first_segment):
                    # Likely a cross-service key like "other-svc-name/path"
                    return ParsedKey(
                        service_name=first_segment,
                        path=key.split("/", 1)[1] if "/" in key else "",
                        original_key=original_key,
                    )

            # Prepend the current service name
            return ParsedKey(
                service_name=kt_service_name,
                path=key,
                original_key=original_key,
            )

    # Return parsed result for non-auto-prepend case
    if parsed_service_name:
        return ParsedKey(
            service_name=parsed_service_name,
            path=parsed_path,
            original_key=original_key,
        )

    return ParsedKey(service_name=None, path=key, original_key=original_key)


def key_to_metadata_key(key: str, auto_prepend_service: bool = False) -> str:
    """
    Convert a user-provided key to the format expected by the metadata server.

    The metadata server expects keys in the format: "service-name/path" or "path"
    This function ensures consistent key formatting.

    Args:
        key: User-provided storage key
        auto_prepend_service: If True and in-cluster, prepend service name

    Returns:
        Key formatted for metadata server
    """
    parsed = parse_key(key, auto_prepend_service=auto_prepend_service)
    return parsed.full_key
