"""
Utilities for parsing and handling storage keys.

Keys follow a simple convention:
- Absolute keys (starting with "/"): Used as-is, no service name prepending
  - Example: "/shared/dataset" -> stores under /data/{namespace}/store/shared/dataset
  - Example: "/my-service/models" -> stores under /data/{namespace}/my-service/models
- Relative keys (not starting with "/"): Service name is prepended when in-cluster
  - Example: "models/v1.pkl" -> stores under /data/{namespace}/{service-name}/models/v1.pkl

When running inside a kubetorch service pod (KT_SERVICE_NAME is set):
- Relative keys automatically get the service name prepended (for put operations)
- Use absolute keys (starting with "/") to access shared/cross-service data
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


def parse_key(
    key: str,
    auto_prepend_service: bool = False,
    in_cluster_service_name: Optional[str] = None,
) -> ParsedKey:
    """
    Parse a storage key into its components.

    Simple convention:
    - Absolute keys (starting with "/"): No service name prepending, first segment is service_name
    - Relative keys: When auto_prepend_service=True and in-cluster, prepend service name

    Args:
        key: The storage key to parse (trailing slashes are stripped)
        auto_prepend_service: If True and running in-cluster, auto-prepend service name
            to relative keys
        in_cluster_service_name: Override for KT_SERVICE_NAME (for testing)

    Returns:
        ParsedKey with service_name (or None) and path

    Examples:
        # Absolute key - first segment becomes service_name
        >>> parse_key("/my-service/models")
        ParsedKey(service_name="my-service", path="models")

        # Absolute key with no slash after service - just service_name
        >>> parse_key("/my-service")
        ParsedKey(service_name="my-service", path="")

        # Relative key in-cluster with auto_prepend - service name prepended
        >>> parse_key("models/v1.pkl", auto_prepend_service=True)  # with KT_SERVICE_NAME=my-svc
        ParsedKey(service_name="my-svc", path="models/v1.pkl")

        # Relative key without auto_prepend - first segment is service_name
        >>> parse_key("my-service/models", auto_prepend_service=False)
        ParsedKey(service_name="my-service", path="models")
    """
    # Strip trailing slashes - keys are abstract, not directory paths
    key = key.rstrip("/")
    original_key = key

    # Handle absolute paths - strip leading slash, first segment is service_name
    if key.startswith("/"):
        path = key[1:]
        if "/" in path:
            service_name, rest = path.split("/", 1)
            return ParsedKey(service_name=service_name, path=rest, original_key=original_key)
        elif path:
            # Just service name, no additional path
            return ParsedKey(service_name=path, path="", original_key=original_key)
        else:
            # Just "/" - empty path
            return ParsedKey(service_name=None, path="", original_key=original_key)

    # For relative keys, check if we should auto-prepend service name
    if auto_prepend_service:
        kt_service_name = in_cluster_service_name or os.getenv("KT_SERVICE_NAME")
        in_cluster = is_running_in_kubernetes()

        if kt_service_name and in_cluster:
            # Check if key already starts with the current service name
            if key == kt_service_name:
                return ParsedKey(
                    service_name=kt_service_name,
                    path="",
                    original_key=original_key,
                )
            elif key.startswith(f"{kt_service_name}/"):
                # Key already has service name prefix - don't double-prepend
                remaining_path = key[len(kt_service_name) + 1 :]
                return ParsedKey(
                    service_name=kt_service_name,
                    path=remaining_path,
                    original_key=original_key,
                )
            else:
                # Prepend the current service name to relative key
                return ParsedKey(
                    service_name=kt_service_name,
                    path=key,
                    original_key=original_key,
                )

    # For non-auto-prepend case or when not in cluster, first segment is service_name
    if "/" in key:
        service_name, rest = key.split("/", 1)
        return ParsedKey(
            service_name=service_name,
            path=rest,
            original_key=original_key,
        )
    elif key:
        # Single segment - treat as service_name
        return ParsedKey(
            service_name=key,
            path="",
            original_key=original_key,
        )

    # Empty key
    return ParsedKey(service_name=None, path="", original_key=original_key)
