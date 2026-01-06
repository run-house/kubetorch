"""
Utilities for parsing and handling storage keys.

Keys follow a simple convention:
- Keys are structured as: service_name/path
- Leading and trailing slashes are stripped
- The first segment is always interpreted as the service_name
- Empty keys result in service_name=None and path=""

Examples:
    "my-service/models/v1" -> service_name="my-service", path="models/v1"
    "/my-service/models" -> service_name="my-service", path="models"
    "my-service" -> service_name="my-service", path=""
    "" -> service_name=None, path=""
"""

from dataclasses import dataclass
from typing import Optional


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


def parse_key(key: str) -> ParsedKey:
    """
    Parse a storage key into its components.

    Keys are always parsed literally - there is no auto-prepending of service names.
    The first segment of the key is treated as the service_name, and the rest is the path.

    Args:
        key (str): The storage key to parse (leading/trailing slashes are stripped).

    Returns:
        ParsedKey with service_name (or None) and path.

    Examples:
        >>> parse_key("my-service/models/v1")
        ParsedKey(service_name="my-service", path="models/v1")

        >>> parse_key("/shared/dataset")
        ParsedKey(service_name="shared", path="dataset")

        >>> parse_key("my-service")
        ParsedKey(service_name="my-service", path="")

        >>> parse_key("")
        ParsedKey(service_name=None, path="")
    """
    original_key = key

    # Strip leading and trailing slashes
    key = key.strip("/")

    # Empty key
    if not key:
        return ParsedKey(service_name=None, path="", original_key=original_key)

    # Split into service_name and path
    if "/" in key:
        service_name, path = key.split("/", 1)
        return ParsedKey(
            service_name=service_name,
            path=path,
            original_key=original_key,
        )
    else:
        # Single segment - treat as service_name with empty path
        return ParsedKey(
            service_name=key,
            path="",
            original_key=original_key,
        )
