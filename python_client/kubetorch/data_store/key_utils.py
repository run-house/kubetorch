"""
Utilities for handling storage keys.

Keys are simple paths that map directly to filesystem locations.
Leading and trailing slashes are stripped for consistency.
"""


def normalize_key(key: str) -> str:
    """
    Normalize a storage key by stripping leading/trailing slashes.

    Args:
        key (str): The storage key to normalize.

    Returns:
        Normalized key string.

    Examples:
        >>> normalize_key("my-service/models/v1")
        'my-service/models/v1'

        >>> normalize_key("/shared/dataset/")
        'shared/dataset'

        >>> normalize_key("simple-key")
        'simple-key'

        >>> normalize_key("")
        ''
    """
    return key.strip("/")
