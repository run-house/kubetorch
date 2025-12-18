"""
Data store utilities for kubetorch.

This package provides a key-value store interface for transferring data to and from the cluster.
It supports both filesystem data (rsync-based transfer) and GPU data (NCCL broadcast-based transfer).

The data type is auto-detected from the `src`/`dest` parameters:
- put(key, src=path) - upload filesystem data
- put(key, src=tensor) - publish GPU tensor or state dict
- get(key, dest=path) - download filesystem data
- get(key, dest=tensor) - retrieve GPU tensor or state dict into pre-allocated tensor
"""

from .data_store_client import DataStoreClient, DataStoreError
from .data_store_cmds import _sync_workdir_from_store, get, ls, put, rm  # Internal use only
from .key_utils import parse_key, ParsedKey
from .rsync_client import RsyncClient
from .types import BroadcastWindow, Lifespan, Locale

__all__ = [
    "BroadcastWindow",
    "DataStoreClient",
    "DataStoreError",
    "Lifespan",
    "Locale",
    "ParsedKey",
    "RsyncClient",
    "_sync_workdir_from_store",  # Internal use only
    "get",
    "ls",
    "parse_key",
    "put",
    "rm",
]
