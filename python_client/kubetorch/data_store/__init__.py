"""
Data store utilities for kubetorch.

An intuitive, scalable distributed data system for Kubernetes, solving two critical gaps:

1. **Out-of-cluster direct transfers**: Sync code and data to your cluster instantly and scalably - no container rebuilds
2. **In-cluster data transfer and caching**: Peer-to-peer data transfer between pods with automatic caching and discovery, for filesystem and GPU data

The unified put()/get() API handles two data types (auto-detected from parameters):
- **Filesystem data**: Files/directories via distributed rsync (zero-copy P2P with locale="local" or central store)
- **GPU data**: CUDA tensors/state dicts via NCCL broadcast

Key capabilities:
- External sync: Push/pull files to/from cluster
- Zero-copy P2P: locale="local" publishes data in-place, consumers fetch directly
- Scalable broadcast: Tree-based propagation for distributing to thousands of pods
- GPU transfers: Point-to-point and coordinated broadcast for tensors/state dicts with Infiniband/RDMA support
- Automatic caching: Every getter can become a source for subsequent getters

Example usage::

    import kubetorch as kt

    # External sync (to/from cluster)
    kt.put(key="my-service/weights", src="./model/")
    kt.get(key="my-service/weights", dest="./local/")

    # P2P within cluster (zero-copy)
    kt.put(key="features", src="/data/", locale="local")  # On producer pod
    kt.get(key="features", dest="/local/")                # On consumer pods

    # GPU tensor transfer
    kt.put(key="layer1", src=cuda_tensor)                 # On producer pod
    kt.get(key="layer1", dest=dest_tensor)                # On consumer pod
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
