"""
Type definitions for data store operations.

This module defines the core types used by the data store API:
- BroadcastWindow: Configuration for coordinated multi-party data transfers
- Locale: Where data is stored (store pod vs local pod)
- Lifespan: How long data persists (cluster-wide vs resource-scoped)
"""

from dataclasses import dataclass
from typing import List, Literal, Optional

# Type aliases for clarity
Locale = Literal["store", "local"]
"""Where data is stored/accessible from.

- "store": Copy data to the central store pod (default). Data is persisted and
  accessible from any pod via the store.
- "local": Zero-copy mode. Data stays on the local pod and is only registered
  with the metadata server. Other pods can rsync directly from this pod.
"""

Lifespan = Literal["cluster", "resource"]
"""How long data persists.

- "cluster": Data persists until explicitly deleted. Can use global keys
  without service prefix.
- "resource": Data is stored under the service's key directory and automatically
  cleaned up when the service is torn down.
"""


@dataclass
class BroadcastWindow:
    """Configuration for coordinated broadcast transfers between multiple participants.

    A broadcast window allows multiple putters and getters to coordinate data
    transfer. The quorum closes when ANY of the specified conditions is met
    (OR semantics).

    When a broadcast window is specified:
    - put() calls join as "putters" (data sources)
    - get() calls join as "getters" (data destinations)
    - Once the quorum closes, putters send data to all getters

    Attributes:
        timeout: Maximum time in seconds to wait for participants. The quorum
            closes after this timeout even if other conditions aren't met.
        world_size: Wait for this many total participants (putters + getters)
            before closing the quorum.
        ips: Wait for participants from these specific IP addresses before
            closing the quorum.
        group_id: Optional name for the broadcast group. If not provided, one
            is auto-generated from the keys being transferred. Use the same
            group_id across put/get calls to ensure they join the same quorum.

    Examples:
        # Wait up to 10 seconds for participants
        BroadcastWindow(timeout=10.0)

        # Wait for exactly 4 participants (e.g., 1 putter + 3 getters)
        BroadcastWindow(world_size=4)

        # Wait for specific pods to join
        BroadcastWindow(ips=["10.0.0.1", "10.0.0.2", "10.0.0.3"])

        # Combined: wait for 4 participants OR 30 seconds, whichever first
        BroadcastWindow(world_size=4, timeout=30.0)
    """

    timeout: Optional[float] = None
    world_size: Optional[int] = None
    ips: Optional[List[str]] = None
    group_id: Optional[str] = None

    def __post_init__(self):
        """Validate that at least one condition is specified."""
        if self.timeout is None and self.world_size is None and self.ips is None:
            raise ValueError("BroadcastWindow requires at least one of: timeout, world_size, or ips")

    def to_dict(self) -> dict:
        """Convert to dictionary for API requests."""
        return {
            "timeout": self.timeout,
            "world_size": self.world_size,
            "ips": self.ips,
            "group_id": self.group_id,
        }
