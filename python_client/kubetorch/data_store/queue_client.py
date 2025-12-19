"""
Client for queue/stream operations using Redis Streams.

This module provides direct Redis access for high-throughput queue operations.
The metadata server returns Redis connection info, then clients connect directly
for streaming data.

Usage with kt.put/get:
    import kubetorch as kt
    from queue import Queue

    # Create a Python Queue
    q = Queue()

    # Put items to the queue in background - kt.put returns immediately
    kt.put("logs/my_job", src=q)

    # Producer puts items
    q.put("log line 1")
    q.put("log line 2")
    q.put(None)  # Sentinel to signal end

    # Consumer reads from queue
    dest_q = Queue()
    kt.get("logs/my_job", dest=dest_q)
    while True:
        item = dest_q.get()
        if item is None:
            break
        print(item)
"""

import asyncio
import threading
from queue import Queue
from typing import Any, Callable, Iterator, Optional, Union
from urllib.parse import urlparse

from kubetorch.logger import get_logger
from kubetorch.servers.http.utils import is_running_in_kubernetes
from kubetorch.utils import http_to_ws

from .websocket_tunnel import TunnelManager

logger = get_logger(__name__)

# Default Redis port
REDIS_PORT = 6379


def _is_queue_data(obj: Any) -> bool:
    """Check if object is a queue-like object (Python Queue or asyncio.Queue).

    This is used by data_store_cmds.py to detect queue src/dest parameters.
    """
    # Check for standard library Queue
    if isinstance(obj, Queue):
        return True

    # Check for asyncio.Queue
    try:
        if isinstance(obj, asyncio.Queue):
            return True
    except Exception:
        pass

    # Check for duck-typing: has put and get methods
    if hasattr(obj, "put") and hasattr(obj, "get") and callable(obj.put) and callable(obj.get):
        # Exclude dicts and other mappings that might have put/get
        if not isinstance(obj, dict):
            return True

    return False


class QueueClient:
    """
    Direct Redis client for queue/stream operations.

    Uses Redis Streams for reliable queue semantics with MAXLEN for memory bounds.
    Supports both in-cluster and external access (via WebSocket tunnel).
    """

    def __init__(
        self,
        namespace: str,
        host: Optional[str] = None,
        port: int = REDIS_PORT,
    ):
        """
        Initialize the queue client.

        Args:
            namespace: Kubernetes namespace for the data store
            host: Redis host (defaults to data store service in namespace)
            port: Redis port (default 6379)
        """
        self.namespace = namespace
        self._explicit_host = host
        self._explicit_port = port
        self._redis = None
        self._stop_event = threading.Event()
        self._tunnel = None  # WebSocket tunnel for external access

    def _get_websocket_info(self) -> tuple:
        """Get websocket connection info for Redis tunnel."""
        from kubetorch import globals as kt_globals

        base_url = kt_globals.service_url()
        ws_url = f"{http_to_ws(base_url)}/redis/{self.namespace}/"
        parsed_url = urlparse(base_url)

        # Return a starting port for the tunnel - TunnelManager caches tunnels by
        # ws_url and reuses them across calls
        start_from = (parsed_url.port or 8000) + 100  # Offset from HTTP port
        return start_from, ws_url

    def _ensure_tunnel(self):
        """Ensure WebSocket tunnel is set up for external access."""
        if self._tunnel is None:
            start_port, ws_url = self._get_websocket_info()
            self._tunnel = TunnelManager.get_tunnel(ws_url, start_port)
            logger.debug(f"Using WebSocket tunnel on port {self._tunnel.local_port} for Redis")
        return self._tunnel

    @property
    def host(self) -> str:
        """Get the Redis host."""
        if self._explicit_host is not None:
            return self._explicit_host

        if is_running_in_kubernetes():
            return f"kubetorch-data-store.{self.namespace}.svc.cluster.local"
        else:
            # Outside cluster - use WebSocket tunnel, connect to localhost
            self._ensure_tunnel()
            return "127.0.0.1"

    @property
    def port(self) -> int:
        """Get the Redis port."""
        if self._explicit_host is not None:
            # Explicit host provided - use explicit port
            return self._explicit_port

        if is_running_in_kubernetes():
            return REDIS_PORT
        else:
            # Outside cluster - use tunnel's local port
            tunnel = self._ensure_tunnel()
            return tunnel.local_port

    @property
    def redis(self):
        """Get or create Redis client (lazy initialization)."""
        if self._redis is None:
            try:
                import redis

                self._redis = redis.Redis(
                    host=self.host,
                    port=self.port,
                    decode_responses=False,
                )
                # Test connection
                self._redis.ping()
                logger.debug(f"Connected to Redis at {self.host}:{self.port}")
            except ImportError:
                raise ImportError("redis-py is required for queue operations. " "Install it with: pip install redis")
            except Exception as e:
                raise ConnectionError(f"Failed to connect to Redis at {self.host}:{self.port}: {e}")
        return self._redis

    def put(
        self,
        key: str,
        data: Union[bytes, str],
        maxlen: int = 10000,
    ) -> str:
        """
        Put a single item to the queue.

        Args:
            key: Queue key (e.g., "logs/launch_123")
            data: Data to put (bytes or str)
            maxlen: Maximum queue length (older items evicted when exceeded)

        Returns:
            Message ID from Redis
        """
        if isinstance(data, str):
            data = data.encode()

        msg_id = self.redis.xadd(key, {"data": data}, maxlen=maxlen)
        return msg_id.decode() if isinstance(msg_id, bytes) else msg_id

    def put_from_queue(
        self,
        key: str,
        src: Queue,
        maxlen: int = 10000,
        sentinel: object = None,
    ):
        """
        Stream items from a Python Queue to Redis Stream.

        Args:
            key: Queue key (e.g., "logs/launch_123")
            src: Source Python Queue
            maxlen: Maximum queue length
            sentinel: Value to stop streaming (default: None)
        """
        while not self._stop_event.is_set():
            try:
                item = src.get(timeout=1.0)
                if item is sentinel:
                    break
                if isinstance(item, str):
                    item = item.encode()
                self.redis.xadd(key, {"data": item}, maxlen=maxlen)
            except (ImportError, ConnectionError):
                # Fatal errors - re-raise immediately
                raise
            except Exception:
                # Queue.get timeout - check stop event and continue
                continue

    def get(
        self,
        key: str,
        block: bool = True,
        timeout: Optional[float] = None,
        start_id: str = "0",
    ) -> Iterator[tuple]:
        """
        Get items from the queue as an iterator.

        Args:
            key: Queue key
            block: Whether to block waiting for new items
            timeout: Block timeout in seconds (None = forever)
            start_id: Start reading from this ID ("0" = beginning, "$" = only new)

        Yields:
            Tuples of (message_id, data)
        """
        last_id = start_id
        block_ms = int(timeout * 1000) if timeout else 0 if block else None

        while not self._stop_event.is_set():
            try:
                if block_ms is not None:
                    entries = self.redis.xread({key: last_id}, block=block_ms, count=100)
                else:
                    entries = self.redis.xread({key: last_id}, count=100)

                if not entries:
                    if not block:
                        break
                    continue

                for stream_key, messages in entries:
                    for msg_id, data in messages:
                        msg_id_str = msg_id.decode() if isinstance(msg_id, bytes) else msg_id
                        yield (msg_id_str, data.get(b"data", data.get("data")))
                        last_id = msg_id

            except (ImportError, ConnectionError):
                # Fatal errors - re-raise immediately
                raise
            except Exception as e:
                # Transient errors - log and retry if blocking
                logger.warning(f"Error reading from queue {key}: {e}")
                if not block:
                    break

    def get_to_queue(
        self,
        key: str,
        dest: Queue,
        block: bool = True,
        timeout: Optional[float] = None,
        start_id: str = "0",
        transform: Optional[Callable] = None,
    ):
        """
        Stream items from Redis Stream to a Python Queue.

        Args:
            key: Queue key
            dest: Destination Python Queue
            block: Whether to block waiting for new items
            timeout: Block timeout in seconds
            start_id: Start reading from this ID
            transform: Optional function to transform data before putting to queue
        """
        for msg_id, data in self.get(key, block=block, timeout=timeout, start_id=start_id):
            if transform:
                data = transform(data)
            dest.put(data)

    def tail(
        self,
        key: str,
        callback: Callable[[bytes], None],
        start_id: str = "$",
    ):
        """
        Tail a queue, calling callback for each new item.

        Args:
            key: Queue key
            callback: Function to call with each item's data
            start_id: Start reading from this ID ("$" = only new items)
        """
        for msg_id, data in self.get(key, block=True, start_id=start_id):
            callback(data)

    def length(self, key: str) -> int:
        """Get the current length of a queue."""
        return self.redis.xlen(key)

    def delete(self, key: str) -> bool:
        """Delete a queue."""
        return self.redis.delete(key) > 0

    def trim(self, key: str, maxlen: int):
        """Trim queue to maximum length."""
        self.redis.xtrim(key, maxlen=maxlen)

    def stop(self):
        """Signal background operations to stop."""
        self._stop_event.set()

    def close(self):
        """Close the Redis connection."""
        self.stop()
        if self._redis is not None:
            self._redis.close()
            self._redis = None


def create_queue_client(namespace: str, **kwargs) -> QueueClient:
    """
    Create a queue client for the given namespace.

    This is a convenience function that handles initialization.

    Args:
        namespace: Kubernetes namespace for the data store
        **kwargs: Additional arguments passed to QueueClient

    Returns:
        Initialized QueueClient
    """
    return QueueClient(namespace=namespace, **kwargs)


# Global queue clients cache (namespace -> QueueClient)
_queue_clients: dict = {}


def _get_queue_client(namespace: str) -> QueueClient:
    """Get or create a QueueClient for the given namespace."""
    if namespace not in _queue_clients:
        _queue_clients[namespace] = QueueClient(namespace=namespace)
    return _queue_clients[namespace]


def _queue_put(
    key: str,
    src: Queue,
    lifespan: str = "cluster",
    namespace: Optional[str] = None,
    maxlen: int = 10000,
    ttl: Optional[int] = None,
    verbose: bool = False,
) -> threading.Thread:
    """
    Stream data from a Python Queue to a Redis Stream.

    This function:
    1. Registers the queue key with the MDS (data_type="queue")
    2. Starts a background thread that pulls from the Python Queue
       and writes to the Redis Stream

    Args:
        key: Queue key (e.g., "logs/launch_123")
        src: Source Python Queue
        lifespan: "cluster" or "resource"
        namespace: Kubernetes namespace
        maxlen: Maximum queue length (MAXLEN for Redis Streams)
        ttl: TTL in seconds for auto-cleanup (e.g., 86400 for 24h)
        verbose: Show debug output

    Returns:
        The background thread (can be joined to wait for completion)

    Example:
        >>> from queue import Queue
        >>> import kubetorch as kt
        >>>
        >>> q = Queue()
        >>> thread = kt.put("logs/my_job", src=q)
        >>>
        >>> # Producer puts items
        >>> q.put("log line 1")
        >>> q.put("log line 2")
        >>> q.put(None)  # Sentinel to stop
        >>>
        >>> thread.join()  # Wait for streaming to complete
    """
    from .metadata_client import MetadataClient

    # Resolve namespace
    if namespace is None:
        import os

        namespace = os.getenv("KT_NAMESPACE") or os.getenv("POD_NAMESPACE", "default")

    # Register with MDS
    metadata_client = MetadataClient(namespace=namespace)
    import socket

    try:
        pod_ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        pod_ip = "127.0.0.1"

    metadata_client.publish_key(
        key=key,
        ip=pod_ip,
        data_type="queue",
        lifespan=lifespan,
        queue_maxlen=maxlen,
        queue_ttl=ttl,
    )

    if verbose:
        logger.info(f"Registered queue key '{key}' with MDS, starting background streaming")

    # Get queue client and start background streaming
    queue_client = _get_queue_client(namespace)

    def stream_worker():
        try:
            queue_client.put_from_queue(key, src, maxlen=maxlen)
        except Exception as e:
            logger.error(f"Error streaming to queue '{key}': {e}")

    thread = threading.Thread(target=stream_worker, daemon=True)
    thread.start()

    return thread


def _queue_get(
    key: str,
    dest: Queue,
    namespace: Optional[str] = None,
    block: bool = True,
    start_from_beginning: bool = True,
    verbose: bool = False,
) -> threading.Thread:
    """
    Stream data from a Redis Stream to a Python Queue.

    This function:
    1. Queries the MDS for queue source info
    2. Starts a background thread that reads from Redis
       and writes to the Python Queue

    Args:
        key: Queue key (e.g., "logs/launch_123")
        dest: Destination Python Queue
        namespace: Kubernetes namespace
        block: Whether to block waiting for new items
        start_from_beginning: If True, read from beginning of stream; if False, only new items
        verbose: Show debug output

    Returns:
        The background thread

    Example:
        >>> from queue import Queue
        >>> import kubetorch as kt
        >>>
        >>> q = Queue()
        >>> thread = kt.get("logs/my_job", dest=q)
        >>>
        >>> # Consumer reads items
        >>> while True:
        ...     item = q.get()
        ...     if item is None:
        ...         break
        ...     print(item.decode())
    """
    from .metadata_client import MetadataClient

    # Resolve namespace
    if namespace is None:
        import os

        namespace = os.getenv("KT_NAMESPACE") or os.getenv("POD_NAMESPACE", "default")

    # Query MDS for source info
    metadata_client = MetadataClient(namespace=namespace)
    source_info = metadata_client.get_source_info(key)

    if not source_info or not source_info.get("found"):
        raise ValueError(f"Queue key '{key}' not found")

    if source_info.get("data_type") != "queue":
        raise ValueError(f"Key '{key}' is not a queue (data_type={source_info.get('data_type')})")

    if verbose:
        logger.info(f"Found queue key '{key}', starting background streaming")

    # Get queue client - let it determine the correct connection method
    # (in-cluster direct connection vs external via WebSocket tunnel)
    # Don't pass redis_host/port from MDS as those are in-cluster addresses
    queue_client = _get_queue_client(namespace)

    start_id = "0" if start_from_beginning else "$"

    def stream_worker():
        try:
            queue_client.get_to_queue(key, dest, block=block, start_id=start_id)
        except Exception as e:
            logger.error(f"Error streaming from queue '{key}': {e}")

    thread = threading.Thread(target=stream_worker, daemon=True)
    thread.start()

    return thread
