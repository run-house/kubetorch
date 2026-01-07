"""
Centralized HTTP clients for the serving module.

Provides global httpx clients with proper connection pooling.
All HTTP calls in serving should use these clients instead of creating their own.
"""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Silence httpx/httpcore INFO logs to prevent noise
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Global sync client for blocking HTTP calls
_sync_client: Optional[httpx.Client] = None

# Global async client for async HTTP calls
_async_client: Optional[httpx.AsyncClient] = None


def get_sync_client() -> httpx.Client:
    """Get the global sync httpx client with connection pooling.

    Use this for all synchronous HTTP calls in the serving module.
    The client is lazily created on first use.
    """
    global _sync_client
    if _sync_client is None:
        _sync_client = httpx.Client(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=30.0,
            ),
        )
        logger.debug("Created global sync httpx client")
    return _sync_client


def get_async_client() -> httpx.AsyncClient:
    """Get the global async httpx client with connection pooling.

    Use this for all async HTTP calls in the serving module.
    The client is lazily created on first use.

    Note: AsyncClient is bound to the event loop it was created on.
    If you're in a different event loop (e.g., subprocess), create a local client.
    """
    global _async_client
    if _async_client is None:
        _async_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=30.0,
            ),
        )
        logger.debug("Created global async httpx client")
    return _async_client


def close_clients():
    """Close all global HTTP clients. Call during shutdown."""
    global _sync_client, _async_client

    if _sync_client is not None:
        try:
            _sync_client.close()
        except Exception:
            pass
        _sync_client = None

    if _async_client is not None:
        # Note: For async client, use close_clients_async() in async context
        # This sync version just clears the reference
        _async_client = None


async def close_clients_async():
    """Close all global HTTP clients (async version). Call during shutdown."""
    global _sync_client, _async_client

    if _sync_client is not None:
        try:
            _sync_client.close()
        except Exception:
            pass
        _sync_client = None

    if _async_client is not None:
        try:
            await _async_client.aclose()
        except Exception:
            pass
        _async_client = None
