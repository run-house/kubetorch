"""
Background Tasks Manager for Kubetorch Controller

This module manages long-running background tasks that run alongside the FastAPI server.
Tasks are started via FastAPI's lifespan context manager and gracefully shut down on exit.

Current background tasks:
- EventWatcher: Streams K8s events to namespace-local Loki instances
- TTLController: Periodically checks for and deletes inactive services

Configuration (via environment variables):
- EVENT_WATCH_ENABLED: Enable/disable K8s event watching (default: true)
- TTL_CONTROLLER_ENABLED: Enable/disable TTL controller (default: false)
- TTL_INTERVAL_SECONDS: Interval between TTL reconciliation cycles (default: 300)

Usage:
    from background_tasks import create_lifespan
    app = FastAPI(lifespan=create_lifespan())
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Callable, List

from event_watcher import EventWatcher

from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Configuration
EVENT_WATCH_ENABLED = os.getenv("EVENT_WATCH_ENABLED", "true").lower() == "true"
TTL_CONTROLLER_ENABLED = os.getenv("TTL_CONTROLLER_ENABLED", "false").lower() == "true"
TTL_INTERVAL_SECONDS = int(os.getenv("TTL_INTERVAL_SECONDS", "300"))


def create_lifespan() -> Callable:
    """Create a FastAPI lifespan context manager that runs background tasks.

    Manages the lifecycle of:
    - EventWatcher: Streams K8s events to Loki (if EVENT_WATCH_ENABLED)
    - TTLController: Deletes inactive services (if TTL_CONTROLLER_ENABLED)

    Each task creates its own isolated K8s client to avoid connection pool
    conflicts with the main API handlers.

    Returns:
        An async context manager for FastAPI's lifespan parameter

    Example:
        app = FastAPI(lifespan=create_lifespan())
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        tasks: List[asyncio.Task] = []
        cleanups: List[Callable] = []

        # Start Event Watcher
        if EVENT_WATCH_ENABLED:
            watcher = EventWatcher()
            task = asyncio.create_task(watcher.run())
            tasks.append(task)
            cleanups.append(watcher.close)
            logger.info("K8s event watcher started")

        # Start TTL Controller
        if TTL_CONTROLLER_ENABLED:
            try:
                from ttl_controller import TTLController

                ttl_controller = TTLController()
                task = asyncio.create_task(ttl_controller.run(TTL_INTERVAL_SECONDS))
                tasks.append(task)
                cleanups.append(ttl_controller.close)
                logger.info(
                    f"TTL controller started (interval: {TTL_INTERVAL_SECONDS}s)"
                )
            except Exception as e:
                logger.error(f"Failed to start TTL controller: {e}")

        yield

        # Cancel all background tasks
        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Run cleanup functions (close K8s clients, etc.)
        for cleanup in cleanups:
            try:
                cleanup()
            except Exception as e:
                logger.warning(f"Cleanup error: {e}")

        logger.info("Background tasks stopped")

    return lifespan
