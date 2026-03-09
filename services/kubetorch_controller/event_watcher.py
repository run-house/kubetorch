"""
Kubernetes Event Watcher -> Loki

This module watches Kubernetes events cluster-wide and pushes them to namespace-local
Loki instances (running in the kubetorch-data-store service).

Architecture:
- Uses the K8s Watch API to stream events in real-time (single long-lived connection)
- Dynamically discovers which namespaces have a kubetorch-data-store service
- Batches events per namespace and pushes to each namespace's Loki endpoint
- Events are stored with labels for efficient querying (reason, kind, name, etc.)

Configuration (via environment variables):
- EVENT_WATCH_ENABLED: Enable/disable event watching (default: true)
- EVENT_BATCH_SIZE: Number of events to batch before pushing (default: 10)
- EVENT_FLUSH_INTERVAL: Max seconds between flushes (default: 5.0)

Usage:
    The event watcher is started automatically via the background_tasks lifespan handler.
    See background_tasks.py for the lifespan context manager.

    The event watcher creates its own isolated K8s client to avoid connection
    pool conflicts with the main API handlers.

Loki Query Examples:
    # All events for a specific pod
    {job="kubetorch-events", name="my-pod-abc123"}

    # All events for a service (matches pods, deployments, replicasets with that prefix)
    {job="kubetorch-events", name=~"my-service.*"}

    # All Warning events in namespace
    {job="kubetorch-events", namespace="default", event_type="Warning"}

    # OOMKilled events
    {job="kubetorch-events", reason="OOMKilled"}

    # All Pod events
    {job="kubetorch-events", kind="Pod"}
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Set

import httpx
from fastapi.concurrency import run_in_threadpool
from kubernetes import client, config, watch

logger = logging.getLogger(__name__)

# Silence httpx INFO logs (too noisy for every Loki push)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Configuration
EVENT_BATCH_SIZE = int(os.getenv("EVENT_BATCH_SIZE", "10"))
EVENT_FLUSH_INTERVAL = float(
    os.getenv("EVENT_FLUSH_INTERVAL", "1.0")
)  # 1s for fast delivery of OOMs etc


class EventWatcher:
    """Watches K8s events and pushes them to namespace-local Loki instances."""

    def __init__(self):
        # Create isolated K8s client to avoid connection pool conflicts with main API handlers.
        # This is critical - sharing the global client causes "Handshake status 200 OK" errors
        # when the Watch API's long-lived connection interferes with regular API calls.
        self._api_client, self.core_v1 = self._create_isolated_k8s_client()
        self._namespaces_with_datastore: Set[str] = set()
        self._datastore_check_interval = 60  # seconds
        self._last_datastore_check = 0.0

    def _create_isolated_k8s_client(self):
        """Create an isolated K8s client for this watcher.

        Uses a separate Configuration and ApiClient to avoid connection pool
        conflicts with the main request handlers.

        Returns:
            Tuple of (ApiClient, CoreV1Api) - caller must close ApiClient when done.
        """
        try:
            watcher_config = client.Configuration()
            try:
                config.load_incluster_config(client_configuration=watcher_config)
            except config.ConfigException:
                config.load_kube_config(client_configuration=watcher_config)

            api_client = client.ApiClient(configuration=watcher_config)
            return api_client, client.CoreV1Api(api_client=api_client)
        except Exception as e:
            logger.error(
                f"[worker-{os.getpid()}] Failed to create isolated K8s client for event watcher: {e}"
            )
            raise

    def close(self):
        """Close the K8s API client to release connection pools."""
        if self._api_client:
            try:
                self._api_client.close()
            except Exception:
                pass

    async def check_datastore_namespaces(self):
        """Check which namespaces have a kubetorch-data-store service running."""
        now = time.time()
        if now - self._last_datastore_check < self._datastore_check_interval:
            return

        self._last_datastore_check = now
        try:
            result = await run_in_threadpool(
                self.core_v1.list_service_for_all_namespaces,
                field_selector="metadata.name=kubetorch-data-store",
            )
            new_namespaces = {svc.metadata.namespace for svc in result.items}
            if new_namespaces != self._namespaces_with_datastore:
                logger.info(f"Data store namespaces updated: {new_namespaces}")
            self._namespaces_with_datastore = new_namespaces
        except Exception as e:
            logger.warning(f"Failed to check data store namespaces: {e}")

    def format_event_for_loki(self, event_obj) -> Dict:
        """Format a K8s event object for Loki ingestion."""
        involved_obj = event_obj.involved_object

        return {
            "event_type": event_obj.type or "Normal",
            "reason": event_obj.reason or "",
            "message": event_obj.message or "",
            "count": event_obj.count or 1,
            "first_timestamp": (
                event_obj.first_timestamp.isoformat()
                if event_obj.first_timestamp
                else ""
            ),
            "last_timestamp": (
                event_obj.last_timestamp.isoformat() if event_obj.last_timestamp else ""
            ),
            "involved_object": {
                "kind": involved_obj.kind if involved_obj else "",
                "name": involved_obj.name if involved_obj else "",
                "namespace": involved_obj.namespace if involved_obj else "",
            },
            "source": {
                "component": event_obj.source.component if event_obj.source else "",
                "host": event_obj.source.host if event_obj.source else "",
            },
        }

    async def push_events_to_loki(self, namespace: str, events: List[Dict]):
        """Push a batch of events to the namespace's Loki instance."""
        if not events:
            return

        streams = []
        for event in events:
            # Convert timestamp to nanoseconds for Loki
            ts = event.get("last_timestamp") or time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            )
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                ts_ns = str(int(dt.timestamp() * 1e9))
            except Exception:
                ts_ns = str(int(time.time() * 1e9))

            involved = event.get("involved_object", {})
            involved_name = involved.get("name", "")
            involved_kind = involved.get("kind", "")

            # Use exact resource labels - client-side queries use regex patterns
            # e.g., {job="kubetorch-events", name=~"my-service.*"} matches all related events
            stream = {
                "stream": {
                    "job": "kubetorch-events",
                    "namespace": namespace,
                    "name": involved_name,  # Exact resource name (pod, deployment, etc.)
                    "kind": involved_kind,  # Pod, Deployment, ReplicaSet, etc.
                    "event_type": event.get("event_type", "Normal"),
                    "reason": event.get("reason", ""),
                },
                "values": [[ts_ns, json.dumps(event)]],
            }
            streams.append(stream)

        url = f"http://kubetorch-data-store.{namespace}:3100/loki/api/v1/push"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json={"streams": streams})
                if resp.status_code == 429:
                    # Rate limited - back off to avoid hammering Loki
                    logger.warning(f"Loki rate limited for {namespace}, backing off 5s")
                    await asyncio.sleep(5)
                if resp.status_code not in (200, 204):
                    logger.warning(
                        f"Loki push to {namespace} returned {resp.status_code}: {resp.text[:200]}"
                    )
        except httpx.ConnectError:
            # Data store might not be ready yet
            pass
        except Exception as e:
            logger.debug(f"Failed to push events to Loki in {namespace}: {e}")

    def _get_current_resource_version(self) -> str:
        """Get the current resource version to start watching from 'now'.

        This avoids replaying all historical events on startup, which would
        load thousands of K8s event objects into memory.
        """
        try:
            # Get just the metadata with limit=1 to minimize data transfer
            events_list = self.core_v1.list_event_for_all_namespaces(limit=1)
            return events_list.metadata.resource_version
        except Exception as e:
            logger.warning(
                f"Failed to get resource version, starting from beginning: {e}"
            )
            return ""

    async def run(self):
        """Main loop: watch K8s events and push to Loki."""
        logger.info("Starting K8s event watcher")

        event_batches: Dict[str, List[Dict]] = {}
        last_flush = time.time()

        while True:
            try:
                await self.check_datastore_namespaces()

                # Get current resource version to start from "now" instead of replaying history.
                resource_version = await run_in_threadpool(
                    self._get_current_resource_version
                )
                logger.info(
                    f"Starting event watch from resource_version={resource_version}"
                )

                # Run the blocking K8s watch in a thread to avoid blocking the event loop
                event_queue: asyncio.Queue = asyncio.Queue()
                stop_event = asyncio.Event()

                # Capture the event loop reference before entering the thread
                loop = asyncio.get_running_loop()

                async def watch_events_in_thread():
                    """Run the blocking K8s watch stream in a thread pool."""

                    def _blocking_watch():
                        w = watch.Watch()
                        try:
                            for event in w.stream(
                                self.core_v1.list_event_for_all_namespaces,
                                resource_version=resource_version,
                                timeout_seconds=300,  # Reconnect every 5 minutes
                            ):
                                if stop_event.is_set():
                                    w.stop()
                                    break
                                # Put event in queue from thread (thread-safe)
                                asyncio.run_coroutine_threadsafe(
                                    event_queue.put(event),
                                    loop,  # Use captured loop reference
                                )
                        except Exception as e:
                            logger.warning(f"Event watch stream error: {e}")
                        finally:
                            # Signal end of stream
                            asyncio.run_coroutine_threadsafe(
                                event_queue.put(None),
                                loop,  # Use captured loop reference
                            )

                    await run_in_threadpool(_blocking_watch)

                # Start the watch in a background task
                watch_task = asyncio.create_task(watch_events_in_thread())

                # Process events from queue without blocking
                while True:
                    try:
                        event = await asyncio.wait_for(
                            event_queue.get(), timeout=EVENT_FLUSH_INTERVAL
                        )
                    except asyncio.TimeoutError:
                        # Flush on timeout
                        now = time.time()
                        for ns, batch in list(event_batches.items()):
                            if batch:
                                await self.push_events_to_loki(ns, batch)
                                event_batches[ns] = []
                        last_flush = now
                        continue

                    if event is None:
                        # Stream ended, break to reconnect
                        break

                    event_type = event["type"]
                    event_obj = event["object"]

                    if event_type not in ("ADDED", "MODIFIED"):
                        continue

                    involved_obj = event_obj.involved_object
                    if not involved_obj or not involved_obj.namespace:
                        continue

                    namespace = involved_obj.namespace

                    # Skip if no data store in this namespace
                    if namespace not in self._namespaces_with_datastore:
                        await self.check_datastore_namespaces()
                        if namespace not in self._namespaces_with_datastore:
                            continue

                    # Batch the event
                    formatted = self.format_event_for_loki(event_obj)
                    if namespace not in event_batches:
                        event_batches[namespace] = []
                    event_batches[namespace].append(formatted)

                    # Flush if needed
                    now = time.time()
                    should_flush = (
                        len(event_batches.get(namespace, [])) >= EVENT_BATCH_SIZE
                        or (now - last_flush) >= EVENT_FLUSH_INTERVAL
                    )

                    if should_flush:
                        for ns, batch in list(event_batches.items()):
                            if batch:
                                await self.push_events_to_loki(ns, batch)
                                event_batches[ns] = []
                        last_flush = now

                # Clean up watch task
                stop_event.set()
                watch_task.cancel()
                try:
                    await watch_task
                except asyncio.CancelledError:
                    pass

            except asyncio.CancelledError:
                logger.info("K8s event watcher cancelled")
                break
            except Exception as e:
                logger.error(f"K8s event watcher error: {e}")
                await asyncio.sleep(5)
