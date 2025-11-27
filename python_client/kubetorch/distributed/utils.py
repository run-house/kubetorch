"""
Distributed utilities for kubetorch services.

This module provides utilities for distributed operations within kubetorch services,
such as discovering peer pod IPs via DNS.
"""

import os
import socket
import time
from typing import List, Optional

from kubetorch.logger import get_logger
from kubetorch.servers.http.utils import is_running_in_kubernetes

logger = get_logger(__name__)


def pod_ips(
    quorum_workers: Optional[int] = None,
    quorum_timeout: float = 0,
) -> List[str]:
    """
    Get pod IPs for the current kubetorch service via DNS lookup.

    This function discovers all pod IPs in the current service's headless service
    using DNS resolution. It can optionally wait for a quorum of workers to appear.

    Args:
        quorum_workers: Expected number of workers. If specified and quorum_timeout > 0,
            will wait for this many workers to appear in DNS.
        quorum_timeout: Maximum time in seconds to wait for quorum_workers.
            If 0 (default), returns immediately after first DNS query.

    Returns:
        List of pod IP addresses (sorted)

    Raises:
        RuntimeError: If required environment variables are not set

    Example:

    .. code-block:: python

        import kubetorch as kt

        # Get all pod IPs immediately
        ips = kt.distributed.pod_ips()
        print(f"Found {len(ips)} pods: {ips}")

        # Wait for 4 workers with 60 second timeout
        ips = kt.distributed.pod_ips(quorum_workers=4, quorum_timeout=60)
    """
    # For testing outside of Kubernetes
    if not is_running_in_kubernetes():
        local_ips = os.environ.get("LOCAL_IPS")
        if local_ips:
            return local_ips.split(",")
        return []

    # Use DNS-based service discovery
    # Check if pre-computed DNS name is available (should point to headless service for distributed)
    service_dns = os.environ.get("KT_SERVICE_DNS")

    if not service_dns:
        # Fall back to computing DNS name from service and namespace
        service_name = os.environ.get("KT_SERVICE_NAME")
        namespace = os.environ.get("POD_NAMESPACE")

        if not service_name:
            raise RuntimeError("KT_SERVICE_NAME environment variable not found")
        if not namespace:
            raise RuntimeError("POD_NAMESPACE environment variable not found")

        # Kubernetes headless service DNS name for distributed pod discovery
        # Format: <service-name>-headless.<namespace>.svc.cluster.local
        service_dns = f"{service_name}-headless.{namespace}.svc.cluster.local"

    start_time = time.time()
    max_wait = quorum_timeout if quorum_timeout else 0
    expected_workers = quorum_workers

    result_ips: List[str] = []
    last_count = 0

    while True:
        try:
            # DNS lookup returns all pod IPs for the headless service
            # getaddrinfo returns list of (family, type, proto, canonname, sockaddr)
            addr_info = socket.getaddrinfo(service_dns, None, socket.AF_INET)

            # Extract unique IP addresses from the results
            result_ips = sorted(list(set([addr[4][0] for addr in addr_info])))

            if not result_ips:
                logger.debug(f"No pod IPs found for service {service_dns}")
            else:
                logger.debug(f"Found {len(result_ips)} pod IPs via DNS for {service_dns}: {result_ips}")

        except socket.gaierror as e:
            logger.debug(f"DNS lookup failed for {service_dns}: {e}")
            result_ips = []

        # Check if we should wait for more workers
        elapsed = time.time() - start_time

        # If we have the expected count, we're done
        if expected_workers and len(result_ips) >= expected_workers:
            logger.info(f"Found {len(result_ips)}/{expected_workers} workers after {elapsed:.1f}s")
            return result_ips

        # If we don't have expected count or timeout is reached, decide what to do
        if elapsed >= max_wait:
            if expected_workers:
                logger.warning(f"Only found {len(result_ips)}/{expected_workers} workers after {elapsed:.1f}s timeout")
            else:
                logger.debug(f"Found {len(result_ips)} workers")
            return result_ips

        # Log progress if count changed
        if len(result_ips) != last_count:
            if expected_workers:
                logger.info(f"{len(result_ips)}/{expected_workers} workers found, waiting for quorum...")
            else:
                logger.debug(f"{len(result_ips)} workers found, no quorum set")
            last_count = len(result_ips)

        # Wait before retrying
        time.sleep(2)
