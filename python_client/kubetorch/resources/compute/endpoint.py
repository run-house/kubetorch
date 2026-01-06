"""Endpoint configuration for custom routing in kubetorch."""

import re
from typing import Dict, Optional

from kubetorch.provisioning.constants import DEFAULT_NGINX_PORT


class Endpoint:
    """Configure how kubetorch routes calls to a compute pool.

    An Endpoint controls the routing layer - how HTTP requests reach pods in the pool.
    This is separate from pool membership (which pods belong to the pool).

    Args:
        url (str, optional): User-provided URL to route calls to. When set, no K8s Service is created.
            Example: "my-service.default.svc.cluster.local:8080". (Default: None)
        selector (Optional[Dict[str, str]]): Custom selector for the K8s Service that kubetorch creates.
            Use this to route to a subset of pool pods.
            Example: {"app": "ray", "ray.io/node-type": "head"}. (Default: None)

    Examples:

        .. code-block:: python

            import kubetorch as kt

            # No specified endpoint - KT creates Service using pool selector
            compute = kt.Compute(cpus=4)

            # URL - Use your own Service/Ingress
            compute = kt.Compute.from_manifest(
                manifest=my_manifest,
                selector={"app": "workers"},
                endpoint=kt.Endpoint(url="my-lb.default.svc:8080")
            )

            # Custom selector - Route to subset of pool
            compute = kt.Compute.from_manifest(
                manifest=ray_manifest,
                selector={"app": "ray"},  # Pool: all ray pods
                endpoint=kt.Endpoint(selector={"app": "ray", "role": "head"})  # Route: head only
            )
    """

    def __init__(
        self,
        url: Optional[str] = None,
        selector: Optional[Dict[str, str]] = None,
    ):
        if url and selector:
            raise ValueError(
                "Cannot specify both 'url' and 'selector'. "
                "Use 'url' for an existing endpoint, or 'selector' to create a Service with custom routing."
            )

        self.url = url
        self.selector = selector

    def to_service_config(self) -> Optional[Dict]:
        """Convert endpoint to service config for controller's register_pool API."""
        if self.url:
            return {"type": "url", "url": self.url}
        elif self.selector:
            return {"type": "selector", "selector": self.selector}
        else:
            return None

    @property
    def mode(self) -> str:
        """Return the endpoint mode for debugging/logging."""
        if self.url:
            return "url"
        elif self.selector:
            return "selector"
        else:
            return "auto"

    def __repr__(self) -> str:
        if self.url:
            return f"Endpoint(url={self.url!r})"
        elif self.selector:
            return f"Endpoint(selector={self.selector!r})"
        else:
            return "Endpoint()"

    def get_proxied_url(self, proxy_port: int) -> Optional[str]:
        """Get URL for accessing this endpoint from outside the cluster.

        For cluster-internal URLs, returns a localhost URL routed through the nginx proxy.
        For external URLs, returns the URL unchanged.

        Args:
            proxy_port (int): The local port where the nginx proxy is listening.

        Returns:
            The URL to use from outside the cluster, or None if no URL is configured.
        """
        if not self.url:
            return None

        # Pattern to detect cluster-internal Kubernetes URLs
        # Matches: http(s)://service-name.namespace.svc[.cluster.local][:port][/path]
        cluster_internal_pattern = re.compile(r"https?://([^.]+)\.([^.]+)\.svc(?:\.cluster\.local)?(?::(\d+))?(/.*)?$")
        match = cluster_internal_pattern.match(self.url)
        if match:
            name, namespace, port, path = match.groups()
            port = port or DEFAULT_NGINX_PORT
            path = path or ""
            return f"http://localhost:{proxy_port}/{namespace}/{name}:{port}{path}"
        return self.url
