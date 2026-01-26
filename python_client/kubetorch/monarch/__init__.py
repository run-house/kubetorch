"""
Kubetorch Monarch Integration

This module provides Monarch integration for Kubetorch, allowing users to
create and interact with Monarch meshes from outside the Kubernetes cluster.

Usage:
    from kubetorch.monarch import KubernetesJob

    # Create a job with fresh compute allocation
    job = KubernetesJob(compute=kt.Compute(cpu="4", gpu=8, replicas=4))
    state = job.state()
    host_mesh = state.workers

    # Spawn processes and actors
    proc_mesh = host_mesh.spawn_procs(per_host={"gpus": 8})
    actors = proc_mesh.spawn("trainers", MyActor)

    # Call actor methods
    result = actors.my_method.call(arg).get()

For pre-allocated compute:
    job = KubernetesJob(selector={"app": "my-monarch-workers"})
"""

from kubetorch.monarch.gateway import MonarchGateway
from kubetorch.monarch.job import GatewayConnection, KubernetesJob
from kubetorch.monarch.proxy import (
    ActorMeshProxy,
    EndpointProxy,
    FutureProxy,
    HostMeshProxy,
    JobState,
    ProcMeshProxy,
    ValueMeshProxy,
)

__all__ = [
    # Main job class
    "KubernetesJob",
    # Gateway (server-side)
    "MonarchGateway",
    "GatewayConnection",
    # Proxy classes (client-side)
    "ActorMeshProxy",
    "EndpointProxy",
    "FutureProxy",
    "HostMeshProxy",
    "JobState",
    "ProcMeshProxy",
    "ValueMeshProxy",
]
