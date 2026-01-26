"""
Monarch Proxy Classes - Client-side proxies for remote Monarch objects.

These classes provide a Monarch-compatible API but route operations through
the MonarchGateway over WebSocket. They allow users to write code as if they
were using Monarch directly, but from outside the Kubernetes cluster.
"""

import pickle
from math import prod
from typing import Dict, Generic, Optional, TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from kubetorch.monarch.client import GatewayConnection

T = TypeVar("T")
R = TypeVar("R")


class FutureProxy(Generic[R]):
    """
    Proxy for a remote Monarch Future.

    Provides the same API as Monarch's Future but routes get() calls
    through the gateway.
    """

    def __init__(self, future_id: str, gateway: "GatewayConnection"):
        self._future_id = future_id
        self._gateway = gateway
        self._result = None
        self._resolved = False

    def get(self, timeout: Optional[float] = None) -> R:
        """
        Get the result of the future, blocking until ready.

        Args:
            timeout: Timeout in seconds (None for no timeout)

        Returns:
            The result value
        """
        if self._resolved:
            return self._result

        result_bytes = self._gateway.call(
            "get_future_result",
            future_id=self._future_id,
            timeout=timeout,
        )
        raw_result = pickle.loads(result_bytes)

        # Convert gateway's serialized ValueMesh back to ValueMeshProxy
        if isinstance(raw_result, dict) and raw_result.get("_type") == "ValueMesh":
            self._result = ValueMeshProxy(raw_result["data"], raw_result["shape"])
        else:
            self._result = raw_result

        self._resolved = True
        return self._result

    def result(self, timeout: Optional[float] = None) -> R:
        """Alias for get() for compatibility."""
        return self.get(timeout=timeout)


class ValueMeshProxy(Generic[R]):
    """
    Proxy for a remote Monarch ValueMesh.

    Holds results from an endpoint.call() indexed by mesh coordinates.
    """

    def __init__(self, data: Dict[tuple, R], shape: Dict[str, int]):
        self._data = data
        self._shape = shape

    def item(self, **coords) -> R:
        """Get a single value by dimension coordinates."""
        key = tuple(coords.get(dim, 0) for dim in self._shape.keys())
        return self._data[key]

    def items(self):
        """Iterate over (point, value) pairs."""
        return self._data.items()

    def values(self):
        """Iterate over values."""
        return self._data.values()

    def __len__(self) -> int:
        return len(self._data)


class EndpointProxy:
    """
    Proxy for a remote Monarch endpoint.

    Provides call(), call_one(), broadcast(), and stream() methods
    that mirror Monarch's endpoint API.
    """

    def __init__(
        self,
        actor_mesh_id: str,
        endpoint_name: str,
        gateway: "GatewayConnection",
    ):
        self._actor_mesh_id = actor_mesh_id
        self._endpoint_name = endpoint_name
        self._gateway = gateway

    def call(self, *args, **kwargs) -> FutureProxy:
        """
        Call the endpoint on all actors, returning a future of ValueMesh.

        Returns:
            FutureProxy that resolves to ValueMesh of results
        """
        result = self._gateway.call(
            "call_endpoint",
            actor_mesh_id=self._actor_mesh_id,
            endpoint_name=self._endpoint_name,
            args_bytes=pickle.dumps(args),
            kwargs_bytes=pickle.dumps(kwargs),
            selection="all",
        )
        return FutureProxy(result["future_id"], self._gateway)

    def call_one(self, *args, **kwargs) -> FutureProxy:
        """
        Call the endpoint on a single actor.

        Returns:
            FutureProxy that resolves to the single result
        """
        result = self._gateway.call(
            "call_endpoint",
            actor_mesh_id=self._actor_mesh_id,
            endpoint_name=self._endpoint_name,
            args_bytes=pickle.dumps(args),
            kwargs_bytes=pickle.dumps(kwargs),
            selection="one",
        )
        return FutureProxy(result["future_id"], self._gateway)

    def broadcast(self, *args, **kwargs) -> None:
        """
        Broadcast to all actors (fire-and-forget, no return value).
        """
        self._gateway.call(
            "broadcast_endpoint",
            actor_mesh_id=self._actor_mesh_id,
            endpoint_name=self._endpoint_name,
            args_bytes=pickle.dumps(args),
            kwargs_bytes=pickle.dumps(kwargs),
        )


class ActorMeshProxy(Generic[T]):
    """
    Proxy for a remote Monarch ActorMesh.

    Provides access to actor endpoints via attribute access, and mesh
    operations like slice, split, flatten.
    """

    def __init__(
        self,
        actor_mesh_id: str,
        shape: Dict[str, int],
        gateway: "GatewayConnection",
    ):
        self._actor_mesh_id = actor_mesh_id
        self._shape = shape
        self._gateway = gateway

    def __getattr__(self, name: str) -> EndpointProxy:
        """Access an endpoint by name."""
        # Avoid infinite recursion for internal attributes
        if name.startswith("_"):
            raise AttributeError(name)
        return EndpointProxy(self._actor_mesh_id, name, self._gateway)

    @property
    def sizes(self) -> Dict[str, int]:
        """Get the shape as a dict of dimension sizes."""
        return self._shape.copy()

    def size(self, dim: Optional[str] = None) -> int:
        """Get the size of a dimension or total elements."""
        if dim is not None:
            return self._shape[dim]
        return prod(self._shape.values()) if self._shape else 1

    def slice(self, **kwargs) -> "ActorMeshProxy[T]":
        """
        Create a sliced view of the actor mesh.

        This is a local operation - no network call needed.
        """
        new_shape = {}
        for dim, size in self._shape.items():
            if dim in kwargs:
                selector = kwargs[dim]
                if isinstance(selector, int):
                    # Single index removes the dimension
                    continue
                elif isinstance(selector, slice):
                    start = selector.start or 0
                    stop = selector.stop or size
                    new_shape[dim] = stop - start
            else:
                new_shape[dim] = size

        return ActorMeshProxy(self._actor_mesh_id, new_shape, self._gateway)

    def stop(self) -> FutureProxy[None]:
        """Stop all actors in this mesh."""
        self._gateway.call("stop_actor_mesh", actor_mesh_id=self._actor_mesh_id)
        # Return a completed future for API compatibility
        future = FutureProxy("", self._gateway)
        future._resolved = True
        future._result = None
        return future


class ProcMeshProxy:
    """
    Proxy for a remote Monarch ProcMesh.

    Provides spawn() for creating ActorMeshes and mesh operations.
    """

    def __init__(
        self,
        proc_mesh_id: str,
        shape: Dict[str, int],
        gateway: "GatewayConnection",
        host_mesh: "HostMeshProxy",
    ):
        self._proc_mesh_id = proc_mesh_id
        self._shape = shape
        self._gateway = gateway
        self._host_mesh = host_mesh

    def spawn(
        self,
        name: str,
        actor_class: type,
        *args,
        **kwargs,
    ) -> ActorMeshProxy:
        """
        Spawn an ActorMesh on this ProcMesh.

        Args:
            name: Name for the actor mesh
            actor_class: The Actor class to instantiate
            *args: Positional arguments for actor __init__
            **kwargs: Keyword arguments for actor __init__

        Returns:
            ActorMeshProxy for the spawned actors
        """
        # Extract import pointers for the actor class so the gateway can import it
        import os

        from kubetorch.resources.callables.utils import extract_pointers, locate_working_dir

        root_path, module_name, class_name = extract_pointers(actor_class)

        # Compute the relative path from git root to the actor's directory
        # This is needed so the gateway knows what to add to sys.path
        git_root, _, _ = locate_working_dir(root_path)
        module_path = os.path.relpath(root_path, git_root)
        if module_path == ".":
            module_path = ""  # Actor is at git root, no extra path needed

        result = self._gateway.call(
            "spawn_actors",
            proc_mesh_id=self._proc_mesh_id,
            name=name,
            module_name=module_name,
            class_name=class_name,
            module_path=module_path,  # Relative path to add to sys.path
            args=list(args),
            kwargs=kwargs,
        )
        return ActorMeshProxy(
            result["actor_mesh_id"],
            result["shape"],
            self._gateway,
        )

    @property
    def sizes(self) -> Dict[str, int]:
        """Get the shape as a dict of dimension sizes."""
        return self._shape.copy()

    @property
    def host_mesh(self) -> "HostMeshProxy":
        """Get the parent HostMesh."""
        return self._host_mesh

    def size(self, dim: Optional[str] = None) -> int:
        """Get the size of a dimension or total elements."""
        if dim is not None:
            return self._shape[dim]
        return prod(self._shape.values()) if self._shape else 1

    def slice(self, **kwargs) -> "ProcMeshProxy":
        """Create a sliced view of the proc mesh."""
        new_shape = {}
        for dim, size in self._shape.items():
            if dim in kwargs:
                selector = kwargs[dim]
                if isinstance(selector, int):
                    continue
                elif isinstance(selector, slice):
                    start = selector.start or 0
                    stop = selector.stop or size
                    new_shape[dim] = stop - start
            else:
                new_shape[dim] = size

        return ProcMeshProxy(self._proc_mesh_id, new_shape, self._gateway, self._host_mesh)

    def stop(self) -> FutureProxy[None]:
        """Stop all processes in this mesh."""
        self._gateway.call("stop_proc_mesh", proc_mesh_id=self._proc_mesh_id)
        future = FutureProxy("", self._gateway)
        future._resolved = True
        future._result = None
        return future


class HostMeshProxy:
    """
    Proxy for a remote Monarch HostMesh.

    Provides spawn_procs() for creating ProcMeshes and mesh operations.
    """

    def __init__(
        self,
        host_mesh_id: str,
        shape: Dict[str, int],
        gateway: "GatewayConnection",
    ):
        self._host_mesh_id = host_mesh_id
        self._shape = shape
        self._gateway = gateway

    def spawn_procs(
        self,
        per_host: Dict[str, int],
        name: str = "procs",
    ) -> ProcMeshProxy:
        """
        Spawn a ProcMesh on this HostMesh.

        Args:
            per_host: Dict specifying processes per host, e.g. {"gpus": 8}
            name: Name for the proc mesh

        Returns:
            ProcMeshProxy for the spawned processes
        """
        result = self._gateway.call(
            "spawn_procs",
            per_host=per_host,
            name=name,
        )
        return ProcMeshProxy(
            result["proc_mesh_id"],
            result["shape"],
            self._gateway,
            self,
        )

    @property
    def sizes(self) -> Dict[str, int]:
        """Get the shape as a dict of dimension sizes."""
        return self._shape.copy()

    def size(self, dim: Optional[str] = None) -> int:
        """Get the size of a dimension or total elements."""
        if dim is not None:
            return self._shape[dim]
        return prod(self._shape.values()) if self._shape else 1

    def slice(self, **kwargs) -> "HostMeshProxy":
        """Create a sliced view of the host mesh."""
        new_shape = {}
        for dim, size in self._shape.items():
            if dim in kwargs:
                selector = kwargs[dim]
                if isinstance(selector, int):
                    continue
                elif isinstance(selector, slice):
                    start = selector.start or 0
                    stop = selector.stop or size
                    new_shape[dim] = stop - start
            else:
                new_shape[dim] = size

        return HostMeshProxy(self._host_mesh_id, new_shape, self._gateway)

    def shutdown(self) -> FutureProxy[None]:
        """Shutdown all hosts in this mesh."""
        self._gateway.call("shutdown")
        future = FutureProxy("", self._gateway)
        future._resolved = True
        future._result = None
        return future


class JobState:
    """
    Holds the state returned by a Job.

    Provides attribute access to named HostMeshes.
    """

    def __init__(self, meshes: Dict[str, HostMeshProxy]):
        self._meshes = meshes

    def __getattr__(self, name: str) -> HostMeshProxy:
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._meshes:
            raise AttributeError(f"No mesh named '{name}' in job state")
        return self._meshes[name]

    def __repr__(self) -> str:
        mesh_names = list(self._meshes.keys())
        return f"JobState(meshes={mesh_names})"
