"""
Distributed backends for different frameworks.

This module contains framework-specific implementations for distributed processing,
including PyTorch, JAX, TensorFlow, Ray, and Monarch.

Backend modules are lazily imported to avoid pulling in heavy dependencies
(FastAPI, Starlette, framework-specific libs) until actually needed.
"""

__all__ = [
    "PyTorchProcess",
    "JaxProcess",
    "TensorflowProcess",
    "RayProcess",
    "RayDistributed",
    "MonarchProcess",
    "MonarchDistributed",
]

# Cache for lazy-loaded backend classes
_backend_cache = {}


def __getattr__(name):
    """Lazy load backend classes to avoid importing heavy dependencies."""
    if name in __all__:
        if name not in _backend_cache:
            if name == "PyTorchProcess":
                from .pytorch import PyTorchProcess

                _backend_cache[name] = PyTorchProcess
            elif name == "JaxProcess":
                from .jax import JaxProcess

                _backend_cache[name] = JaxProcess
            elif name == "TensorflowProcess":
                from .tensorflow import TensorflowProcess

                _backend_cache[name] = TensorflowProcess
            elif name == "RayProcess":
                from .ray import RayProcess

                _backend_cache[name] = RayProcess
            elif name == "RayDistributed":
                from .ray import RayDistributed

                _backend_cache[name] = RayDistributed
            elif name == "MonarchProcess":
                from .monarch import MonarchProcess

                _backend_cache[name] = MonarchProcess
            elif name == "MonarchDistributed":
                from .monarch import MonarchDistributed

                _backend_cache[name] = MonarchDistributed
        return _backend_cache[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
