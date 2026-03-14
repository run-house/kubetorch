"""
Centralized Kubernetes client registry.

All K8s API clients are initialized once in server.py and accessed from here.
"""

# K8s API clients - initialized by init()
apps_v1 = None
core_v1 = None
custom_objects = None
dynamic = None


def init(apps, core, custom, dyn):
    """Initialize K8s API clients. Called once from server.py."""
    global apps_v1, core_v1, custom_objects, dynamic
    apps_v1 = apps
    core_v1 = core
    custom_objects = custom
    dynamic = dyn
