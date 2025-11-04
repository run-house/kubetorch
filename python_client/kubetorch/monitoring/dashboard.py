import subprocess
import webbrowser

from kubernetes import client

from kubetorch.constants import DASHBOARD_PORT, GRAFANA_PORT

from kubetorch.serving.constants import GRAFANA_HEALTH_ENDPOINT, KUBETORCH_NAMESPACE
from kubetorch.serving.utils import wait_for_port_forward


def port_forward_grafana(v1_api: client.CoreV1Api):
    """
    Port-forward svc/kubetorch-grafana in kubetorch ns to localhost:<local_port>.
    Auto-detects the service's remote port (prefers 3000).
    """
    namespace = KUBETORCH_NAMESPACE
    local_port = DASHBOARD_PORT

    svc_name = "kubetorch-grafana"

    # Discover the correct remote port from the Service
    svc = v1_api.read_namespaced_service(svc_name, namespace)

    remote_port = GRAFANA_PORT
    if svc.spec and svc.spec.ports:
        # Prefer explicit 3000 first
        pref = next((p for p in svc.spec.ports if p.port == 3000), None)
        if pref:
            remote_port = pref.port
        else:
            # Otherwise pick first declared port
            remote_port = svc.spec.ports[0].port

    cmd = [
        "kubectl",
        "port-forward",
        "-n",
        namespace,
        f"svc/{svc_name}",
        f"{local_port}:{remote_port}",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    wait_for_port_forward(
        proc,
        local_port,
        health_endpoint=GRAFANA_HEALTH_ENDPOINT,
        validate_kubetorch_versions=False,
    )
    return proc


def open_grafana_dashboard(
    user_namespace: str,
    console: "Console",
    v1_api: client.CoreV1Api,
):
    try:
        v1_api.read_namespaced_service("kubetorch-prometheus", KUBETORCH_NAMESPACE)
    except client.exceptions.ApiException as e:
        raise RuntimeError(str(e))

    url = (
        f"http://localhost:{DASHBOARD_PORT}/d/kubetorch-basic/metrics"
        f"?orgId=1&from=now-5m&to=now&timezone=browser&refresh=10s"
        f"&var-namespace={user_namespace}"
    )
    proc = port_forward_grafana(v1_api=v1_api)

    try:
        webbrowser.open(url)
        console.print("[yellow]Press Ctrl+C to stop port-forwarding.[/yellow]")
        proc.wait()

    except KeyboardInterrupt:
        console.print("Stopping port-forward...")
        proc.terminate()
