from kubernetes.client import CustomObjectsApi


class K8sMetricsClient:
    def __init__(self, namespace: str, objects_api: CustomObjectsApi):
        self.namespace = namespace
        self.objects_api = objects_api

    def get_pod_metrics(self):
        metrics = self.objects_api.list_namespaced_custom_object(
            group="metrics.k8s.io",
            version="v1beta1",
            namespace=self.namespace,
            plural="pods",
        )
        pod_data = {}
        for item in metrics.get("items", []):
            pod = item["metadata"]["name"]
            containers = item.get("containers", [])
            cpu_millicores = 0
            mem_mib = 0
            for c in containers:
                usage = c["usage"]
                cpu_str = usage.get("cpu", "0")
                mem_str = usage.get("memory", "0")
                cpu_millicores += self._parse_cpu(cpu_str)
                mem_mib += self._parse_mem(mem_str)
            pod_data[pod] = {"CPU": cpu_millicores / 1000, "Mem": mem_mib}
        return pod_data

    def _parse_cpu(self, cpu_str):
        if cpu_str.endswith("n"):  # nanocores
            return int(cpu_str[:-1]) / 1e6
        if cpu_str.endswith("m"):  # millicores
            return float(cpu_str[:-1])
        return float(cpu_str) * 1000

    def _parse_mem(self, mem_str):
        # Convert to MiB
        units = {"Ki": 1 / 1024, "Mi": 1, "Gi": 1024}
        for suffix, factor in units.items():
            if mem_str.endswith(suffix):
                return float(mem_str[: -len(suffix)]) * factor
        return float(mem_str) / (1024**2)
