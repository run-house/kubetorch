# kubetorch

![Version: 0.2.3](https://img.shields.io/badge/Version-0.2.3-informational?style=flat-square) ![Type: application](https://img.shields.io/badge/Type-application-informational?style=flat-square) ![AppVersion: 0.2.3](https://img.shields.io/badge/AppVersion-0.2.3-informational?style=flat-square)

A Helm chart for kubetorch

## Requirements

| Repository | Name | Version |
|------------|------|---------|
| https://nvidia.github.io/dcgm-exporter/helm-charts | dcgm-exporter | 4.5.0 |
| https://nvidia.github.io/k8s-device-plugin | nvidia-device-plugin | 0.14.1 |
| https://open-telemetry.github.io/opentelemetry-helm-charts | opentelemetry-collector | 0.132.0 |

## Values

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| dcgm-exporter.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].key | string | `"nvidia.com/gpu.product"` |  |
| dcgm-exporter.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].operator | string | `"Exists"` |  |
| dcgm-exporter.enabled | bool | `true` |  |
| dcgm-exporter.image.repository | string | `"nvcr.io/nvidia/k8s/dcgm-exporter"` |  |
| dcgm-exporter.image.tag | string | `"4.4.1-4.6.0-ubuntu22.04"` |  |
| dcgm-exporter.namespaceOverride | string | `"kubetorch"` |  |
| dcgm-exporter.readinessProbe.enabled | bool | `false` |  |
| dcgm-exporter.serviceMonitor.enabled | bool | `false` |  |
| dcgm-exporter.tolerations[0].effect | string | `"NoSchedule"` |  |
| dcgm-exporter.tolerations[0].key | string | `"nvidia.com/gpu"` |  |
| dcgm-exporter.tolerations[0].operator | string | `"Exists"` |  |
| dcgm-exporter.tolerations[1].effect | string | `"NoSchedule"` |  |
| dcgm-exporter.tolerations[1].key | string | `"dedicated"` |  |
| dcgm-exporter.tolerations[1].operator | string | `"Equal"` |  |
| dcgm-exporter.tolerations[1].value | string | `"gpu"` |  |
| ephemeralLogStorage.enabled | bool | `true` |  |
| ephemeralLogStorage.image | string | `"grafana/loki:3.5.3"` |  |
| ephemeralLogStorage.maxConcurrentTailRequests | int | `100` |  |
| ephemeralLogStorage.port | int | `3100` |  |
| ephemeralLogStorage.resources.cpu | string | `"100m"` |  |
| ephemeralLogStorage.resources.memory | string | `"256Mi"` |  |
| ephemeralLogStorage.retentionPeriod | string | `"24h"` |  |
| ephemeralMonitoring.enabled | bool | `true` |  |
| ephemeralMonitoring.grafana.anonymous.allowEmbedding | bool | `true` |  |
| ephemeralMonitoring.grafana.anonymous.disableLoginForm | bool | `true` |  |
| ephemeralMonitoring.grafana.anonymous.enabled | bool | `true` |  |
| ephemeralMonitoring.grafana.anonymous.orgRole | string | `"Viewer"` |  |
| ephemeralMonitoring.grafana.image | string | `"grafana/grafana:main"` |  |
| ephemeralMonitoring.grafana.prometheusURL | string | `"http://kubetorch-prometheus.kubetorch.svc.cluster.local:9090"` |  |
| ephemeralMonitoring.prometheus.image | string | `"prom/prometheus:v3.7.2"` |  |
| ephemeralMonitoring.prometheus.port | int | `9090` |  |
| ephemeralMonitoring.prometheus.resources.cpu | string | `"200m"` |  |
| ephemeralMonitoring.prometheus.resources.memory | string | `"512Mi"` |  |
| ephemeralMonitoring.prometheus.retention | string | `"1h"` |  |
| ephemeralMonitoring.prometheus.scrapeInterval | string | `"5s"` |  |
| ephemeralMonitoring.scrapeKubeStateMetrics | bool | `false` |  |
| ephemeralMonitoring.scrapeKubelet | bool | `true` |  |
| ephemeralMonitoring.scrapeNodeExporter | bool | `false` |  |
| gpuSupport.enabled | bool | `true` |  |
| kubetorchConfig.deployment_namespaces[0] | string | `"default"` |  |
| kubetorchConfig.deployment_namespaces[1] | string | `"kubetorch"` |  |
| kubetorchConfig.environment.PROMETHEUS_NAMESPACE | string | `""` |  |
| kubetorchConfig.environment.PROMETHEUS_URL | string | `""` |  |
| kubetorchConfig.otelEnabled | bool | `true` |  |
| nginx.resolver | string | `"kube-dns.kube-system.svc.cluster.local"` |  |
| nginxProxy.backends.health.route | string | `"/health"` |  |
| nginxProxy.backends.logging.host | string | `"loki-gateway.kubetorch.svc.cluster.local"` |  |
| nginxProxy.backends.logging.route | string | `"/loki"` |  |
| nginxProxy.backends.metrics.ephemeral.host | string | `"kubetorch-prometheus.kubetorch.svc.cluster.local"` |  |
| nginxProxy.backends.metrics.ephemeral.port | int | `9090` |  |
| nginxProxy.backends.metrics.persistent.host | string | `"runhouse-kube-prometheus-s-prometheus.runhouse.svc.cluster.local"` |  |
| nginxProxy.backends.metrics.persistent.port | int | `9090` |  |
| nginxProxy.image.pullPolicy | string | `"IfNotPresent"` |  |
| nginxProxy.image.repository | string | `"nginx"` |  |
| nginxProxy.image.tag | string | `"1.29.0-alpine"` |  |
| nginxProxy.maxBodySize.api | string | `"250M"` |  |
| nginxProxy.maxBodySize.rsync | string | `"10G"` |  |
| nvidia-device-plugin.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].key | string | `"nvidia.com/gpu.product"` |  |
| nvidia-device-plugin.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].operator | string | `"Exists"` |  |
| nvidia-device-plugin.namespaceOverride | string | `"kubetorch"` |  |
| nvidia-device-plugin.tolerations[0].effect | string | `"NoSchedule"` |  |
| nvidia-device-plugin.tolerations[0].key | string | `"nvidia.com/gpu"` |  |
| nvidia-device-plugin.tolerations[0].operator | string | `"Exists"` |  |
| nvidia-device-plugin.tolerations[1].effect | string | `"NoSchedule"` |  |
| nvidia-device-plugin.tolerations[1].key | string | `"dedicated"` |  |
| nvidia-device-plugin.tolerations[1].operator | string | `"Equal"` |  |
| nvidia-device-plugin.tolerations[1].value | string | `"gpu"` |  |
| opentelemetry-collector.configMap.create | bool | `false` |  |
| opentelemetry-collector.configMap.existingName | string | `"otel-collector-config"` |  |
| opentelemetry-collector.image.pullPolicy | string | `"IfNotPresent"` |  |
| opentelemetry-collector.image.repository | string | `"otel/opentelemetry-collector-contrib"` |  |
| opentelemetry-collector.mode | string | `"daemonset"` |  |
| opentelemetry-collector.presets.kubernetesAttributes.enabled | bool | `true` |  |
| opentelemetry-collector.presets.kubernetesAttributes.extractAllPodLabels | bool | `true` |  |
| opentelemetry-collector.presets.kubernetesEvents.enabled | bool | `true` |  |
| opentelemetry-collector.presets.logsCollection.enabled | bool | `true` |  |
| opentelemetry-collector.presets.logsCollection.includeCollectorLogs | bool | `true` |  |
| opentelemetry-collector.tolerations[0].operator | string | `"Exists"` |  |
| otelCollector.enabled | bool | `true` |  |
| rsync.cleanupCron.enabled | bool | `false` |  |
| rsync.cpu.limit | int | `4` |  |
| rsync.cpu.request | int | `2` |  |
| rsync.ephemeralStorage.limit | string | `"10Gi"` |  |
| rsync.ephemeralStorage.request | string | `"5Gi"` |  |
| rsync.image | string | `"ghcr.io/run-house/kubetorch-rsync:v5"` |  |
| rsync.maxConnections | int | `500` |  |
| rsync.maxConnectionsPerModule | int | `0` |  |
| rsync.maxVerbosity | int | `0` |  |
| rsync.memory.limit | string | `"8Gi"` |  |
| rsync.memory.request | string | `"4Gi"` |  |
| rsync.timeout | int | `600` |  |

----------------------------------------------
Autogenerated from chart metadata using [helm-docs v1.14.2](https://github.com/norwoodj/helm-docs/releases/v1.14.2)
