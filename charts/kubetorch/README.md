# kubetorch

![Version: 0.2.9](https://img.shields.io/badge/Version-0.2.9-informational?style=flat-square) ![Type: application](https://img.shields.io/badge/Type-application-informational?style=flat-square) ![AppVersion: 0.2.9](https://img.shields.io/badge/AppVersion-0.2.9-informational?style=flat-square)

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
| controller.affinity | object | `{}` |  |
| controller.image | string | `"ghcr.io/run-house/kubetorch-controller"` |  |
| controller.imagePullPolicy | string | `"Always"` |  |
| controller.resources.cpu.limit | string | `"500m"` |  |
| controller.resources.cpu.request | string | `"100m"` |  |
| controller.resources.memory.limit | string | `"512Mi"` |  |
| controller.resources.memory.request | string | `"256Mi"` |  |
| controller.tag | string | `"v0.1.0"` |  |
| controller.tolerations | list | `[]` |  |
| dcgm-exporter.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].key | string | `"karpenter.k8s.aws/instance-gpu-manufacturer"` |  |
| dcgm-exporter.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].operator | string | `"In"` |  |
| dcgm-exporter.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].values[0] | string | `"nvidia"` |  |
| dcgm-exporter.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[1].key | string | `"karpenter.k8s.aws/instance-gpu-name"` |  |
| dcgm-exporter.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[1].operator | string | `"In"` |  |
| dcgm-exporter.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[1].values[0] | string | `"a10g"` |  |
| dcgm-exporter.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[1].values[1] | string | `"a100"` |  |
| dcgm-exporter.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[1].values[2] | string | `"t4"` |  |
| dcgm-exporter.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[1].matchExpressions[0].key | string | `"nvidia.com/gpu.present"` |  |
| dcgm-exporter.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[1].matchExpressions[0].operator | string | `"Exists"` |  |
| dcgm-exporter.enabled | bool | `false` |  |
| dcgm-exporter.image.repository | string | `"nvcr.io/nvidia/k8s/dcgm-exporter"` |  |
| dcgm-exporter.image.tag | string | `"4.4.1-4.6.0-ubuntu22.04"` |  |
| dcgm-exporter.namespaceOverride | string | `"kubetorch"` |  |
| dcgm-exporter.readinessProbe.enabled | bool | `false` |  |
| dcgm-exporter.serviceMonitor.enabled | bool | `false` |  |
| dcgm-exporter.tolerations[0].effect | string | `"NoSchedule"` |  |
| dcgm-exporter.tolerations[0].key | string | `"nvidia.com/gpu"` |  |
| dcgm-exporter.tolerations[0].operator | string | `"Exists"` |  |
| kubetorchConfig.deployment_namespaces[0] | string | `"default"` |  |
| kubetorchConfig.deployment_namespaces[1] | string | `"kubetorch"` |  |
| kubetorchConfig.otelEnabled | bool | `true` |  |
| logStreaming.enabled | bool | `true` |  |
| logStreaming.image | string | `"grafana/loki:3.5.3"` |  |
| logStreaming.maxConcurrentTailRequests | int | `100` |  |
| logStreaming.port | int | `3100` |  |
| logStreaming.resources.cpu | string | `"100m"` |  |
| logStreaming.resources.memory | string | `"256Mi"` |  |
| logStreaming.retentionPeriod | string | `"24h"` |  |
| metrics.enabled | bool | `true` |  |
| metrics.prometheus.additionalScrapeConfigs[0].honor_labels | bool | `true` |  |
| metrics.prometheus.additionalScrapeConfigs[0].job_name | string | `"gke-managed-dcgm"` |  |
| metrics.prometheus.additionalScrapeConfigs[0].kubernetes_sd_configs[0].namespaces.names[0] | string | `"gke-managed-system"` |  |
| metrics.prometheus.additionalScrapeConfigs[0].kubernetes_sd_configs[0].role | string | `"pod"` |  |
| metrics.prometheus.additionalScrapeConfigs[0].relabel_configs[0].action | string | `"keep"` |  |
| metrics.prometheus.additionalScrapeConfigs[0].relabel_configs[0].regex | string | `"dcgm-exporter.*"` |  |
| metrics.prometheus.additionalScrapeConfigs[0].relabel_configs[0].source_labels[0] | string | `"__meta_kubernetes_pod_name"` |  |
| metrics.prometheus.additionalScrapeConfigs[0].relabel_configs[1].action | string | `"keep"` |  |
| metrics.prometheus.additionalScrapeConfigs[0].relabel_configs[1].regex | string | `"9400"` |  |
| metrics.prometheus.additionalScrapeConfigs[0].relabel_configs[1].source_labels[0] | string | `"__meta_kubernetes_pod_container_port_number"` |  |
| metrics.prometheus.additionalScrapeConfigs[0].relabel_configs[2].replacement | string | `"$1:9400"` |  |
| metrics.prometheus.additionalScrapeConfigs[0].relabel_configs[2].source_labels[0] | string | `"__meta_kubernetes_pod_ip"` |  |
| metrics.prometheus.additionalScrapeConfigs[0].relabel_configs[2].target_label | string | `"__address__"` |  |
| metrics.prometheus.additionalScrapeConfigs[0].relabel_configs[3].source_labels[0] | string | `"__meta_kubernetes_namespace"` |  |
| metrics.prometheus.additionalScrapeConfigs[0].relabel_configs[3].target_label | string | `"namespace"` |  |
| metrics.prometheus.additionalScrapeConfigs[0].relabel_configs[4].source_labels[0] | string | `"__meta_kubernetes_pod_name"` |  |
| metrics.prometheus.additionalScrapeConfigs[0].relabel_configs[4].target_label | string | `"pod"` |  |
| metrics.prometheus.image | string | `"prom/prometheus:v3.7.2"` |  |
| metrics.prometheus.port | int | `9090` |  |
| metrics.prometheus.resources.cpu | string | `"200m"` |  |
| metrics.prometheus.resources.memory | string | `"512Mi"` |  |
| metrics.prometheus.retention | string | `"24h"` |  |
| metrics.prometheus.scrapeInterval | string | `"3s"` |  |
| metrics.scrapeKubelet | bool | `true` |  |
| nginx.resolver | string | `"kube-dns.kube-system.svc.cluster.local"` |  |
| nginxProxy.affinity | object | `{}` |  |
| nginxProxy.backends.health.route | string | `"/health"` |  |
| nginxProxy.backends.logging.route | string | `"/loki"` |  |
| nginxProxy.image.pullPolicy | string | `"IfNotPresent"` |  |
| nginxProxy.image.repository | string | `"nginx"` |  |
| nginxProxy.image.tag | string | `"1.29.0-alpine"` |  |
| nginxProxy.maxBodySize.api | string | `"250M"` |  |
| nginxProxy.maxBodySize.rsync | string | `"10G"` |  |
| nginxProxy.tolerations | list | `[]` |  |
| nvidia-device-plugin.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].key | string | `"nvidia.com/gpu.product"` |  |
| nvidia-device-plugin.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].operator | string | `"Exists"` |  |
| nvidia-device-plugin.enabled | bool | `true` |  |
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
| opentelemetry-collector.enabled | bool | `true` |  |
| opentelemetry-collector.image.pullPolicy | string | `"IfNotPresent"` |  |
| opentelemetry-collector.image.repository | string | `"otel/opentelemetry-collector-contrib"` |  |
| opentelemetry-collector.mode | string | `"daemonset"` |  |
| opentelemetry-collector.presets.kubernetesAttributes.enabled | bool | `true` |  |
| opentelemetry-collector.presets.kubernetesAttributes.extractAllPodLabels | bool | `true` |  |
| opentelemetry-collector.presets.kubernetesEvents.enabled | bool | `true` |  |
| opentelemetry-collector.presets.logsCollection.enabled | bool | `true` |  |
| opentelemetry-collector.presets.logsCollection.includeCollectorLogs | bool | `true` |  |
| opentelemetry-collector.serviceAccount.create | bool | `false` |  |
| opentelemetry-collector.serviceAccount.name | string | `"kubetorch-opentelemetry-collector"` |  |
| opentelemetry-collector.tolerations[0].operator | string | `"Exists"` |  |
| rsync.affinity | object | `{}` |  |
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
| rsync.tolerations | list | `[]` |  |

----------------------------------------------
Autogenerated from chart metadata using [helm-docs v1.14.2](https://github.com/norwoodj/helm-docs/releases/v1.14.2)
