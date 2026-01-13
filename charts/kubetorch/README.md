# kubetorch

![Version: 0.3.0](https://img.shields.io/badge/Version-0.3.0-informational?style=flat-square) ![Type: application](https://img.shields.io/badge/Type-application-informational?style=flat-square) ![AppVersion: 0.3.0](https://img.shields.io/badge/AppVersion-0.3.0-informational?style=flat-square)

A Helm chart for kubetorch

## Requirements

| Repository | Name | Version |
|------------|------|---------|
| https://nvidia.github.io/dcgm-exporter/helm-charts | dcgm-exporter | 4.5.0 |
| https://nvidia.github.io/k8s-device-plugin | nvidia-device-plugin | 0.14.1 |

## Values

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| dataStore.affinity | object | `{}` |  |
| dataStore.cleanupCron.enabled | bool | `false` |  |
| dataStore.cpu.request | int | `1` |  |
| dataStore.ephemeralStorage.limit | string | `"10Gi"` |  |
| dataStore.ephemeralStorage.request | string | `"5Gi"` |  |
| dataStore.image | string | `"ghcr.io/run-house/kubetorch-data-store:0.4.0"` |  |
| dataStore.imagePullPolicy | string | `"Always"` |  |
| dataStore.maxConnections | int | `500` |  |
| dataStore.maxConnectionsPerModule | int | `0` |  |
| dataStore.maxVerbosity | int | `0` |  |
| dataStore.memory.limit | string | `"8Gi"` |  |
| dataStore.memory.request | string | `"4Gi"` |  |
| dataStore.storage.size | string | `"100Gi"` |  |
| dataStore.storage.storageClassName | string | `""` |  |
| dataStore.timeout | int | `600` |  |
| dataStore.tolerations | list | `[]` |  |
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
| kubetorchConfig.logStreamingEnabled | bool | `true` |  |
| kubetorchConfig.metricsEnabled | bool | `true` |  |
| kubetorchConfig.serviceAccountAnnotations | object | `{}` |  |
| kubetorchController.affinity | object | `{}` |  |
| kubetorchController.connectionPoolSize | int | `20` |  |
| kubetorchController.eventWatcher.batchSize | int | `10` |  |
| kubetorchController.eventWatcher.enabled | bool | `true` |  |
| kubetorchController.eventWatcher.flushInterval | float | `1` |  |
| kubetorchController.image | string | `"ghcr.io/run-house/kubetorch-controller"` |  |
| kubetorchController.imagePullPolicy | string | `"Always"` |  |
| kubetorchController.nginx.healthRoute | string | `"/health"` |  |
| kubetorchController.nginx.image.pullPolicy | string | `"IfNotPresent"` |  |
| kubetorchController.nginx.image.repository | string | `"nginx"` |  |
| kubetorchController.nginx.image.tag | string | `"1.29.0-alpine"` |  |
| kubetorchController.nginx.maxBodySize.api | string | `"250M"` |  |
| kubetorchController.nginx.maxBodySize.rsync | string | `"10G"` |  |
| kubetorchController.nginx.resolver | string | `"kube-dns.kube-system.svc.cluster.local"` |  |
| kubetorchController.nginx.resources.cpu.limit | string | `"1"` |  |
| kubetorchController.nginx.resources.cpu.request | string | `"200m"` |  |
| kubetorchController.nginx.resources.memory.limit | string | `"512Mi"` |  |
| kubetorchController.nginx.resources.memory.request | string | `"256Mi"` |  |
| kubetorchController.port | int | `8081` |  |
| kubetorchController.resources.cpu.limit | string | `"2000m"` |  |
| kubetorchController.resources.cpu.request | string | `"200m"` |  |
| kubetorchController.resources.memory.limit | string | `"4Gi"` |  |
| kubetorchController.resources.memory.request | string | `"2Gi"` |  |
| kubetorchController.servicePort | int | `8080` |  |
| kubetorchController.storage.size | string | `"1Gi"` |  |
| kubetorchController.tag | string | `"0.4.0"` |  |
| kubetorchController.tolerations | list | `[]` |  |
| logStreaming.enabled | bool | `true` |  |
| logStreaming.ingestionBurstSizeMb | int | `20` |  |
| logStreaming.ingestionRateMb | int | `10` |  |
| logStreaming.maxConcurrentTailRequests | int | `100` |  |
| logStreaming.maxGlobalStreamsPerUser | int | `50000` |  |
| logStreaming.port | int | `3100` |  |
| logStreaming.retentionPeriod | string | `"24h"` |  |
| metrics.enabled | bool | `true` |  |
| metrics.prometheus.additionalScrapeConfigs[0].honor_labels | bool | `true` |  |
| metrics.prometheus.additionalScrapeConfigs[0].job_name | string | `"dcgm-exporter"` |  |
| metrics.prometheus.additionalScrapeConfigs[0].kubernetes_sd_configs[0].namespaces.names[0] | string | `"gke-managed-system"` |  |
| metrics.prometheus.additionalScrapeConfigs[0].kubernetes_sd_configs[0].namespaces.names[1] | string | `"gpu-operator"` |  |
| metrics.prometheus.additionalScrapeConfigs[0].kubernetes_sd_configs[0].namespaces.names[2] | string | `"nvidia-gpu-operator"` |  |
| metrics.prometheus.additionalScrapeConfigs[0].kubernetes_sd_configs[0].role | string | `"pod"` |  |
| metrics.prometheus.additionalScrapeConfigs[0].relabel_configs[0].action | string | `"keep"` |  |
| metrics.prometheus.additionalScrapeConfigs[0].relabel_configs[0].regex | string | `".*dcgm-exporter.*"` |  |
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
| ttlController.enabled | bool | `true` |  |
| ttlController.namespaces | list | `[]` |  |
| ttlController.networkPolicy.enabled | bool | `false` |  |
| ttlController.networkPolicy.ports[0].port | int | `9090` |  |
| ttlController.networkPolicy.ports[0].protocol | string | `"TCP"` |  |
| ttlController.podMonitor.additionalLabels | object | `{}` |  |
| ttlController.podMonitor.enabled | bool | `false` |  |
| ttlController.podMonitor.prometheusLabel | string | `"kube-prometheus"` |  |
| ttlController.prometheusNamespace | string | `"kubetorch"` |  |
| ttlController.prometheusUrl | string | `"http://kubetorch-metrics.kubetorch.svc.cluster.local:9090"` |  |

----------------------------------------------
Autogenerated from chart metadata using [helm-docs v1.14.2](https://github.com/norwoodj/helm-docs/releases/v1.14.2)
