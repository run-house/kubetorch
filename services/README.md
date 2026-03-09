# Services

Customer-facing service source now lives in this repo:

- `services/kubetorch_controller/`: the Kubernetes API/controller service deployed by the OSS Helm chart
- `services/data_store/`: the data store and rsync/logging sidecar service deployed by the OSS Helm chart

These directories are the build contexts used by the release scripts in `release/`.
