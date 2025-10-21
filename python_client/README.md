# 📦Kubetorch🔥

**A Python interface for running ML workloads on Kubernetes**

Kubetorch enables you to run any Python code on Kubernetes at any scale by specifying required resources, distribution, and scaling directly in code. It provides caching and hot redeployment for 1-2 second iteration cycles, handles hardware faults and preemptions programmatically, and orchestrates complex, heterogeneous workloads with built-in observability and fault tolerance.

## Hello World

```python
from kubetorch import fn

def hello_world():
    return "Hello from Kubetorch!"

if __name__ == "__main__":
    # Define your compute
    compute = kt.Compute(cpus=".1")

    # Send local function to freshly launched remote compute
    remote_hello = kt.fn(hello_world).to(compute)

    # Runs remotely on your Kubernetes cluster
    result = hello_world()
    print(result)  # "Hello from Kubetorch!"
```

## What Kubetorch Enables

- **100x faster iteration** from 10+ minutes to 1-3 seconds for complex ML applications like RL and distributed training
- **50%+ compute cost savings** through intelligent resource allocation, bin-packing, and dynamic scaling
- **95% fewer production faults** with built-in fault handling with programmatic error recovery and resource adjustment

## Installation

### 1. Python Client

```bash
pip install "kubetorch[client]"
```

### 2. Kubernetes Deployment (Helm)

```bash
# Option 1: Install directly from OCI registry
helm upgrade --install kubetorch oci://ghcr.io/run-house/charts/kubetorch \
  --version 0.2.0 -n kubetorch --create-namespace

# Option 2: Download chart locally first
helm pull oci://ghcr.io/run-house/charts/kubetorch --version 0.2.0 --untar
helm upgrade --install kubetorch ./kubetorch -n kubetorch --create-namespace
```

For detailed setup instructions, see our [Installation Guide](https://www.run.house/kubetorch/installation).


## Learn More

- **[Documentation](https://www.run.house/kubetorch/introduction)** - API Reference, concepts, and guides
- **[Examples](https://www.run.house/kubetorch/examples)** - Real-world usage patterns and tutorials
- **[Join our Slack](https://join.slack.com/t/kubetorch/shared_invite/zt-3g76q5i4j-uP60AdydxnAmjGVAQhtALA)** - Connect with the community and get support

---

[Apache 2.0 License](LICENSE)

**🏃‍♀️ Built by [Runhouse](https://www.run.house) 🏠**
