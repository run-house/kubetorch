Python API
==========

The API Reference provides detailed information about the Kubetorch Python API and CLI commands.

If you are just getting started with Kubetorch or looking for use cases and examples, we recommend first checking out:

* `Guides <https://www.run.house/kubetorch/introduction>`_: quick start, high level concepts, developer guides, and more

* `Examples <https://www.run.house/examples>`_: end-to-end examples using Kubetorch

Compute
-------

The ``Compute`` class allows you to define the resources and environment needed for your workloads,
while controlling how the compute is managed and scaled based on demand. This includes specifying
hardware requirements that can be either generic or tailored to your specific Kubernetes infrastructure and setup.

.. toctree::
   :maxdepth: 1

   python/compute

Image
------

The ``Image`` class enables you to define and customize the containerized environment for your workloads.
You can specify a pre-built Docker image as your foundation and layer on additional setup steps that run
at launch time, eliminating the need to rebuild images for every code change.

.. toctree::
   :maxdepth: 1

   python/image


Module
------

The ``Fn`` and ``Cls`` classes are wrappers around your locally defined Python functions and classes, respectively.
Once wrapped, these objects can be sent ``.to(compute)``, which launches a service on your cluster (taking into
account the compute requirements) and syncs over the necessary files to run the function remotely.


.. toctree::
   :maxdepth: 1

   python/fn

.. toctree::
   :maxdepth: 1

   python/cls


App
---

The ``App`` class wraps a Python CLI command or script, enabling you to run entire applications remotely on the cluster.
Unlike ``Fn`` and ``Cls`` which wrap individual functions or classes, ``App`` deploys and executes complete Python files
with all their dependencies, making it ideal for training scripts, data processing pipelines, or even web applications.

.. toctree::
   :maxdepth: 1

   python/app

Secrets
-------

Secrets such as provider keys and environment variables can be set when defining compute. These are set at launch time
and accessible during the scope of your program.

.. toctree::
   :maxdepth: 1

   python/secret

Config
------

Kubetorch uses a local configuration file (stored at ``~/.kt/config.yaml``) to allow you to set global defaults for
your services. You can update the config file manually, use the ``kt config`` command, or set them as environment variables.
You can also override defaults directly in the resource constructor for a specific service.

.. toctree::
   :maxdepth: 1

   python/config

Volumes
-------

The ``Volume`` class enables persistent storage for your workloads, allowing data to persist beyond individual pod lifecycles.
Kubetorch automatically manages Kubernetes PersistentVolumeClaims (PVCs) while providing a simple Python interface for
storage configuration.

.. toctree::
   :maxdepth: 1

   python/volumes
