Data Store
==========

The Kubetorch Data Store is a intuitive and scalable distributed data system for Kubernetes, solving two critical gaps:

1. **Out-of-cluster direct transfers**: Sync code and data up to your cluster instantly and scalably - no need for container rebuilds or bouncing off blob storage
2. **In-cluster data transfer and caching**: fast peer-to-peer data transfer between pods with automatic caching and discovery, for filesystem and GPU data

The unified APIs handle two types of data:
- **Filesystem data**: Files/directories via distributed rsync (zero-copy P2P or to/from central store)
- **GPU data**: CUDA tensors/state dicts via NCCL broadcast

Key capabilities include:
- External sync: Push/pull files to/from cluster
- Zero-copy P2P: locale="local" publishes data in-place, consumers fetch directly
- Scalable broadcast: Tree-based differential propagation for distributing to thousands of pods (no NFS thundering herd or "many small files" problems)
- GPU transfers: Point-to-point and coordinated broadcasts for tensors/state dicts with Infiniband/RDMA support
- Automatic caching: Every getter can become a source for subsequent getters
- Automatic lifecycle management, with TTLs and cleanup built-in


Python API
----------

The top-level functions ``kt.put()``, ``kt.get()``, ``kt.ls()``, and ``kt.rm()`` provide a simple interface
for data operations. The data type (filesystem vs GPU) is auto-detected based on the parameters you provide.

put
^^^

.. autofunction:: kubetorch.data_store.put

get
^^^

.. autofunction:: kubetorch.data_store.get

ls
^^

.. autofunction:: kubetorch.data_store.ls

rm
^^

.. autofunction:: kubetorch.data_store.rm

Supporting Types
----------------

BroadcastWindow
^^^^^^^^^^^^^^^

.. autoclass:: kubetorch.data_store.types.BroadcastWindow
   :members:
   :show-inheritance:

Locale
^^^^^^

.. autodata:: kubetorch.data_store.types.Locale

Lifespan
^^^^^^^^

.. autodata:: kubetorch.data_store.types.Lifespan

CLI Commands
------------

The following CLI commands provide the same functionality from the command line:

kt put
^^^^^^

Store files or directories in the cluster.

.. code-block:: bash

   kt put <key> --src <path> [options]

   # Examples
   kt put my-service/weights --src ./trained_model/
   kt put datasets/train --src ./data/ --contents
   kt put my-service/models --src ./model1/ --src ./model2/

Options:

- ``--src, -s``: Local file(s) or directory(s) to upload (required, can be specified multiple times)
- ``--force, -f``: Force overwrite of existing files
- ``--exclude``: Exclude patterns (rsync format, e.g., ``'*.pyc'``)
- ``--include``: Include patterns to override .gitignore exclusions
- ``--contents, -c``: Copy directory contents (adds trailing slashes for rsync)
- ``--verbose, -v``: Show detailed progress
- ``--namespace, -n``: Kubernetes namespace

kt get
^^^^^^

Retrieve files or directories from the cluster.

.. code-block:: bash

   kt get <key> [--dest <path>] [options]

   # Examples
   kt get my-service/weights                    # Download to current directory
   kt get my-service/weights --dest ./local/    # Download to specific path
   kt get datasets/train --contents             # Download directory contents

Options:

- ``--dest, -d``: Local destination path (defaults to current directory)
- ``--force, -f``: Force overwrite of existing files
- ``--exclude``: Exclude patterns (rsync format)
- ``--include``: Include patterns
- ``--contents, -c``: Copy directory contents
- ``--seed-data/--no-seed-data``: Automatically seed data after retrieval for peer-to-peer (default: enabled)
- ``--verbose, -v``: Show detailed progress
- ``--namespace, -n``: Kubernetes namespace

kt ls
^^^^^

List files and directories in the cluster store.

.. code-block:: bash

   kt ls [key] [options]

   # Examples
   kt ls                      # List root of store
   kt ls my-service           # List contents of my-service
   kt ls my-service/models    # List models directory

Options:

- ``--verbose, -v``: Show detailed progress
- ``--namespace, -n``: Kubernetes namespace

kt rm
^^^^^

Delete files or directories from the cluster store.

.. code-block:: bash

   kt rm <key> [options]

   # Examples
   kt rm my-service/old-model.pkl              # Delete a file
   kt rm my-service/temp-data --recursive      # Delete a directory

Options:

- ``--recursive, -r``: Delete directories recursively
- ``--verbose, -v``: Show detailed progress
- ``--namespace, -n``: Kubernetes namespace
