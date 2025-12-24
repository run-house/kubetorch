Data Store
==========

The Data Store provides a key-value interface for transferring data to and from your Kubernetes cluster.
It supports two data types:

- **Filesystem data**: Files and directories transferred via rsync
- **GPU data**: CUDA tensors and state dicts transferred via NCCL broadcast

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
