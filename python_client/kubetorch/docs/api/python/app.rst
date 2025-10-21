App
===

The ``App`` class wraps a Python CLI command. It syncs over the file and any necessary requirements to the specified
compute, where it runs your file remotely. The file can be any Python file: a basic training script, a script that
uses kubetorch to deploy further services, or a FastAPI app.


Factory Method
~~~~~~~~~~~~~~

.. autofunction:: kubetorch.app

App Class
~~~~~~~~~

.. autoclass:: kubetorch.App
   :members:
   :exclude-members: from_name

    .. automethod:: __init__
