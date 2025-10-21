Compute
=======

The ``Compute`` class lets you specify the right resources to request for your workloads, and control how that compute
behaves.

Compute Class
~~~~~~~~~~~~~~

.. autoclass:: kubetorch.Compute
   :members:
   :exclude-members: autoscale, distribute

    .. automethod:: __init__


Autoscaling
~~~~~~~~~~~

.. automethod:: kubetorch.Compute.autoscale

Distributed
~~~~~~~~~~~

.. automethod:: kubetorch.Compute.distribute
