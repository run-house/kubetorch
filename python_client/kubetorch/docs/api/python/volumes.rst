Volumes
=======

Kubetorch provides persistent storage through the ``Volume`` class, which abstracts Kubernetes PersistentVolumeClaims
while maintaining the flexibility to work with any storage backend your cluster supports.

Volume Class
~~~~~~~~~~~~~~

.. autoclass:: kubetorch.Volume
   :members:

    .. automethod:: __init__
