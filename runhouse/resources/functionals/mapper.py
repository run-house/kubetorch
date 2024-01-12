import logging
from typing import List, Optional, Union

import ray

from runhouse.resources.functions import function, Function

from runhouse.resources.module import Module

logger = logging.getLogger(__name__)


class Mapper(Module):
    def __init__(
        self,
        module: Module,
        method: str,
        num_replicas: Optional[int] = -1,
        replicas: Optional[List[Module]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.module = module
        self.method = method
        self.num_replicas = num_replicas
        self._auto_replicas = []
        self._user_replicas = replicas or []
        self._last_called = 0
        if self.num_replicas > len(self.replicas) and self.num_replicas > 0:
            self._add_auto_replicas(self.num_replicas - len(self.replicas))

    @property
    def replicas(self):
        return [self.module] + self._auto_replicas + self._user_replicas

    def add_replicas(self, replicas: Union[int, List[Module]]):
        if isinstance(replicas, int):
            self.num_replicas += replicas
            self._add_auto_replicas(self.num_replicas - len(self.replicas))
        else:
            self._user_replicas.extend(replicas)

    def drop_replicas(self, num_replicas: int, reap: bool = True):
        if reap:
            for replica in self._auto_replicas[-num_replicas:]:
                replica.system.kill(replica.env.name)
        self._auto_replicas = self._auto_replicas[:-num_replicas]

    def _add_auto_replicas(self, num_replicas: int):
        self._auto_replicas.extend(self.module.replicate(num_replicas))

    def increment_counter(self):
        self._last_called += 1
        if self._last_called >= len(self.replicas):
            self._last_called = 0
        return self._last_called

    @staticmethod
    def _call_method_on_replica(replica, method, args, kwargs):
        return getattr(replica, method)(*args, **kwargs)

    def map(self, *args, **kwargs):
        """Map the function or method over a list of arguments.

        Example:
            >>> def local_sum(arg1, arg2, arg3):
            >>>     return arg1 + arg2 + arg3
            >>>
            >>> remote_fn = rh.function(local_sum).to(my_cluster)
            >>> mapper = rh.mapper(remote_fn, num_replicas=2)
            >>> mapper.map([1, 2], [1, 4], [2, 3])
            >>> # output: [4, 9]

        """
        ray_wrapped_fn = ray.remote(self._call_method_on_replica)
        kwargs["stream_logs"] = kwargs.get("stream_logs", False)
        return ray.get(
            [
                ray_wrapped_fn.remote(
                    self.replicas[self.increment_counter()], self.method, args, kwargs
                )
                for args in zip(*args)
            ]
        )

    def starmap(self, args_lists: List, **kwargs):
        """Like :func:`map` except that the elements of the iterable are expected to be iterables
        that are unpacked as arguments. An iterable of [(1,2), (3, 4)] results in [func(1,2), func(3,4)].

        Example:
            >>> def local_sum(arg1, arg2, arg3):
            >>>     return arg1 + arg2 + arg3
            >>>
            >>> remote_fn = rh.function(local_sum).to(my_cluster)
            >>> mapper = rh.mapper(remote_fn, num_replicas=2)
            >>> arg_list = [(1,2), (3, 4)]
            >>> # runs the function twice, once with args (1, 2) and once with args (3, 4)
            >>> mapper.starmap(arg_list)
        """
        ray_wrapped_fn = ray.remote(self._call_method_on_replica)
        kwargs["stream_logs"] = kwargs.get("stream_logs", False)
        return ray.get(
            [
                ray_wrapped_fn.remote(
                    self.replicas[self.increment_counter()], self.method, args, kwargs
                )
                for args in args_lists
            ]
        )

    def call(self, *args, **kwargs):
        """Call the function or method on a single replica.

        Example:
            >>> def local_sum(arg1, arg2, arg3):
            >>>     return arg1 + arg2 + arg3
            >>>
            >>> remote_fn = rh.function(local_sum).to(my_cluster)
            >>> mapper = rh.mapper(remote_fn, num_replicas=2)
            >>> for i in range(10):
            >>>     mapper.call(i, 1, 2)
            >>>     # output: 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, run in round-robin replica order

        """
        return getattr(self.replicas[self.increment_counter()], self.method)(
            *args, **kwargs
        )


def mapper(
    module: Module,
    method: Optional[str] = None,
    num_replicas: Optional[int] = -1,
    replicas: Optional[List[Module]] = None,
    **kwargs
):
    """
    A factory method for creating Mapper modules. A mapper is a module that can map a function or module method over
    a list of inputs in various ways.

    If num_replicas is -1, then the number of replicas will be equal to the number of available CPUs (according
    to Ray)), either on the system if one is passed or locally if none is.
    If num_replicas is greater than the number of user-specified replicas, then the remaining replicas will be
    auto-generated by duplicating the module.
    If num_replicas equals 0, it will be left to the number of user-replicas passed into module.
    If it is less than the number of user-specified replicas, then only num_replicas user-specified replicas
    will be used.
    """
    if callable(module) and not isinstance(module, Module):
        module = function(module, **kwargs)

    if isinstance(module, Function):
        method = method or "call"

    return Mapper(module, method, num_replicas, replicas, **kwargs)
