from kubetorch.logger import get_logger
from kubetorch.resources.callables.module import Module
from kubetorch.resources.callables.utils import extract_pointers

logger = get_logger(__name__)


class Fn(Module):
    MODULE_TYPE = "fn"

    def __init__(
        self,
        name: str,
        pointers: tuple = None,
    ):
        """
        Initialize a Fn object for remote function execution.

        .. note::

            To create a Function, please use the factory method :func:`fn`.

        Args:
            name (str): The name of the function to be executed remotely.
            pointers (tuple, optional): A tuple containing pointers/references needed to locate and execute
                the function, typically including module path, class name (if applicable), and
                function name.
        """
        super().__init__(name=name, pointers=pointers)

    def __call__(self, *args, **kwargs):
        async_ = kwargs.pop("async_", self.async_)

        if async_:
            return self._call_async(*args, **kwargs)
        else:
            return self._call_sync(*args, **kwargs)

    def _call_sync(self, *args, **kwargs):
        client = self._client()
        stream_logs = kwargs.pop("stream_logs", None)
        stream_metrics = kwargs.pop("stream_metrics", None)
        pdb = kwargs.pop("pdb", None)
        if pdb:
            logger.info(f"Debugging remote function {self.name}")
        elif stream_logs:
            logger.info(f"Calling remote function {self.name}")

        response = client.call_method(
            self.endpoint(),
            stream_logs=stream_logs,
            stream_metrics=stream_metrics,
            headers=self.request_headers,
            body={"args": list(args), "kwargs": kwargs},
            pdb=pdb,
            serialization=kwargs.pop("serialization", self.serialization),
        )
        return response

    async def _call_async(self, *args, **kwargs):
        """Asynchronous call implementation."""
        client = self._client()
        stream_logs = kwargs.pop("stream_logs", None)
        stream_metrics = kwargs.pop("stream_metrics", None)
        pdb = kwargs.pop("pdb", None)
        if pdb:
            logger.info(f"Debugging remote function {self.name}")
        elif stream_logs:
            logger.info(f"Calling remote function {self.name}")

        response = await client.call_method_async(
            self.endpoint(),
            stream_logs=stream_logs,
            stream_metrics=stream_metrics,
            headers=self.request_headers,
            body={"args": list(args), "kwargs": kwargs},
            pdb=pdb,
            serialization=kwargs.pop("serialization", self.serialization),
        )
        return response


def fn(function_obj=None, name: str = None, get_if_exists=True, reload_prefixes=None):
    """
    Builds an instance of :class:`Fn`.

    Args:
        function_obj (Fn, optional): The function to be executed remotely. If not provided and name is
            specified, will reload an existing fn object.
        name (str, optional): The name to give the remote function. If not provided,
            will use the function's name.
        get_if_exists (bool, optional):
            Controls how service lookup is performed when loading by name.

            - If True (default): Attempt to find an existing service using a standard fallback order
              (e.g., username, git branch, then prod).
            - If False: Only look for an exact name match; do not attempt any fallback.

            This allows you to control whether and how the loader should fall back to alternate
            versions of a service (such as QA, prod, or CI versions) if the exact name is not found.
        reload_prefixes (Union[str, List[str]], optional):
            A list of prefixes to use when reloading the function (e.g., ["qa", "prod", "git-branch-name"]).
            If not provided, will use the current username, git branch, and prod.

    Example:

    .. code-block:: python

        import kubetorch as kt

        remote_fn = kt.fn(my_func, name="some-func").to(kt.Compute(cpus=".1"))
        result = remote_fn(1, 2)
    """
    if function_obj:
        fn_pointers = extract_pointers(function_obj)
        name = name or (fn_pointers[2] if fn_pointers else function_obj.__name__)
        new_fn = Fn(
            name=name,
            pointers=fn_pointers,
        )
        new_fn.get_if_exists = get_if_exists
        new_fn.reload_prefixes = reload_prefixes or []
        return new_fn

    if name is None:
        raise ValueError("Name must be provided to reload an existing function")

    if get_if_exists is False:
        raise ValueError(
            "Either provide a function object or a name with get_if_exists=True to reload an existing function"
        )

    reloaded_fn = Fn.from_name(name, reload_prefixes=reload_prefixes)
    return reloaded_fn


FN_METHODS = dir(Fn)
