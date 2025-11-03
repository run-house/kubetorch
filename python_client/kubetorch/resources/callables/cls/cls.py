from kubetorch.logger import get_logger
from kubetorch.resources.callables.module import Module
from kubetorch.resources.callables.utils import extract_pointers, SHELL_COMMANDS

logger = get_logger(__name__)


class Cls(Module):
    MODULE_TYPE = "cls"

    def __init__(
        self,
        name: str,
        pointers: tuple = None,
        init_args: dict = None,
    ):
        """
        Initialize a Cls object for remote class execution.

        .. note::

            To create a Cls, please use the factory method :func:`cls`.

        Args:
            name (str): The name of the class to be executed remotely.
            pointers (tuple, optional): A tuple containing pointers/references needed to locate and execute
                the class, typically including module path and class name.
            init_args (dict, optional): Dictionary of arguments to pass to the class constructor.
                Defaults to None.
        """
        self._init_args = init_args
        if not pointers:
            # local to the class definition
            pointers = extract_pointers(self.__class__)

        super().__init__(name=name, pointers=pointers)

    def __getattr__(self, attr_name):
        if attr_name in SHELL_COMMANDS:
            return getattr(self.compute, attr_name)

        if not attr_name.startswith("_") and attr_name not in CLASS_METHODS:

            def remote_method_wrapper(*args, **kwargs):
                async_ = kwargs.pop("async_", self.async_)

                if async_:
                    return self._call_async(attr_name, *args, **kwargs)
                else:
                    return self._call_sync(attr_name, *args, **kwargs)

            return remote_method_wrapper

    @property
    def init_args(self):
        return self._init_args

    @init_args.setter
    def init_args(self, value):
        self._init_args = value

    def _call_sync(self, method_name, *args, **kwargs):
        """Synchronous call implementation."""
        client = self._client(method_name=method_name)
        stream_logs = kwargs.pop("stream_logs", None)
        pdb = kwargs.pop("pdb", None)
        if pdb:
            logger.info(f"Debugging remote cls {self.name}.{method_name}")
        elif stream_logs:
            logger.info(f"Calling remote cls {self.name}.{method_name}")

        response = client.call_method(
            self.endpoint(method_name),
            stream_logs=stream_logs,
            headers=self.request_headers,
            body={"args": list(args), "kwargs": kwargs},
            pdb=pdb,
            serialization=kwargs.pop("serialization", self.serialization),
        )
        return response

    async def _call_async(self, method_name, *args, **kwargs):
        """Asynchronous call implementation."""
        client = self._client(method_name=method_name)
        stream_logs = kwargs.pop("stream_logs", None)
        pdb = kwargs.pop("pdb", None)
        if pdb:
            logger.info(f"Debugging remote cls {self.name}.{method_name} (async)")
        elif stream_logs:
            logger.info(f"Calling remote cls {self.name}.{method_name} (async)")

        response = await client.call_method_async(
            self.endpoint(method_name),
            stream_logs=stream_logs,
            headers=self.request_headers,
            body={"args": list(args), "kwargs": kwargs},
            pdb=pdb,
            serialization=kwargs.pop("serialization", self.serialization),
        )
        return response


def cls(class_obj=None, name: str = None, get_if_exists=True, reload_prefixes=None):
    """
    Builds an instance of :class:`Cls`.

    Args:
        class_obj (Cls, optional): The class to be executed remotely. If not provided and name is
            specified, will reload an existing cls object.
        name (str, optional): The name to give the remote class. If not provided,
            will use the class's name.
        get_if_exists (bool, optional):
            Controls how service lookup is performed when loading by name.

            - If True (default): Attempt to find an existing service using a standard fallback order
              (e.g., username, git branch, then prod).
            - If False: Only look for an exact name match; do not attempt any fallback.

            This allows you to control whether and how the loader should fall back to alternate
            versions of a service (such as QA, prod, or CI versions) if the exact name is not found.
        reload_prefixes (Union[str, List[str]], optional):
            A list of prefixes to use when reloading the class (e.g., ["qa", "prod", "git-branch-name"]).
            If not provided, will use the current username, git branch, and prod.

    Example:

    .. code-block:: python

        import kubetorch as kt

        remote_cls = kt.cls(MyClass, name="my-class").to(kt.Compute(cpus=".1"))
        result = remote_cls.my_method(1, 2)
    """
    if class_obj:
        cls_pointers = extract_pointers(class_obj)
        name = name or (cls_pointers[2] if cls_pointers else cls.__name__)
        new_cls = Cls(
            name=name,
            pointers=cls_pointers,
        )
        new_cls.get_if_exists = get_if_exists
        new_cls.reload_prefixes = reload_prefixes or []
        return new_cls

    if name is None:
        raise ValueError("Name must be provided to reload an existing class")

    if get_if_exists is False:
        raise ValueError(
            "Either provide a class object or a name with get_if_exists=True to reload an existing class"
        )

    reloaded_cls = Cls.from_name(name, reload_prefixes=reload_prefixes)
    return reloaded_cls


CLASS_METHODS = dir(Cls)
