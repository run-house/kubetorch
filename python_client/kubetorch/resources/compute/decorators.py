import os
from functools import update_wrapper
from typing import List, Union

from kubetorch.resources.callables.cls.cls import cls
from kubetorch.resources.callables.fn.fn import fn
from kubetorch.resources.compute.compute import Compute

# Helper class which allows us to chain decorators in a way that allows us to reverse the order of the decorators
# The `compute` decorator ultimately unwinds the calls to properly construct the Module.
class PartialModule:
    def __init__(
        self,
        fn_or_cls=None,
        distribute_args=None,
        autoscale_args=None,
        async_=False,
    ):
        self.fn_or_cls = fn_or_cls
        self.distribute_args = distribute_args
        self.autoscale_args = autoscale_args
        self.async_ = async_


# @kubetorch.compute decorator that the user can use to wrap a function they want to deploy to a cluster,
# and then deploy it with `kt deploy my_app.py` (we collect all the decorated functions imported in the file
# to deploy them).
def compute(get_if_exists: bool = False, reload_prefixes: Union[str, List[str]] = [], **kwargs):
    def decorator(func_or_cls):
        from kubetorch.globals import disable_decorators

        if disable_decorators:
            return func_or_cls

        if isinstance(func_or_cls, PartialModule):
            distribute_args = func_or_cls.distribute_args
            autoscale_args = func_or_cls.autoscale_args
            async_ = func_or_cls.async_
            func_or_cls = func_or_cls.fn_or_cls
        else:
            distribute_args = None
            autoscale_args = None
            async_ = False

        # If we're on the server attempting to load this function or class, just return it as is
        if (
            os.environ.get("KT_CLS_OR_FN_NAME") == func_or_cls.__name__
            and os.environ.get("KT_MODULE_NAME") == func_or_cls.__module__
        ):
            return func_or_cls

        module_name = kwargs.pop("name", None)
        kt_deploy_mode = os.environ.get("KT_CLI_DEPLOY_MODE") == "1"

        if isinstance(func_or_cls, type):
            new_module = cls(
                func_or_cls,
                name=module_name,
                get_if_exists=get_if_exists,
                reload_prefixes=reload_prefixes,
            )
        else:
            new_module = fn(
                func_or_cls,
                name=module_name,
                get_if_exists=get_if_exists,
                reload_prefixes=reload_prefixes,
            )

        if async_:
            new_module.async_ = async_

        if kt_deploy_mode:
            # Create new Compute and pass in remaining kwargs only in kt deploy mode, not when importing
            # Imported kt module will be reloaded from name when called
            new_module.compute = Compute(**kwargs)
            new_module.compute.service_name = new_module.service_name
            if distribute_args:
                distribute_args, distribute_kwargs = distribute_args
                new_module.compute.distribute(*distribute_args, **distribute_kwargs)
            if autoscale_args:
                autoscale_args, autoscale_kwargs = autoscale_args
                new_module.compute.autoscale(*autoscale_args, **autoscale_kwargs)

        # update_wrapper(new_module, func_or_cls)
        return new_module

    return decorator


def async_(func_or_cls):
    from kubetorch.globals import disable_decorators

    if disable_decorators:
        return func_or_cls

    # If it's already a PartialModule (from other decorators), update it
    if isinstance(func_or_cls, PartialModule):
        func_or_cls.async_ = True
        return func_or_cls

    # Otherwise, create a new PartialModule
    partial_module = PartialModule(fn_or_cls=func_or_cls, async_=True)
    update_wrapper(partial_module, func_or_cls)
    return partial_module


def distribute(*args, **kwargs):
    def decorator(func_or_cls):
        from kubetorch.globals import disable_decorators

        if disable_decorators:
            return func_or_cls

        # This is a partial so the order of decorator chaining can be reversed for best aesthetics
        # the deploy method will actually call .distribute on the function or class after it's been deployed
        partial_module = PartialModule(fn_or_cls=func_or_cls, distribute_args=(args, kwargs))
        update_wrapper(partial_module, func_or_cls)
        return partial_module

    return decorator


def autoscale(*args, **kwargs):
    def decorator(func_or_cls):
        from kubetorch.globals import disable_decorators

        if disable_decorators:
            return func_or_cls

        # This is a partial so the order of decorator chaining can be reversed for best aesthetics
        # the deploy method will actually call .distribute on the function or class after it's been deployed
        partial_module = PartialModule(fn_or_cls=func_or_cls, autoscale_args=(args, kwargs))
        update_wrapper(partial_module, func_or_cls)
        return partial_module

    return decorator
