import os
from contextlib import nullcontext
from functools import wraps

import torch


_ENABLED = os.environ.get("MILES_COMPONENT_PROFILE") == "1"


def component_profile(name: str):
    """Create a named PyTorch-profiler range when component profiling is enabled."""
    if not _ENABLED:
        return nullcontext()
    return torch.profiler.record_function(name)


def profile_function(name: str):
    """Add an opt-in profiler range around a function or method."""

    def decorator(func):
        if not _ENABLED:
            return func

        @wraps(func)
        def wrapped(*args, **kwargs):
            with component_profile(name):
                return func(*args, **kwargs)

        return wrapped

    return decorator


def profile_each_call(name: str):
    """Add an incrementing profiler range around each call to a function."""

    def decorator(func):
        if not _ENABLED:
            return func

        call_index = 0

        @wraps(func)
        def wrapped(*args, **kwargs):
            nonlocal call_index
            current_index = call_index
            call_index += 1
            with component_profile(f"{name}.{current_index}"):
                return func(*args, **kwargs)

        return wrapped

    return decorator


def profile_method(cls, method_name: str, name: str) -> None:
    """Add an opt-in profiler range around a class method once."""
    if not _ENABLED:
        return

    method = getattr(cls, method_name)
    if getattr(method, "_miles_component_profiled", False):
        return

    wrapped = profile_function(name)(method)
    wrapped._miles_component_profiled = True
    setattr(cls, method_name, wrapped)


def profile_module_forward(module: torch.nn.Module, name: str) -> None:
    """Add an opt-in profiler range around one module instance's forward call."""
    if not _ENABLED or getattr(module, "_miles_component_profiled", False):
        return

    module.forward = profile_function(name)(module.forward)
    module._miles_component_profiled = True
