"""Deferred imports for heavy dependencies that are only needed inside functions.

colour-science 0.4.x imports ``colour.plotting`` (and with it matplotlib) from its package
``__init__``, which costs about two seconds. Modules that only use colour inside functions
import the proxy below instead, so ``import ColorCorrectionPipeline`` and the bundle-apply
path do not pay for it. The first attribute access performs a normal ``import colour``.

A forwarding proxy is used instead of ``importlib.util.LazyLoader`` because colour replaces
its own ``sys.modules`` entry while importing, which LazyLoader rejects.
"""

import importlib
import threading
from types import ModuleType
from typing import Any, Optional

__all__ = ["LazyModule", "colour", "pandas"]


class LazyModule:
    """Forward attribute access to a module that is imported on first use."""

    __slots__ = ("_lazy_name", "_lazy_module", "_lazy_lock")

    def __init__(self, name: str) -> None:
        object.__setattr__(self, "_lazy_name", name)
        object.__setattr__(self, "_lazy_module", None)
        object.__setattr__(self, "_lazy_lock", threading.Lock())

    def _load(self) -> ModuleType:
        module: Optional[ModuleType] = object.__getattribute__(self, "_lazy_module")
        if module is None:
            with object.__getattribute__(self, "_lazy_lock"):
                module = object.__getattribute__(self, "_lazy_module")
                if module is None:
                    module = importlib.import_module(object.__getattribute__(self, "_lazy_name"))
                    object.__setattr__(self, "_lazy_module", module)
        return module

    def __getattr__(self, attribute: str) -> Any:
        return getattr(self._load(), attribute)

    def __setattr__(self, attribute: str, value: Any) -> None:
        setattr(self._load(), attribute, value)

    def __dir__(self):
        return dir(self._load())

    def __repr__(self) -> str:
        state = (
            "loaded" if object.__getattribute__(self, "_lazy_module") is not None else "not loaded"
        )
        return "<lazy module %r (%s)>" % (object.__getattribute__(self, "_lazy_name"), state)


colour = LazyModule("colour")
pandas = LazyModule("pandas")
