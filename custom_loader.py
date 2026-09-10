"""Public custom-runtime loader; keep Hermes' integration surface to this module."""
from __future__ import annotations

from functools import lru_cache
import importlib

from custom_runtime import Runtime


@lru_cache(maxsize=16)
def _runtime(name: str):
    if name == "job-finder":
        return importlib.import_module("custom.job-finder.runtime").JobFinderRuntime()
    return None


def get_runtime(job: dict) -> Runtime | None:
    """Return an operator-selected runtime; unknown selectors fail closed."""
    name = job.get("custom_runtime")
    if not isinstance(name, str) or not name:
        return None
    runtime = _runtime(name)
    if runtime is None:
        raise ValueError(f"Unknown custom runtime: {name}")
    return runtime
