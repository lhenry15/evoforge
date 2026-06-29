"""EvoForge public import surface.

This package re-exports the existing ``foundry`` module so users can write:

    import evoforge

while preserving backwards compatibility with:

    import foundry
"""

from __future__ import annotations

import importlib
import sys

from foundry import *  # noqa: F401,F403
from foundry import __all__ as _foundry_all

__all__ = list(_foundry_all)

_ALIASED_SUBMODULES = [
    "bootstrap",
    "collection",
    "context",
    "core",
    "coverage",
    "data",
    "environment",
    "eval",
    "evolution",
    "factory",
    "forecast",
    "intelligence_dashboard",
    "llm",
    "mining",
    "synthesis",
    "trace",
    "training",
    "dashboard",
]

for _submodule in _ALIASED_SUBMODULES:
    sys.modules[f"{__name__}.{_submodule}"] = importlib.import_module(f"foundry.{_submodule}")
