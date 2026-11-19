"""The applications `delivery apps` runs with and without hotkernels.

Each application is a module of this package with these names:

    TITLE      One line saying what the program does.
    SOURCE     Where its code comes from.
    LICENCE    The licence of copied code, or None for code written here or
               a call into an installed library.
    REQUIRES   Names of the modules it imports beyond NumPy.
    SIZE       One line saying how large its data is.
    make_data  Builds its inputs from a fixed seed; returns a tuple.
    run        Takes the inputs, runs the program once, returns its results.

See docs/applications-design.md.
"""
from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType

# Every application, in the order the report lists them.
NAMES = ("mlp_train", "min_char_rnn", "trimesh_mesh", "sklearn_metrics", "empyrical_stats")


def load(name: str) -> ModuleType:
    """Import one application's module.

    Args:
        name: One of NAMES.

    Returns:
        The module.
    """
    return importlib.import_module(f"{__name__}.{name}")


def missing(name: str) -> list[str]:
    """Say which libraries an application needs that are not installed.

    Args:
        name: One of NAMES.

    Returns:
        The names of the modules that cannot be imported; empty when the
        application can run.
    """
    return [module for module in load(name).REQUIRES if importlib.util.find_spec(module) is None]
