"""examples/workload.py, the program `delivery demo` runs.

It was written by this project to use eleven hooked calls, so it is reported
apart from the others. It has one size: the element count it was written with.
"""
from __future__ import annotations

import importlib.util
from functools import cache
from pathlib import Path
from types import ModuleType

from . import Workload

PROGRAM = Path(__file__).resolve().parents[2] / "examples" / "workload.py"
# The program fixes its own element count, two million, which puts it in this size.
SIZES = {"S": {}}


@cache
def _program() -> ModuleType:
    """Load the program from its file, once, as Python would run it."""
    spec = importlib.util.spec_from_file_location("workload", PROGRAM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_data() -> tuple:
    """Build the program's inputs with its own function and seed."""
    return _program().make_data()


def run(*data) -> tuple:
    """Run the program's analysis once.

    Args:
        data: The inputs make_data returned.

    Returns:
        Every result, ordered by name.
    """
    results = _program().analyse(*data)
    return tuple(results[name] for name in sorted(results))


WORKLOAD = Workload(
    name="example",
    title="examples/workload.py: clean a signal, summarise an image, look up ids",
    source="examples/workload.py in this repository",
    licence=None,
    ours=True,
    sizes=SIZES,
    make_data=make_data,
    run=run,
)
