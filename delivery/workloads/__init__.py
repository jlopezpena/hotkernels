"""Workloads: programs and library calls not written for hotkernels.

Each module in this package defines one `WORKLOAD`. `uv run delivery
workloads` runs each of them on plain NumPy and after `hotkernels.install()`,
checks that the results agree, and times both. The design is in
docs/workloads-design.md.
"""
from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass, field
from typing import Callable

# Names of the sizes, smallest first. Each is chosen so that plain NumPy takes
# about 10, 100 and 1,000 ms on an Apple M4 Max, where the sizes were set.
SIZE_NAMES = ("S", "M", "L")

# Modules of this package that each define a WORKLOAD, in the order they are run.
MODULES = ("example", "flame", "azimint_naive", "azimint_hist", "mandelbrot1", "mandelbrot2",
           "make_decision", "pandas_filter", "equalize_hist")


@dataclass(frozen=True)
class Workload:
    """One workload and how to run it."""

    # Short name, used by `--only` and in the reports.
    name: str
    # What the workload does, in the words of the code it calls.
    title: str
    # Where the code comes from: a URL and commit, or a library function.
    source: str
    # Licence of copied code, or None for code called in an installed library.
    licence: str | None
    # Whether this project wrote the code. Such a workload is reported apart.
    ours: bool
    # Size name -> keyword arguments of make_data. Not every size need exist.
    sizes: dict[str, dict]
    # Builds the inputs from a fixed seed. Takes one size's keyword arguments.
    make_data: Callable[..., tuple]
    # Runs the workload once on the inputs and returns its results as arrays,
    # scalars and tuples, which delivery.check.describe can compare.
    run: Callable[..., tuple]
    # Distribution names of the libraries the workload needs beyond NumPy.
    requires: tuple[str, ...] = field(default=())

    def missing(self) -> list[str]:
        """Return the libraries this workload needs that are not installed."""
        return [name for name in self.requires if importlib.util.find_spec(_import_name(name)) is None]


# Distribution name -> import name, where they differ.
_IMPORT_NAMES = {"scikit-image": "skimage", "scikit-learn": "sklearn"}


def _import_name(distribution: str) -> str:
    """Return the name a distribution is imported by.

    Args:
        distribution: The name it is installed by, as in pyproject.toml.

    Returns:
        The top-level module name.
    """
    return _IMPORT_NAMES.get(distribution, distribution)


def load() -> list[Workload]:
    """Import every workload module and return their workloads, in run order."""
    return [importlib.import_module(f"{__name__}.{module}").WORKLOAD for module in MODULES]
