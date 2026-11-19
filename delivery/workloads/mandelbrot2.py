"""NPBench's mandelbrot2, from Nicolas Rougier's "From Python to NumPy": escaped points are dropped.

The NumPy code is in vendor/npbench, copied unchanged. Every iteration keeps
the points that have not escaped with `Z[I]`, `Xi[I]`, `Yi[I]` and `C[I]`.
The int64 index arrays `Xi` and `Yi` reach the mask selection kernel; the
complex128 `Z` and `C` go to NumPy.
"""
from __future__ import annotations

import warnings

from . import Workload
from .vendor.npbench.mandelbrot2_numpy import mandelbrot

# The region of the complex plane and the escape radius, from NPBench's M preset.
REGION = {"xmin": -2.0, "xmax": 0.5, "ymin": -1.25, "ymax": 1.25}
HORIZON = 2.0
# Iterations, from NPBench's M preset.
MAX_ITERATIONS = 80
# Points along each side of the grid. NPBench's own presets are 200, 500 and
# 1,000, with other regions and iteration counts. These keep its M region and
# iterations and change only the grid, for the times in SIZE_NAMES.
SIZES = {
    "S": {"side": 300},
    "M": {"side": 950},
    "L": {"side": 3000},
}


def make_data(side: int) -> tuple:
    """Build NPBench's arguments: the program makes its own arrays.

    Args:
        side: Points along each side of the square grid.

    Returns:
        The positional arguments of `mandelbrot`.
    """
    return (REGION["xmin"], REGION["xmax"], REGION["ymin"], REGION["ymax"],
            side, side, MAX_ITERATIONS, HORIZON)


def run(*arguments) -> tuple:
    """Compute the set once.

    Args:
        arguments: The arguments make_data returned.

    Returns:
        (final values of z, escape iteration of each point), as NPBench returns them.
    """
    # The program reshapes arrays by setting `.shape`, which NumPy 2.5
    # deprecates. The warning would print on every run; the result is the same.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return mandelbrot(*arguments)


WORKLOAD = Workload(
    name="mandelbrot2",
    title="NPBench mandelbrot2: escape-time Mandelbrot set, escaped points dropped",
    source="https://github.com/spcl/npbench, commit f2d7f27, npbench/benchmarks/mandelbrot2",
    licence="BSD-3-Clause (NPBench); code from Nicolas Rougier under the BSD licence",
    ours=False,
    sizes=SIZES,
    make_data=make_data,
    run=run,
)
