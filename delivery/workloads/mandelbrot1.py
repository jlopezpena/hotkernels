"""NPBench's mandelbrot1, from Nicolas Rougier's "From Python to NumPy": masks over the whole grid.

The NumPy code is in vendor/npbench, copied unchanged. Every iteration
selects and updates the points that have not escaped with `N[I] = n` and
`Z[I] = Z[I]**2 + C[I]`. Both reach the mask hooks and go to NumPy: `Z` is
complex128, which no mask kernel moves, and `N[I] = n` stores into an int64
array, where `x[mask] = v` is served on float64 only.
"""
from __future__ import annotations

from . import Workload
from .vendor.npbench.mandelbrot1_numpy import mandelbrot

# The region of the complex plane and the escape radius, from NPBench's M preset.
REGION = {"xmin": -1.75, "xmax": 0.25, "ymin": -1.0, "ymax": 1.0}
HORIZON = 2.0
# Iterations, from NPBench's M preset.
MAX_ITERATIONS = 150
# Points along each side of the grid. NPBench's own presets are 125, 250 and
# 833, with other regions and iteration counts. These keep its M region and
# iterations and change only the grid, for the times in SIZE_NAMES.
SIZES = {
    "S": {"side": 150},
    "M": {"side": 470},
    "L": {"side": 1500},
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
    return mandelbrot(*arguments)


WORKLOAD = Workload(
    name="mandelbrot1",
    title="NPBench mandelbrot1: escape-time Mandelbrot set, masks over the whole grid",
    source="https://github.com/spcl/npbench, commit f2d7f27, npbench/benchmarks/mandelbrot1",
    licence="BSD-3-Clause (NPBench); code from Nicolas Rougier under the BSD licence",
    ours=False,
    sizes=SIZES,
    make_data=make_data,
    run=run,
)
