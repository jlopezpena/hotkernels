"""Runs three routines of the Veros ocean model, as pyhpc-benchmarks publishes them.

pyhpc-benchmarks takes routines from Veros, an ocean model written in NumPy,
and gives each an input generator and one implementation per array library.
The NumPy implementations and the generators of three of them are copied
unchanged in `vendor/pyhpc/`:

    equation_of_state         gsw_dHdT, a polynomial of salinity, temperature
                              and pressure on the whole grid
    isoneutral_mixing         isoneutral_diffusion_pre, slopes and mixing
                              tensors from finite differences over slices
    turbulent_kinetic_energy  integrate_tke, an implicit vertical step solved
                              by LAPACK's tridiagonal solver, then advection
                              with a superbee limiter

Each routine is run once per run of this application, in that order. Two of
them write into their inputs, so each run starts from copies of the generated
arrays; the copies are a few percent of a run and the same on both sides.
"""
from __future__ import annotations

import numpy as np

TITLE = "Three routines of the Veros ocean model: equation of state, isoneutral mixing, turbulent kinetic energy"
SOURCE = "https://github.com/dionhaefner/pyhpc-benchmarks at d438ef8, NumPy implementations copied unchanged"
LICENCE = "Unlicense"
REQUIRES = ("scipy",)

# The copied package: delivery/apps/vendor/pyhpc.
VENDOR = "delivery.apps.vendor.pyhpc"
# The routines run, as subpackages of VENDOR, in the order they run.
ROUTINES = ("equation_of_state", "isoneutral_mixing", "turbulent_kinetic_energy")
# The module of each routine's NumPy implementation, in its subpackage.
IMPLEMENTATION = "numpy"
# By size: the number of grid points pyhpc-benchmarks' generators are asked for.
# They build a grid of 2n^(1/3) by 2n^(1/3) by n^(1/3)/4 points from it.
GRID_POINTS = {"small": 65_536, "large": 1_048_576}


def _shape(points: int) -> tuple[int, int, int]:
    """Return the grid pyhpc-benchmarks' generators build for a number of points.

    Args:
        points: The size passed to the generators.

    Returns:
        The grid's three dimensions, as `generate_inputs` computes them.
    """
    side = points ** (1 / 3)
    return (int(np.ceil(2 * side)), int(np.ceil(2 * side)), int(np.ceil(0.25 * side)))


# One line per size saying how large its data is.
SIZES = {size: "a grid of {:,} × {:,} × {:,} points".format(*_shape(points)) for size, points in GRID_POINTS.items()}


def make_data(size: str) -> tuple:
    """Build each routine's inputs with its own generator.

    Args:
        size: One of SIZES.

    Returns:
        One pair per routine of ROUTINES: (its `run` function, its inputs as a
        tuple of arrays).
    """
    import importlib

    pairs = []
    for routine in ROUTINES:
        package = importlib.import_module(f"{VENDOR}.{routine}")
        module = package.try_import(IMPLEMENTATION)
        pairs.append((module.run, package.generate_inputs(GRID_POINTS[size])))
    return tuple(pairs)


def run(*pairs: tuple) -> tuple:
    """Run each routine once on fresh copies of its inputs.

    Args:
        pairs: What make_data returns: (run function, inputs) per routine.

    Returns:
        Every routine's outputs, in the order of ROUTINES.
    """
    results = []
    for routine, inputs in pairs:
        # The routines write into some of their inputs: isoneutral mixing
        # fills its four tensors and the kinetic energy step its tke arrays.
        fresh = [np.copy(array) for array in inputs]
        results.append(routine(*fresh))
    return tuple(results)
