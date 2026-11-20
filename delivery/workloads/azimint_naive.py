"""NPBench's naive azimuthal integration, from pyFAI: one boolean mask per radial bin.

The NumPy code is in vendor/npbench, copied unchanged. Each of `npt` bins
selects its elements with `data[mask]`, which reaches the mask selection
kernel.
"""
from __future__ import annotations

from . import Workload
from .vendor.npbench.azimint_naive import initialize
from .vendor.npbench.azimint_naive_numpy import azimint_naive

# Number of radial bins. NPBench uses 1,000 at every size; the sizes here
# change only the element count, so that the time per size is about right.
BINS = 1000
# NPBench's own presets are S: 400,000, M: 4,000,000 and L: 40,000,000
# elements, set on a two-socket Xeon. These are set for the times in SIZE_NAMES.
SIZES = {
    "S": {"elements": 20_000},
    "M": {"elements": 200_000},
    "L": {"elements": 2_000_000},
}


def make_data(elements: int) -> tuple:
    """Build NPBench's inputs.

    Args:
        elements: Number of pixels: the length of `data` and `radius`.

    Returns:
        (data, radius, number of bins), as NPBench passes them.
    """
    data, radius = initialize(elements)
    return data, radius, BINS


def run(data, radius, npt) -> tuple:
    """Integrate once.

    Args:
        data: Intensity of each pixel, float64.
        radius: Radius of each pixel, float64.
        npt: Number of radial bins.

    Returns:
        A one-element tuple: the mean intensity per bin.
    """
    return (azimint_naive(data, radius, npt),)


WORKLOAD = Workload(
    name="azimint_naive",
    title="NPBench azimint_naive: mean intensity per radial bin, one mask per bin",
    source="https://github.com/spcl/npbench, commit f2d7f27, npbench/benchmarks/azimint_naive",
    licence="BSD-3-Clause (NPBench); code from pyFAI under CC-BY 3.0",
    ours=False,
    sizes=SIZES,
    make_data=make_data,
    run=run,
)
