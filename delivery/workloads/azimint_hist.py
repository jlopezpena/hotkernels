"""NPBench's azimuthal integration by histogram, from pyFAI: two `np.histogram` calls.

The NumPy code is in vendor/npbench, copied unchanged. The unweighted
histogram reaches the histogram kernel; the weighted one goes to NumPy, and
inside it NumPy reaches the mask hooks.
"""
from __future__ import annotations

from . import Workload
from .vendor.npbench.azimint_hist import initialize
from .vendor.npbench.azimint_hist_numpy import azimint_hist

# Number of radial bins. NPBench uses 1,000 at every size; the sizes here
# change only the element count, as in azimint_naive.
BINS = 1000
# NPBench's own presets are S: 400,000, M: 4,000,000 and L: 40,000,000
# elements. These are set for the times in SIZE_NAMES.
SIZES = {
    "S": {"elements": 1_000_000},
    "M": {"elements": 10_000_000},
    "L": {"elements": 100_000_000},
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
    return (azimint_hist(data, radius, npt),)


WORKLOAD = Workload(
    name="azimint_hist",
    title="NPBench azimint_hist: mean intensity per radial bin, by two histograms",
    source="https://github.com/spcl/npbench, commit f2d7f27, npbench/benchmarks/azimint_hist",
    licence="BSD-3-Clause (NPBench); code from pyFAI under CC-BY 3.0",
    ours=False,
    sizes=SIZES,
    make_data=make_data,
    run=run,
)
