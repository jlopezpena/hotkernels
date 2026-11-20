"""scikit-image's histogram equalisation of a float64 image.

`skimage.exposure.equalize_hist` takes a histogram, its cumulative sum, and
interpolates every pixel on it: `np.histogram`, `np.cumsum` and `np.interp`.
"""
from __future__ import annotations

import numpy as np

from . import Workload

# Seed of the input image.
SEED = 0
# Number of histogram bins: scikit-image's default.
BINS = 256
# Side of the square image, in pixels, per size.
SIZES = {
    "S": {"side": 512},
    "M": {"side": 1700},
    "L": {"side": 5400},
}


def make_data(side: int) -> tuple:
    """Build a square image of uniform random values in [0, 1).

    Args:
        side: Width and height in pixels.

    Returns:
        A one-element tuple: the image, float64.
    """
    return (np.random.default_rng(SEED).random((side, side)),)


def run(image) -> tuple:
    """Equalise once.

    Args:
        image: A float64 image.

    Returns:
        A one-element tuple: the equalised image.
    """
    from skimage.exposure import equalize_hist

    return (equalize_hist(image, nbins=BINS),)


WORKLOAD = Workload(
    name="equalize_hist",
    title="scikit-image: exposure.equalize_hist on a float64 image",
    source="skimage.exposure.equalize_hist",
    licence=None,
    ours=False,
    sizes=SIZES,
    make_data=make_data,
    run=run,
    requires=("scikit-image",),
)
