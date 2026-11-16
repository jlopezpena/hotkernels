"""Histogram with equal-width bins: `np.histogram(x, bins=BINS, range=(LO, HI))[0]`.

The judge passes the same bin count and range in every call. They are chosen so
that the bin width is not exactly representable, which exercises NumPy's rule
for values that fall on or near a bin edge.
"""
import numpy as np

from judge.spec import In, Kernel, Out, Scalar, Size

BINS = 50
LO, HI = -3.0, 3.0


def oracle(x):
    """How many elements of x fall into each bin. Elements outside the range, and NaN, are not counted."""
    return np.histogram(x, bins=BINS, range=(LO, HI))[0]


KERNEL = Kernel(
    name="histogram_f64",
    returns="void",
    args=[
        In("x", "f64"),
        Out("counts", "i64", capacity=BINS),
        Size("n"),
        Scalar("nbins", "usize", BINS),
        Scalar("lo", "f64", LO),
        Scalar("hi", "f64", HI),
    ],
    oracle=oracle,
)
