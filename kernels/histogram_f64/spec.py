"""Histogram with equal-width bins: `np.histogram(x, bins=BINS, range=(LO, HI))[0]`.

The default bin count and range are chosen so that the bin width is not exactly
representable, which exercises NumPy's rule for values that fall on or near a
bin edge. SCALAR_SETS adds other bin counts and ranges, which the judge checks
but does not time.

The kernel is defined for a finite range with lo < hi and at least one bin.
"""
import numpy as np

from judge.spec import In, Kernel, Out, Scalar, ScalarSet, Size

BINS = 50
LO, HI = -3.0, 3.0

# The largest finite float64, and the width of the two ranges that end at it.
# With 1,000 bins, EDGE_WIDTH is narrow enough against FLOAT_MAX that an
# estimated bin can be off by one, and wide enough that width * 2**39 overflows.
FLOAT_MAX = float(np.finfo(np.float64).max)
EDGE_WIDTH = 1e298

# Further bin counts and ranges. The judge's data is mostly within a few units
# of zero, so every range but the last two overlaps that; a range far from the
# data counts nothing and tests little.
SCALAR_SETS = [
    # The smallest tables: every in-range element lands in the last bin, or one of two.
    ScalarSet("one_bin", {"nbins": 1}),
    ScalarSet("two_bins", {"nbins": 2, "lo": 0.0, "hi": 1.0}),
    # Bin widths that are exact in binary, so many elements sit exactly on an edge.
    ScalarSet("exact_width", {"nbins": 64, "lo": -4.0, "hi": 4.0}),
    ScalarSet("half_edges", {"nbins": 12}),
    # Widths that are not representable, of several sizes.
    ScalarSet("thirds", {"nbins": 3, "lo": -1.0, "hi": 1.0}),
    ScalarSet("narrow", {"nbins": 10, "lo": 0.1, "hi": 0.7}),
    ScalarSet("lopsided", {"nbins": 7, "lo": -0.1, "hi": 2.9}),
    ScalarSet("prime_bins", {"nbins": 997, "lo": -2.5, "hi": 3.5}),
    # More bins than the default by a wide margin: the table no longer fits a few cache lines.
    ScalarSet("many_bins", {"nbins": 4096}),
    # Ranges whose width is far from 1, where the scaling loses or gains many bits.
    ScalarSet("tiny_range", {"nbins": 20, "lo": -1e-3, "hi": 1e-3}),
    ScalarSet("huge_range", {"nbins": 33, "lo": -1e300, "hi": 1e300}),
    ScalarSet("denormal_range", {"nbins": 5, "lo": -1e-310, "hi": 1e-310}),
    # A range that does not contain zero, so lo is not small against the width.
    ScalarSet("offset", {"nbins": 40, "lo": 0.25, "hi": 1.75}),
    # Narrow ranges at the two ends of float64, where max(|lo|, |hi|) * nbins
    # overflows. A kernel that decides how to count from such a product has to
    # notice the overflow. The only values the judge generates inside these
    # ranges are the largest magnitudes themselves, which sit on the outer
    # edge, so these sets check that nothing else is counted and that the
    # outer bin is right. They do not check the bins in between: that needs
    # data placed inside the range.
    ScalarSet("top_of_range", {"nbins": 1000, "lo": FLOAT_MAX - EDGE_WIDTH, "hi": FLOAT_MAX}),
    ScalarSet("bottom_of_range", {"nbins": 1000, "lo": -FLOAT_MAX, "hi": -FLOAT_MAX + EDGE_WIDTH}),
]


def oracle(x, nbins, lo, hi):
    """How many elements of x fall into each bin. Elements outside the range, and NaN, are not counted."""
    return np.histogram(x, bins=nbins, range=(lo, hi))[0]


KERNEL = Kernel(
    name="histogram_f64",
    returns="void",
    args=[
        In("x", "f64"),
        Out("counts", "i64", capacity="nbins"),
        Size("n"),
        Scalar("nbins", "usize", BINS),
        Scalar("lo", "f64", LO),
        Scalar("hi", "f64", HI),
    ],
    oracle=oracle,
    scalar_sets=SCALAR_SETS,
)
