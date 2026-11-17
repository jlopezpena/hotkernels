"""Linear interpolation: `np.interp(x, xp, fp)` for n sample points and any number of queries."""
import numpy as np

from judge.spec import In, Kernel, Out, Scalar, ScalarSet, Size

# One past a power of two, so the block boundary just below it is still tested.
CHECK_MAX_N = 65537
# Queries in the default scalar set: one full group. This set gets the full
# sweep and is timed.
QUERIES = 64
# Queries in the second timed set: several groups and a leftover in both
# paths (1,000 leaves 8 over groups of 16 and 40 over 64).
MANY_QUERIES = 1000
# The kernel handles 16 or 64 queries at once and covers leftovers by redoing
# the last full group, so the counts either side of each width are the ones
# that matter. Set name -> number of queries.
BOUNDARY_QUERIES = {
    "none": 0, "one": 1,
    "k15": 15, "k16": 16, "k17": 17,
    "k63": 63, "k65": 65,
}


def oracle(x, xp, fp, queries):
    """The piecewise-linear interpolant through (xp, fp), evaluated at x."""
    return np.interp(x, xp, fp)


def _sample_points(values):
    """Make an array usable as `xp`: ascending, without NaN.

    NumPy documents interp only for increasing sample points. Repeated values
    and infinities still have one well-defined answer, and are kept. A NaN among
    the sample points does not: the result then depends on which elements the
    search happens to probe, so NaNs are replaced by zero before sorting.
    """
    return np.sort(np.where(np.isnan(values), 0.0, values))


KERNEL = Kernel(
    name="interp_f64",
    returns="void",
    args=[
        # Drawn from `xp` as well as random: a query on a sample point, and
        # on a repeated one, is where the search's comparison decides which
        # value is returned, and random data never gives one.
        In("x", "f64", len="queries", draws_from="xp"),
        In("xp", "f64", prepare=_sample_points),
        In("fp", "f64"),
        Out("out", "f64", capacity="queries"),
        # The number of sample points: the length of xp and of fp. It selects
        # the kernel's code path and the depth of the search, so it is the
        # size that is swept.
        Size("n"),
        Scalar("queries", "usize", QUERIES),
    ],
    oracle=oracle,
    min_n=1,  # np.interp raises when the sample points are empty
    # np.interp takes seconds per call on millions of points, and there are
    # three inputs to vary, so correctness stops at 65,537 sample points.
    check_max_n=CHECK_MAX_N,
    scalar_sets=[ScalarSet(name, {"queries": count}) for name, count in BOUNDARY_QUERIES.items()]
    + [ScalarSet("many", {"queries": MANY_QUERIES}, timed=True)],
)
