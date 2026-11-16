"""Linear interpolation: `np.interp(x, xp, fp)` with n sample points and n queries."""
import numpy as np

from judge.spec import In, Kernel, Out, Size

# One past a power of two, so the block boundary just below it is still tested.
CHECK_MAX_N = 65537


def oracle(x, xp, fp):
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
        In("x", "f64"),
        In("xp", "f64", prepare=_sample_points),
        In("fp", "f64"),
        Out("out", "f64"),
        Size("n"),
    ],
    oracle=oracle,
    min_n=1,  # np.interp raises when the sample points are empty
    # np.interp takes seconds per call on millions of points, and there are
    # three inputs to vary, so correctness stops at 65,537 elements.
    check_max_n=CHECK_MAX_N,
)
