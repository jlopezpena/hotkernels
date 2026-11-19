"""bottleneck's `nanstd(a, axis=0, ddof=ddof)` on a two-dimensional float64 array whose rows' elements are adjacent.

`a` holds `n` elements. Element (i, j) of the matrix is `a[i * step + j]`,
for `j < cols` and `i < rows`, where `rows` is `(n - cols) // step + 1` when
`n >= cols` and 0 otherwise: as many rows as fit. `step` below `cols` makes the
rows overlap, as in a matrix of rolling windows (`step` 1); `step` equal to
`cols` is a C-contiguous matrix. The kernel is defined for `cols >= 1` and
`step >= 1`.

The kernel writes the standard deviation of each column, skipping NaN, to
`out` and returns `cols`; with no rows it returns 0 and writes nothing. Each
result is bottleneck's bit for bit, in two passes over the column in order: the
sum of the elements that are not NaN from +0.0 and their count; then, when the
count exceeds `ddof`, the mean `sum / count` and, from +0.0, the sum of
`(v - mean) * (v - mean)` over the same elements, giving
`sqrt(sum / (count - ddof))`; NaN (the positive quiet NaN) otherwise. `ddof` is
any int64, and `count - ddof` is formed in int64 and converted to a double.

Whether the square and the add are rounded once or twice is bottleneck's
build's choice: its ARM64 build fuses them, an x86-64 build for the baseline
instruction set cannot. The oracle is bottleneck itself, on the same view, on
the machine the judge runs on, so each architecture's reference is checked
against that architecture's bottleneck. See docs/bottleneck-design.md.
"""
import numpy as np
from numpy.lib.stride_tricks import as_strided

from judge.spec import USED_RETURN, In, Kernel, Out, Scalar, ScalarSet, Size

# Bytes of one element.
ITEMSIZE = 8
# Columns and row step of the default scalar set: a C-contiguous matrix of a
# width that is neither a power of two nor a divisor of a vector width.
DEFAULT_COLS = 37
# Widths checked as C-contiguous matrices (step = cols) and with gaps between
# the rows (step = cols + GAP): the narrowest, and either side of each block
# of columns a kernel may work in (2, 4, 8, 16 and 32 doubles).
CHECKED_COLS = (1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100)
# Elements left between one row's end and the next row's start in the sets
# with gaps.
GAP = 5
# Widths checked as rolling windows (step 1), whose rows overlap.
WINDOW_COLS = (1, 2, 3, 9, 31, 32, 33, 65, 100)
# Degrees of freedom of the default set: empyrical's, the sample deviation.
DEFAULT_DDOF = 1
# Further values of ddof, checked at the default width: the population
# deviation, more than one, a negative one (bottleneck takes any int), and one
# above every count the default sweep makes, which gives NaN everywhere.
CHECKED_DDOF = (0, 2, -1, 1_000_000)
# Widths timed as well: a C-contiguous matrix and rolling windows of 64
# columns, the second the layout of the application this kernel was made for.
TIMED_C_COLS = 64
TIMED_WINDOW_COLS = 64


def scalar_sets():
    """Return the scalar sets: C-contiguous, with gaps, and rolling windows, at each width; then values of ddof."""
    sets = [ScalarSet(f"c_{c}", {"cols": c, "step": c}, timed=c == TIMED_C_COLS) for c in CHECKED_COLS]
    sets += [ScalarSet(f"gap_{c}", {"cols": c, "step": c + GAP}) for c in CHECKED_COLS]
    sets += [ScalarSet(f"win_{c}", {"cols": c, "step": 1}) for c in WINDOW_COLS]
    sets.append(ScalarSet(f"win_{TIMED_WINDOW_COLS}", {"cols": TIMED_WINDOW_COLS, "step": 1}, timed=True))
    sets += [ScalarSet(f"ddof_{d}".replace("-", "minus_"), {"ddof": d}) for d in CHECKED_DDOF]
    return sets


def matrix(a, cols, step):
    """Return the matrix the kernel reduces, as a view of `a`, or None when it has no rows.

    Args:
        a: The n elements.
        cols: Elements per row, at least 1.
        step: Elements from the start of one row to the start of the next, at least 1.

    Returns:
        A (rows, cols) view with strides (step * 8, 8), or None.
    """
    n = len(a)
    if n < cols:
        return None
    rows = (n - cols) // step + 1
    return as_strided(a, shape=(rows, cols), strides=(step * ITEMSIZE, ITEMSIZE), writeable=False)


def oracle(a, cols, step, ddof):
    """bottleneck's standard deviation of each column, skipping NaN.

    Args:
        a: The n elements.
        cols: Elements per row.
        step: Elements from one row's start to the next.
        ddof: Delta degrees of freedom: the divisor is the count less this.

    Returns:
        One float64 per column, or an empty array when there are no rows.
    """
    import bottleneck

    view = matrix(a, cols, step)
    if view is None:
        return np.empty(0)
    return bottleneck.nanstd(view, axis=0, ddof=ddof)


def numpy_call(a, cols, step, ddof):
    """NumPy's own `np.nanstd(view, axis=0, ddof=ddof)`, timed as NumPy: the call a user without bottleneck makes."""
    view = matrix(a, cols, step)
    return np.empty(0) if view is None else np.nanstd(view, axis=0, ddof=ddof)


KERNEL = Kernel(
    name="nanstd_cols_f64",
    returns="usize",  # cols when there is a row, else 0: the results written
    args=[
        In("a", "f64"),
        Out("out", "f64", capacity="cols", used=USED_RETURN),
        Size("n"),
        Scalar("cols", "usize", DEFAULT_COLS),
        Scalar("step", "usize", DEFAULT_COLS),
        Scalar("ddof", "i64", DEFAULT_DDOF),
    ],
    oracle=oracle,
    numpy=numpy_call,
    scalar_sets=scalar_sets(),
)
