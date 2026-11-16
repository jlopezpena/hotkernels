"""Count of non-zero elements: `np.count_nonzero(x)` for float64 data."""
import numpy as np

from judge.spec import In, Kernel, Size


def oracle(x):
    """How many elements of x are not zero. NaN counts; negative zero does not."""
    return np.count_nonzero(x)


KERNEL = Kernel(
    name="count_nonzero_f64",
    returns="usize",
    args=[In("x", "f64"), Size("n")],
    oracle=oracle,
)
