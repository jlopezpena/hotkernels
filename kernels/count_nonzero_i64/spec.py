"""Count of non-zero elements: `np.count_nonzero(x)` for int64 data."""
import numpy as np

from judge.spec import In, Kernel, Size


def oracle(x):
    """How many elements of x are not zero. NaN counts; negative zero does not."""
    return np.count_nonzero(x)


KERNEL = Kernel(
    name="count_nonzero_i64",
    returns="usize",
    args=[In("x", "i64"), Size("n")],
    oracle=oracle,
)
