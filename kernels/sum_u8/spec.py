"""Sum of bytes: `np.sum(x)` for uint8 data, which NumPy accumulates as uint64."""
import numpy as np

from judge.spec import In, Kernel, Size


def oracle(x):
    """The sum of the elements of x."""
    return np.sum(x)


KERNEL = Kernel(
    name="sum_u8",
    returns="u64",
    args=[In("x", "u8"), Size("n")],
    oracle=oracle,
)
