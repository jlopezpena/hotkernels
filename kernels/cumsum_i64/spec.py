"""Running sum: `np.cumsum(x)` for int64 data."""
import numpy as np

from judge.spec import In, Kernel, Out, Size


def oracle(x):
    """The running sum of x, accumulated left to right in the type of x."""
    return np.cumsum(x)


KERNEL = Kernel(
    name="cumsum_i64",
    returns="void",
    args=[In("x", "i64"), Out("out", "i64"), Size("n")],
    oracle=oracle,
)
