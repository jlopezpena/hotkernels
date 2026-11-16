"""Running sum: `np.cumsum(x)` for float64 data."""
import numpy as np

from judge.spec import In, Kernel, Out, Size


def oracle(x):
    """The running sum of x, accumulated left to right in the type of x."""
    return np.cumsum(x)


KERNEL = Kernel(
    name="cumsum_f64",
    returns="void",
    args=[In("x", "f64"), Out("out", "f64"), Size("n")],
    oracle=oracle,
)
