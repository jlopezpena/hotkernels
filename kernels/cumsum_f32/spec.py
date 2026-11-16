"""Running sum: `np.cumsum(x)` for float32 data."""
import numpy as np

from judge.spec import In, Kernel, Out, Size


def oracle(x):
    """The running sum of x, accumulated left to right in the type of x."""
    return np.cumsum(x)


KERNEL = Kernel(
    name="cumsum_f32",
    returns="void",
    args=[In("x", "f32"), Out("out", "f32"), Size("n")],
    oracle=oracle,
)
