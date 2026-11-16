"""Index of the minimum: `np.argmin(x)`."""
import numpy as np

from judge.spec import In, Kernel, Size


def oracle(x):
    """Index of the minimum, as NumPy defines it."""
    return np.argmin(x)


KERNEL = Kernel(
    name="argmin_f64",
    returns="i64",
    args=[In("x", "f64"), Size("n")],
    oracle=oracle,
    min_n=1,  # np.argmin raises on empty input; the kernel is never called with n = 0
)
