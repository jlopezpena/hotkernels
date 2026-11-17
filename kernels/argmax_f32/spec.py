"""Index of the maximum: `np.argmax(x)`."""
import numpy as np

from judge.spec import In, Kernel, Size


def oracle(x):
    """Index of the maximum, as NumPy defines it."""
    return np.argmax(x)


KERNEL = Kernel(
    name="argmax_f32",
    returns="i64",
    args=[In("x", "f32"), Size("n")],
    oracle=oracle,
    min_n=1,  # np.argmax raises on empty input; the kernel is never called with n = 0
)
