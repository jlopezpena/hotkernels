"""Running count of True: `np.cumsum(mask)` for boolean data, which NumPy accumulates as int64.

Every mask byte is 0 or 1. NumPy counts any non-zero byte of a boolean array
once; a caller that may hold such bytes checks for them before the call.
"""
import numpy as np

from judge.spec import In, Kernel, Out, Size


def oracle(mask):
    """How many elements of mask up to and including each position are True."""
    return np.cumsum(mask)


KERNEL = Kernel(
    name="cumsum_bool",
    returns="void",
    args=[In("mask", "bool"), Out("out", "i64"), Size("n")],
    oracle=oracle,
)
