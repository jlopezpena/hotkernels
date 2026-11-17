"""Clamp to a range: `np.clip(x, LO, HI)` for float32 data.

The kernel is defined for `lo <= hi` with neither bound NaN. Outside that
NumPy returns `hi` for every non-NaN element (reversed bounds) or NaN (a NaN
bound) and `naive.c` does not, so whatever calls the kernel must send such
calls to NumPy.

The judge passes the same bounds in every call, so it verifies one pair only.
"""
import numpy as np

from judge.spec import In, Kernel, Out, Scalar, Size

LO, HI = -0.5, 0.5


def oracle(x):
    """x with every element below LO raised to LO and every element above HI lowered to HI."""
    return np.clip(x, LO, HI)


KERNEL = Kernel(
    name="clip_f32",
    returns="void",
    args=[In("x", "f32"), Out("out", "f32"), Size("n"), Scalar("lo", "f32", LO), Scalar("hi", "f32", HI)],
    oracle=oracle,
)
