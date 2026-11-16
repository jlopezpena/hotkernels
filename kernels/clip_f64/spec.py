"""Clamp to a range: `np.clip(x, LO, HI)` for float64 data.

The judge passes the same bounds in every call.
"""
import numpy as np

from judge.spec import In, Kernel, Out, Scalar, Size

LO, HI = -0.5, 0.5


def oracle(x):
    """x with every element below LO raised to LO and every element above HI lowered to HI."""
    return np.clip(x, LO, HI)


KERNEL = Kernel(
    name="clip_f64",
    returns="void",
    args=[In("x", "f64"), Out("out", "f64"), Size("n"), Scalar("lo", "f64", LO), Scalar("hi", "f64", HI)],
    oracle=oracle,
)
