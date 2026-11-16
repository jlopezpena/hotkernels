"""Clamp to a range: `np.clip(x, LO, HI)` for float32 data.

The judge passes the same bounds in every call.
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
