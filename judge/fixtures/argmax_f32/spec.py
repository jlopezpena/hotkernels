"""Fixture kernel: tests the judge on a kernel that returns a single value."""
import numpy as np

from judge.spec import In, Kernel, Size


def oracle(x):
    return np.argmax(x)


KERNEL = Kernel(
    name="argmax_f32",
    returns="i64",
    args=[In("x", "f32"), Size("n")],
    oracle=oracle,
    mode="exact",
    min_n=1,  # np.argmax raises on empty input; the kernel is never called with n = 0
)
