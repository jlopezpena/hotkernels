"""Clamp to a range: `np.clip(x, LO, HI)` for float32 data.

The kernel is defined for `lo <= hi` with neither bound NaN. Outside that
NumPy returns `hi` for every non-NaN element (reversed bounds) or NaN (a NaN
bound) and `naive.c` does not, so whatever calls the kernel must send such
calls to NumPy.

The judge checks the pairs of bounds in SCALAR_SETS as well as the default pair,
and times the default pair and `zero_lo`.
"""
import numpy as np

from judge.spec import In, Kernel, Out, Scalar, ScalarSet, Size

LO, HI = -0.5, 0.5

# A denormal magnitude and one near the largest finite value of the type.
TINY, BIG = 1e-40, 3e38

# Further pairs of bounds, all with lo <= hi. The zero bounds matter most:
# NumPy keeps the sign of a zero element that a bound of the other sign would
# otherwise replace, and `np.clip(x, 0, 1)` is probably the commonest call.
SCALAR_SETS = [
    ScalarSet("zero_lo", {"lo": 0.0, "hi": 1.0}, timed=True),
    ScalarSet("neg_zero_lo", {"lo": -0.0, "hi": 1.0}),
    ScalarSet("zero_hi", {"lo": -1.0, "hi": 0.0}),
    ScalarSet("neg_zero_hi", {"lo": -1.0, "hi": -0.0}),
    ScalarSet("both_zero", {"lo": 0.0, "hi": 0.0}),
    ScalarSet("zeros_apart", {"lo": -0.0, "hi": 0.0}),
    ScalarSet("equal", {"lo": 0.25, "hi": 0.25}),
    ScalarSet("infinite", {"lo": float("-inf"), "hi": float("inf")}),
    ScalarSet("half_open", {"lo": 0.5, "hi": float("inf")}),
    ScalarSet("denormal", {"lo": -TINY, "hi": TINY}),
    ScalarSet("wide", {"lo": -BIG, "hi": BIG}),
]


def oracle(x, lo, hi):
    """x with every element below lo raised to lo and every element above hi lowered to hi."""
    return np.clip(x, lo, hi)


KERNEL = Kernel(
    name="clip_f32",
    returns="void",
    args=[In("x", "f32"), Out("out", "f32"), Size("n"), Scalar("lo", "f32", LO), Scalar("hi", "f32", HI)],
    oracle=oracle,
    scalar_sets=SCALAR_SETS,
)
