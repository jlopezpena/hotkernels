"""Running maximum: `np.maximum.accumulate(x)` for int64 data.

`out[i]` is the largest of `x[0]` to `x[i]`. The comparison is signed. `out`
does not overlap `x`.
"""
import numpy as np

from judge.spec import In, Kernel, Out, Size

# Bounds of the values in the timing distributions: the whole int64 range, so
# that a new maximum is as rare in random data as it can be (about ln(n) of
# them in n elements).
LOWEST = np.iinfo(np.int64).min
HIGHEST = np.iinfo(np.int64).max


def oracle(x):
    """The running maximum of x: element i is the largest of x[0] to x[i]."""
    return np.maximum.accumulate(x)


def _random(rng, n):
    """Build one timing input of uniformly random values.

    The maximum changes about ln(n) times, nearly all of them early.

    Args:
        rng: The judge's NumPy random generator.
        n: Element count.

    Returns:
        The kernel's input arrays by argument name.
    """
    return {"x": rng.integers(LOWEST, HIGHEST, size=n, dtype=np.int64, endpoint=True)}


def _rising(rng, n):
    """Build one timing input sorted ascending, where every element is a new maximum.

    Args:
        rng: The judge's NumPy random generator.
        n: Element count.

    Returns:
        The kernel's input arrays by argument name.
    """
    return {"x": np.sort(_random(rng, n)["x"])}


KERNEL = Kernel(
    name="cummax_i64",
    returns="void",
    args=[In("x", "i64"), Out("out", "i64"), Size("n")],
    oracle=oracle,
    distributions={"random": _random, "rising": _rising},
)
