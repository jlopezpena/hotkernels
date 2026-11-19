"""Selection between two arrays: `np.where(c, a, b)` for a boolean condition and float64 data.

Every byte of `c` is 0 or 1. NumPy takes any non-zero byte of a boolean array
as true; a caller that may hold such bytes checks for them before the call.

The elements are moved and never computed with: each output element has the
bits of the element of `a` or of `b` it was taken from, so a NaN keeps its
payload and a zero its sign.
"""
import numpy as np

from judge.spec import In, Kernel, Out, Size

# Share of the condition's elements that are True in each timing distribution.
HALF_TRUE = 0.5
MOSTLY_TRUE = 0.95


def oracle(c, a, b):
    """The elements of a where c is True and those of b where it is not."""
    return np.where(c, a, b)


def _conditioned(fraction):
    """Return a timing distribution in which the condition is True with a given probability.

    Args:
        fraction: Probability, from 0 to 1, that an element of `c` is True.

    Returns:
        A function `make(rng, n)` that builds the inputs of one timed call.
    """
    def make(rng, n):
        """Build a random condition and two random arrays of n elements.

        Args:
            rng: The NumPy random generator to draw from.
            n: Element count of each array.

        Returns:
            Argument name -> array, for `c`, `a` and `b`.
        """
        return {"c": rng.random(n) < fraction, "a": rng.standard_normal(n), "b": rng.standard_normal(n)}
    return make


KERNEL = Kernel(
    name="where_f64",
    returns="void",
    args=[In("c", "bool"), In("a", "f64"), In("b", "f64"), Out("out", "f64"), Size("n")],
    oracle=oracle,
    distributions={"half_true": _conditioned(HALF_TRUE), "mostly_true": _conditioned(MOSTLY_TRUE)},
)
