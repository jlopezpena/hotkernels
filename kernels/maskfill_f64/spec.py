"""Masked assignment of a scalar: `x[mask] = VALUE`, in place.

The judge passes the same value in every call.
"""
from judge.spec import In, InOut, Kernel, Scalar, Size

VALUE = 0.0

# Share of mask elements that are True in each timing distribution.
HALF_SELECTED = 0.5
FEW_SELECTED = 0.05


def oracle(x, mask):
    """A copy of x with VALUE written wherever mask is True."""
    result = x.copy()
    result[mask] = VALUE
    return result


def assign(x, mask):
    """The statement a user writes, timed as NumPy. Applying it twice changes nothing."""
    x[mask] = VALUE


def _masked(fraction):
    """Return a timing distribution in which `fraction` of the mask is True."""
    def make(rng, n):
        """Build random data and a random mask for one timing input."""
        return {"x": rng.standard_normal(n), "mask": rng.random(n) < fraction}
    return make


KERNEL = Kernel(
    name="maskfill_f64",
    returns="void",
    args=[InOut("x", "f64"), In("mask", "bool"), Size("n"), Scalar("value", "f64", VALUE)],
    oracle=oracle,
    numpy=assign,
    distributions={"half_selected": _masked(HALF_SELECTED), "few_selected": _masked(FEW_SELECTED)},
)
