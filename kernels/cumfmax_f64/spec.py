"""Running maximum that ignores NaN: `np.fmax.accumulate(x)` for float64 data.

`out[i]` is the largest of `x[0]` to `x[i]` that is not NaN. `out` does not
overlap `x`. The output is NumPy's bit for bit, which fixes three things a
maximum leaves open:

- **NaN.** A NaN element never changes the running value. While every element
  so far is NaN the output is `x[0]`, sign and payload included; from the
  first element that is not NaN on, the output is never NaN again.
- **Zeros.** `-0.0` and `+0.0` compare equal, and of two equal values the
  later one is the running value. So the sign of a zero output is the sign of
  the last zero among the elements so far that equal the maximum.
- **Two elements.** NumPy computes a single pair with its operands the other
  way round: for `n == 2`, of two zeros the first is `out[1]`, and of two
  NaNs the second.

**No signaling NaN.** Every NaN in `x` is a quiet one, as every NaN that
arithmetic produces is. NumPy's result for a signaling NaN comes from the C
library's `fmax`, which does not ignore it: the running value becomes a quiet
NaN, and the maximum so far is forgotten. A caller that may hold one checks
before the call.
"""
import numpy as np

from judge.spec import In, Kernel, Out, Size

# Share of NaN in the `some_nan` timing distribution.
NAN_SHARE = 0.05


def oracle(x):
    """The running maximum of x over the elements that are not NaN."""
    return np.fmax.accumulate(x)


def _random(rng, n):
    """Build one timing input of random values.

    The maximum changes about ln(n) times, nearly all of them early.

    Args:
        rng: The judge's NumPy random generator.
        n: Element count.

    Returns:
        The kernel's input arrays by argument name.
    """
    return {"x": rng.standard_normal(n)}


def _rising(rng, n):
    """Build one timing input sorted ascending, where every element is a new maximum.

    Args:
        rng: The judge's NumPy random generator.
        n: Element count.

    Returns:
        The kernel's input arrays by argument name.
    """
    return {"x": np.sort(rng.standard_normal(n))}


def _some_nan(rng, n):
    """Build one timing input of random values with NAN_SHARE of them NaN.

    Args:
        rng: The judge's NumPy random generator.
        n: Element count.

    Returns:
        The kernel's input arrays by argument name.
    """
    x = rng.standard_normal(n)
    x[rng.random(n) < NAN_SHARE] = np.nan
    return {"x": x}


KERNEL = Kernel(
    name="cumfmax_f64",
    returns="void",
    args=[In("x", "f64"), Out("out", "f64"), Size("n")],
    oracle=oracle,
    distributions={"random": _random, "rising": _rising, "some_nan": _some_nan},
)
