"""Running maximum that ignores NaN: `np.fmax.accumulate(x)` for float64 data.

`out[i]` is the largest of `x[0]` to `x[i]` that is not NaN. `out` does not
overlap `x`.

**The domain** is what the binding passes to the kernel; it sends every other
array to NumPy (`fmax_accumulate_left_to_numpy` in
`hotkernels/src/hotkernels/_binding/hook_method.c`):

- **No negative zero.** No element of `x` is `-0.0`.
- **One NaN.** Every NaN in `x` has the same bit pattern.
- **No signaling NaN.** Every NaN in `x` is a quiet one, as every NaN that
  arithmetic produces is. NumPy's result for a signaling NaN comes from the C
  library's `fmax`, which does not ignore it: the running value becomes a
  quiet NaN, and the maximum so far is forgotten.

`x` is passed through `_no_negative_zero` before every call the judge checks,
which makes every `-0.0` a `+0.0`. The judge's value classes make one NaN
pattern only, `np.nan`, and no signaling NaN, so the other two conditions hold
of every generated input as it is.

On this domain every correct maximum gives NumPy's output bit for bit, on
every architecture: a NaN element never changes the running value, and while
every element so far is NaN the output is that NaN, sign and payload
included; from the first element that is not NaN on, the output is never NaN
again. Outside it NumPy's builds differ, which is why the binding leaves those
arrays to NumPy. NumPy's ARM64 build is one `fmaxnm` per element, so `+0.0`
beats `-0.0`. Its x86-64 build keeps the later of two equal values, so the
sign of a zero output is the sign of the last zero equal to the maximum, and
for `n == 2` it takes the pair the other way round: of two zeros the first is
`out[1]`, and of two NaNs the second. `naive.c` and the x86-64 kernel do what
the x86-64 build does; the judge no longer checks it.
"""
import numpy as np

from judge.spec import In, Kernel, Out, Size

# Share of NaN in the `some_nan` timing distribution.
NAN_SHARE = 0.05


def oracle(x):
    """The running maximum of x over the elements that are not NaN."""
    return np.fmax.accumulate(x)


def _no_negative_zero(x):
    """Make an input satisfy the domain: every `-0.0` becomes `+0.0`.

    The binding sends an array that holds a negative zero to NumPy, so the
    kernel is never called with one.

    Args:
        x: A generated float64 array.

    Returns:
        A new array equal to `x` except that every zero is `+0.0`.
    """
    return np.where(x == 0, 0.0, x)


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
    args=[In("x", "f64", prepare=_no_negative_zero), Out("out", "f64"), Size("n")],
    oracle=oracle,
    distributions={"random": _random, "rising": _rising, "some_nan": _some_nan},
)
