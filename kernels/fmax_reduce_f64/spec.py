"""Maximum that ignores NaN: `np.fmax.reduce(x)` for float64 data.

The kernel behind `np.fmax.reduceat(x, starts)`, which is this reduction once
per run: NumPy gives a run of `reduceat` the bits it gives the same elements
in `np.fmax.reduce`.

The result is the largest element of `x` that is not a NaN, and a NaN only if
every element is one.

**The domain** is what the binding passes to the kernel for one run; it sends
every other call to NumPy (`fmax_runs` and `fmax_runs_ambiguous` in
`hotkernels/src/hotkernels/_binding/hook_method.c`):

- **No negative zero at a zero maximum.** If the maximum is a zero, no
  element of `x` is `-0.0`. A `-0.0` below a larger maximum is in the domain.
- **One NaN when all are NaN.** If every element of `x` is a NaN, they all
  have the same bit pattern.
- **No signaling NaN.** Every NaN in `x` is a quiet one, as every NaN that
  arithmetic produces is. NumPy's leftover elements go through the C
  library's `fmax`, which does not ignore a signaling NaN.

`x` is passed through `_no_negative_zero_at_a_zero_maximum` before every call
the judge checks, which makes every `-0.0` a `+0.0` when the maximum is a
zero. The judge's value classes make one NaN pattern only, `np.nan`, and no
signaling NaN, so the other two conditions hold of every generated input as
it is.

On this domain every correct maximum gives NumPy's result bit for bit, on
every architecture. Outside it NumPy's answer for the sign of a zero maximum,
and for which NaN an all-NaN array returns, depends on its build, which is
why the binding leaves those calls to NumPy. NumPy's ARM64 build is one
`fmaxnm` per element from `x[0]`: a zero maximum is `+0.0` if any `+0.0` is
present, and an all-NaN array gives `x[0]`. Its x86-64 build with AVX2 and
without AVX-512 goes through the array in vectors of `VECTOR_LANES` elements,
and `naive.c` and the x86-64 kernel reproduce that loop, which the judge no
longer checks:

- `x[0]` starts `VECTOR_LANES` running maxima, one per lane.
- The elements from `x[1]` on are taken `VECTOR_LANES` at a time, each into
  its own lane, as long as a whole group is left. A NaN element changes
  nothing; otherwise the element replaces a NaN, a smaller value and an equal
  one, so a lane holds the last of its equal maxima.
- The lanes are combined: lane 0 with lane 2 and lane 1 with lane 3, then
  the two results, each time keeping the second of two equal values and a
  number over a NaN.
- The elements left over, at most `VECTOR_LANES - 1`, are taken one at a
  time. Here an element replaces a NaN and a smaller value, and an equal value
  does not replace.

What follows from it, with `groups = (n - 1) // VECTOR_LANES` whole groups
and `n - 1 - VECTOR_LANES * groups` leftover elements:

- **Zeros.** When the maximum is a zero, its sign is that of one particular
  zero: the last zero of lane 3 (the elements `x[4]`, `x[8]`, ...); if lane 3
  has none, `x[0]` if it is a zero; otherwise the last zero of lane 1, of
  lane 2, of lane 0, in that order; otherwise the first zero among the
  leftover elements. For `n <= VECTOR_LANES` that is the first zero of the
  array.
- **All NaN.** The result is `x[0]`, sign and payload, when there are no
  leftover elements, and the last element when there are.

NumPy built with AVX-512 has vectors of another length and answers those two
cases differently again.
"""
import numpy as np

from judge.spec import In, Kernel, Size

# Elements in one vector of NumPy's own loop for this reduction on x86-64 with
# AVX2. The naive reference follows NumPy's lanes, which matter only outside
# the domain; see the module docstring.
VECTOR_LANES = 4
# Share of NaN in the `some_nan` timing distribution.
NAN_SHARE = 0.05


def oracle(x):
    """The maximum of x over the elements that are not NaN.

    Args:
        x: The input array, at least one element.

    Returns:
        The maximum as a float64 scalar, a NaN if every element is one.
    """
    return np.fmax.reduce(x)


def _no_negative_zero_at_a_zero_maximum(x):
    """Make an input satisfy the domain: every `-0.0` becomes `+0.0` when the maximum is a zero.

    The binding sends a call to NumPy when a run's maximum is a zero and the
    array holds a negative zero. Where the maximum is not a zero the array is
    left as it is, so a `-0.0` below the maximum is still checked.

    Args:
        x: A generated float64 array, at least one element.

    Returns:
        `x` itself when its maximum is not a zero, and otherwise a new array
        equal to `x` except that every zero is `+0.0`.
    """
    if x.size == 0 or np.fmax.reduce(x) != 0:
        return x
    return np.where(x == 0, 0.0, x)


def _random(rng, n):
    """Build one timing input of random values.

    Args:
        rng: The judge's NumPy random generator.
        n: Element count.

    Returns:
        The kernel's input arrays by argument name.
    """
    return {"x": rng.standard_normal(n)}


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
    name="fmax_reduce_f64",
    returns="f64",
    args=[In("x", "f64", prepare=_no_negative_zero_at_a_zero_maximum), Size("n")],
    oracle=oracle,
    min_n=1,  # np.fmax.reduce raises on empty input; the kernel is never called with n = 0
    distributions={"random": _random, "some_nan": _some_nan},
)
