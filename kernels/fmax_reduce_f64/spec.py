"""Maximum that ignores NaN: `np.fmax.reduce(x)` for float64 data.

The kernel behind `np.fmax.reduceat(x, starts)`, which is this reduction once
per run: NumPy gives a run of `reduceat` the bits it gives the same elements
in `np.fmax.reduce`.

The result is the largest element of `x` that is not a NaN, and a NaN only if
every element is one. It is NumPy's bit for bit, which fixes two things a
maximum leaves open: the sign of a zero maximum when zeros of both signs are
present, and which NaN is returned when every element is a NaN. NumPy's
answer to both depends on where the elements sit, because of how its loop
goes through the array, and no rule simpler than that loop reproduces it. So
`naive.c` goes through the array the way NumPy does **on an x86-64 machine
with AVX2 and without AVX-512**, where a vector of NumPy's loop has
`VECTOR_LANES` elements:

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

NumPy built for another instruction set (AVX-512, NEON) has vectors of
another length and may answer those two cases differently; this spec is
NumPy's answer on the machines named above.

**No signaling NaN.** Every NaN in `x` is a quiet one, as every NaN that
arithmetic produces is. NumPy's leftover elements go through the C library's
`fmax`, which does not ignore a signaling NaN. A caller that may hold one
checks before the call.
"""
import numpy as np

from judge.spec import In, Kernel, Size

# Elements in one vector of NumPy's own loop for this reduction on x86-64 with
# AVX2. The naive reference follows NumPy's lanes; see the module docstring.
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
    args=[In("x", "f64"), Size("n")],
    oracle=oracle,
    min_n=1,  # np.fmax.reduce raises on empty input; the kernel is never called with n = 0
    distributions={"random": _random, "some_nan": _some_nan},
)
