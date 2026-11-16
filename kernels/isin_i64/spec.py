"""Membership in a small set: `np.isin(x, test)` with TEST_VALUES test values."""
import numpy as np

from judge.spec import In, Kernel, Out, Size

# Number of test values. Small sets are where NumPy's sort-based method is slowest
# relative to comparing against every value.
TEST_VALUES = 8


def oracle(x, test):
    """For each element of x, whether it equals any of the test values."""
    return np.isin(x, test)


def _mostly_misses(rng, n):
    """Timing input: values and test values drawn from a range wide enough that few match."""
    return {"x": rng.integers(0, 1000, n), "test": rng.integers(0, 1000, TEST_VALUES)}


def _wide_values(rng, n):
    """Timing input: values and test values spread over the whole int64 range.

    Small non-negative values, as in `mostly_misses`, allow tricks that narrow
    the comparison. This distribution shows what a kernel does when they do not
    apply. Half of x is drawn from the test values so that matches occur.
    """
    info = np.iinfo(np.int64)
    test = rng.integers(info.min, info.max, TEST_VALUES)
    x = rng.integers(info.min, info.max, n)
    hit = rng.random(n) < 0.5
    x[hit] = rng.choice(test, int(hit.sum()))
    return {"x": x, "test": test}


KERNEL = Kernel(
    name="isin_i64",
    returns="void",
    args=[In("x", "i64"), In("test", "i64", len=TEST_VALUES), Out("out", "bool"), Size("n")],
    oracle=oracle,
    distributions={"mostly_misses": _mostly_misses, "wide_values": _wide_values},
)
