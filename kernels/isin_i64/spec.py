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


KERNEL = Kernel(
    name="isin_i64",
    returns="void",
    args=[In("x", "i64"), In("test", "i64", len=TEST_VALUES), Out("out", "bool"), Size("n")],
    oracle=oracle,
    distributions={"mostly_misses": _mostly_misses},
)
