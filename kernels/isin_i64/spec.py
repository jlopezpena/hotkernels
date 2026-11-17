"""Membership in a small set: `np.isin(x, test)` with TEST_VALUES test values."""
import numpy as np

from judge.spec import In, Kernel, Out, Size

# Number of test values. Small sets are where NumPy's sort-based method is slowest
# relative to comparing against every value.
TEST_VALUES = 8


def oracle(x, test):
    """For each element of x, whether it equals any of the test values."""
    return np.isin(x, test)


def _padded_like_the_binding(test):
    """Repeat the first test value over the tail of the set, as the binding pads a short one.

    `np.isin(x, [1, 2, 3])` reaches the kernel as `[1, 2, 3, 1, 1, 1, 1, 1]`:
    the binding fills the set to TEST_VALUES by repeating its first value
    (`isin_test_values` in `hotkernels/src/hotkernels/_binding/hook_function.c`).
    Most calls from Python are therefore made with repeated test values, so
    the check cases are given that shape too.

    A `prepare` function sees only the array, so the number of values kept is
    taken from the array itself: the first value's remainder modulo
    TEST_VALUES, plus one. Sets that start with different values are cut at
    different lengths, from a single value repeated eight times to all eight
    kept. The value classes decide where the first value lies in the set:
    smallest in `asc`, largest in `desc`, anywhere in `random`.

    Args:
        test: The TEST_VALUES test values as generated.

    Returns:
        A copy in which every value after the kept ones equals `test[0]`.
    """
    kept = int(test[0] % TEST_VALUES) + 1
    padded = test.copy()
    padded[kept:] = test[0]
    return padded


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
    args=[In("x", "i64"), In("test", "i64", len=TEST_VALUES, prepare=_padded_like_the_binding), Out("out", "bool"), Size("n")],
    oracle=oracle,
    distributions={"mostly_misses": _mostly_misses, "wide_values": _wide_values},
)
