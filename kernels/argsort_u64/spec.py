"""Stable sort order: `np.argsort(x, kind="stable")` for uint64.

`out[0..n)` receives the indices that sort `x`, equal elements in their
original order. A stable sort has one right answer, and the kernel returns it
for every input.

`out` has room for `3n` elements: the result, and `2n` of working memory that
the kernel leaves in any state. It is a scratch output, so whatever calls the
kernel owns the buffer and hands on only the first `n` elements.

The kernel is defined for `n` below 2**26.
"""
import numpy as np

from judge.spec import USED_RETURN, In, Kernel, Out, Size, Times

# Element type of the keys.
DTYPE = np.uint64

# Elements of working memory and result together, per element of input.
OUT_PER_ELEMENT = 3

# `codes` holds values in [0, CODES): a window of ten bits, sorted in one pass.
CODES = 1000
# `high_window` shifts such values up by this many bits: the bits that differ
# are high in the word and the low ones are all the same.
HIGH_SHIFT = 44
# `two_fields` adds a second group of differing bits this far up, with bits
# that never differ between the two, so some passes have nothing to do.
FIELD_SHIFT = 40
# `around_zero` holds values in [-AROUND, AROUND]: as unsigned numbers, the
# smallest and the largest there are.
AROUND = 3
# Timing: the range of `small_range`.
SMALL_RANGE = 1000


def _codes(base):
    """Value class: values in [0, CODES), so duplicates are everywhere."""
    return np.mod(base, CODES).astype(DTYPE)


def _high_window(base):
    """Value class: values in [0, CODES) shifted up by HIGH_SHIFT bits."""
    return (np.mod(base, CODES).astype(np.uint64) << np.uint64(HIGH_SHIFT)).astype(DTYPE)


def _two_fields(base):
    """Value class: two small fields of differing bits FIELD_SHIFT apart."""
    low = np.mod(base, 8).astype(np.uint64)
    high = np.mod(base // 8, 8).astype(np.uint64)
    return (low | (high << np.uint64(FIELD_SHIFT))).astype(DTYPE)


def _around_zero(base):
    """Value class: the values next to zero, which as unsigned numbers are the smallest and the largest."""
    return (np.mod(base, 2 * AROUND + 1).astype(np.int64) - AROUND).astype(DTYPE)


def _nearly_sorted(base):
    """Value class: ascending, with the last few elements out of place."""
    out = np.sort(base)
    tail = max(1, len(out) // 20)
    out[len(out) - tail:] = base[:tail]
    return out


X_CLASSES = {
    "codes": _codes,
    "high_window": _high_window,
    "two_fields": _two_fields,
    "around_zero": _around_zero,
    "nearly_sorted": _nearly_sorted,
}


def oracle(x):
    """The indices that sort x, equal elements in their original order."""
    return np.argsort(x, kind="stable")


def _full_range(rng, n):
    """Timing input: uniform over the whole range of the type, as hashes are."""
    info = np.iinfo(DTYPE)
    return {"x": rng.integers(info.min, info.max, n, dtype=DTYPE, endpoint=True)}


def _small_range(rng, n):
    """Timing input: values in [0, SMALL_RANGE), as labels and codes are."""
    return {"x": rng.integers(0, SMALL_RANGE, n).astype(DTYPE)}


KERNEL = Kernel(
    name="argsort_u64",
    returns="usize",  # number of indices written: n
    args=[
        In("x", "u64", classes=X_CLASSES),
        Out("out", "i64", capacity=Times(OUT_PER_ELEMENT), used=USED_RETURN, scratch=True),
        Size("n"),
    ],
    oracle=oracle,
    distributions={"full_range": _full_range, "small_range": _small_range},
)
