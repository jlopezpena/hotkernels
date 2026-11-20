"""The distinct values of int64 labels: `np.unique(x)` when the values lie in a narrow range.

`out` has `span` elements. The kernel writes the distinct values of `x` that
lie in `[lo, lo + span)` to `out[0:count]` in ascending order and returns
`count`. Elements outside that range are ignored. That is deliberate, as for
the bincount kernels: the kernel is safe on any input, so no caller has to be
trusted. A caller that wants NumPy's answer reads the smallest and largest
element first and calls the kernel only when every element is inside the
range, where the two agree.

The range is tested in unsigned arithmetic: `x` is inside it when
`(uint64)x - (uint64)lo < span`. The kernel is defined for ranges whose last
value, `lo + span - 1`, is at most the largest int64, as every range the
binding passes is (`lo` the smallest element, `lo + span - 1` the largest). A
range past it would wrap round to the smallest int64, and its values would
come out in unsigned order rather than NumPy's.

`out` is scratch: the kernel may use all of it as a table before it writes
the values to its front. Only the first `count` elements are compared.

The kernel is defined for every `span` from 1 up within that limit.
"""
import numpy as np

from judge.spec import In, Kernel, Out, Scalar, ScalarSet, Size, USED_RETURN

# Element type of the labels.
DTYPE = np.int64

# The default range: ten classes from 0, the multi-class labels of the
# application the kernel was written for (docs/unique-design.md).
LO = 0
SPAN = 10

# The smallest and largest int64.
INT64_MIN = int(np.iinfo(DTYPE).min)
INT64_MAX = int(np.iinfo(DTYPE).max)

# Further ranges. The judge checks every one and times `byte_span`.
SCALAR_SETS = [
    # One value only, and the two values of binary labels.
    ScalarSet("one_value", {"span": 1}),
    ScalarSet("binary", {"span": 2}),
    # A range below zero and across it.
    ScalarSet("negative", {"lo": -8, "span": 11}),
    # The largest range the binding keeps on its stack: 256 values, 2 KB.
    ScalarSet("byte_span", {"lo": -8, "span": 256}, timed=True),
    # A range the size of a large array: 512 KB of table.
    ScalarSet("wide", {"lo": -8, "span": 65536}),
    # Ranges at both ends of int64. The second ends at the largest int64,
    # where `lo + span` overflows in signed arithmetic.
    ScalarSet("bottom", {"lo": INT64_MIN, "span": 4}),
    ScalarSet("top", {"lo": INT64_MAX - 3, "span": 4}),
]

# `near` draws from [NEAR_LO, NEAR_HI): around zero, some values inside each
# narrow range and some outside it.
NEAR_LO, NEAR_HI = -12, 300

# The value every element of `one_value` has.
ONE_VALUE = 5

# The values of `edges`: the first and last value of each range of the
# scalar sets and the values next to them, and the ends of int64.
EDGE_VALUES = np.array([
    -9, -8, -7, -1, 0, 1, 2, 3, 9, 10, 11, 247, 248, 65527, 65528,
    INT64_MIN, INT64_MIN + 1, INT64_MIN + 3, INT64_MIN + 4,
    INT64_MAX - 4, INT64_MAX - 3, INT64_MAX - 1, INT64_MAX,
], DTYPE)

# `high_bits` adds a multiple of HIGH_STEP, from 0 to HIGH_STEPS - 1 times, to
# a label: a value whose low 32 bits are a valid label and which is far
# outside every range unless nothing was added.
HIGH_STEP = 1 << 32
HIGH_STEPS = 3

# Distinct values of the timing distributions: binary labels and ten classes.
BINARY_VALUES = 2
CLASS_VALUES = 10


def _labels(base):
    """Value class: binary labels, 0 and 1, from the uniform default array."""
    return np.mod(base, BINARY_VALUES).astype(DTYPE)


def _classes(base):
    """Value class: ten classes, 0 to 9, from the uniform default array."""
    return np.mod(base, CLASS_VALUES).astype(DTYPE)


def _near(base):
    """Value class: uniform in [NEAR_LO, NEAR_HI), from the uniform default array."""
    return (np.mod(base, NEAR_HI - NEAR_LO) + NEAR_LO).astype(DTYPE)


def _one_value(base):
    """Value class: every element is ONE_VALUE."""
    return np.full(base.shape, ONE_VALUE, DTYPE)


def _edges(base):
    """Value class: only the values of EDGE_VALUES, picked by the default array."""
    return EDGE_VALUES[np.mod(base, len(EDGE_VALUES))]


def _high_bits(base):
    """Value class: classes 0 to 9, two in three of them with bits above the low 32 set."""
    return (np.mod(base, CLASS_VALUES) + np.mod(base // CLASS_VALUES, HIGH_STEPS) * HIGH_STEP).astype(DTYPE)


# Value classes of `x`, next to the general integer ones, which are mostly
# large or negative values that the kernel ignores. A class receives the
# default random array and no generator, so each derives its values from that
# array, which is uniform over the whole of int64.
X_CLASSES = {
    "labels": _labels,
    "classes": _classes,
    "near": _near,
    "one_value": _one_value,
    "edges": _edges,
    "high_bits": _high_bits,
}


def oracle(x, lo, span):
    """The distinct values of x inside [lo, lo + span), ascending; the others are ignored."""
    offset = x.view(np.uint64) - np.uint64(lo % (1 << 64))
    return np.unique(x[offset < np.uint64(span)])


def unique(x, lo, span):
    """The call a user writes, timed as NumPy. Every timing input is inside [lo, lo + span)."""
    return np.unique(x)


def _binary(rng, n):
    """Timing input: binary labels, 0 and 1."""
    return {"x": rng.integers(0, BINARY_VALUES, n).astype(DTYPE)}


def _ten_classes(rng, n):
    """Timing input: ten classes, 0 to 9."""
    return {"x": rng.integers(0, CLASS_VALUES, n).astype(DTYPE)}


KERNEL = Kernel(
    name="unique_i64",
    returns="usize",
    args=[
        In("x", "i64", classes=X_CLASSES),
        Out("out", "i64", capacity="span", used=USED_RETURN, scratch=True),
        Size("n"),
        Scalar("lo", "i64", LO),
        Scalar("span", "usize", SPAN),
    ],
    oracle=oracle,
    numpy=unique,
    scalar_sets=SCALAR_SETS,
    distributions={"binary": _binary, "ten_classes": _ten_classes},
)
