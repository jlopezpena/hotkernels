"""Count occurrences of each code: `np.bincount(x, minlength=bins)` for int16 codes.

`counts` has `bins` elements, and `counts[v]` is how many elements of `x` equal
`v`, for every `v` from 0 to `bins - 1`. Elements outside that range, negative
ones and ones at or above `bins`, are ignored. That is deliberate: the kernel
is safe on any input, so no caller has to be trusted. NumPy raises for a
negative element and sizes its result by the largest one; a caller that wants
NumPy's answer reads the range of `x` first and calls the kernel only when
every element is in `[0, bins)`, where the two agree.

The kernel is defined for every `bins` from 1 up.
"""
import numpy as np

from judge.spec import In, Kernel, Out, Scalar, ScalarSet, Size

# Element type of the codes.
DTYPE = np.int16

# The default number of counters: 512 bytes of them, eight cache lines.
BINS = 64

# Further counter counts. The judge checks every one and times `thousand_bins`.
SCALAR_SETS = [
    # The smallest tables: only zeros are counted, or zeros and ones.
    ScalarSet("one_bin", {"bins": 1}),
    ScalarSet("two_bins", {"bins": 2}),
    # One counter per byte value.
    ScalarSet("byte_bins", {"bins": 256}),
    # 8,000 bytes of counters against the default's 512: two 4 KB pages to
    # clear on every call, and a table several of which no longer fit the
    # first-level cache together.
    ScalarSet("thousand_bins", {"bins": 1000}, timed=True),
    # More counters than int16 has non-negative values: every element that is
    # not negative is counted, and a negative one read as unsigned would be too.
    ScalarSet("all_codes", {"bins": 65536}),
]

# `codes_wide` draws from [WIDE_LO, WIDE_HI): some elements are negative, and
# some are outside a small `bins` and inside a large one.
WIDE_LO, WIDE_HI = -8, 300

# The value every element of `one_code` has: one counter takes every increment.
ONE_CODE = 5

# The values of `edges`: the last counter and the first value past it for the
# counter counts 1, 64, 256 and 1000, with -1 and 1. All fit in int16.
EDGE_VALUES = np.array([-1, 0, 1, 63, 64, 255, 256, 999, 1000], DTYPE)

# How many distinct values the `few_values` timing distribution has.
FEW_VALUES = 4


def _codes(base):
    """Value class: uniform codes in [0, BINS), from the uniform default array."""
    return np.mod(base, BINS).astype(DTYPE)


def _codes_wide(base):
    """Value class: uniform in [WIDE_LO, WIDE_HI), from the uniform default array."""
    return (np.mod(base, WIDE_HI - WIDE_LO) + WIDE_LO).astype(DTYPE)


def _one_code(base):
    """Value class: every element is ONE_CODE."""
    return np.full(base.shape, ONE_CODE, DTYPE)


def _edges(base):
    """Value class: only the values of EDGE_VALUES, picked by the default array."""
    return EDGE_VALUES[np.mod(base, len(EDGE_VALUES))]


# Value classes of `x`, next to the general integer ones, which are mostly
# large or negative values that the kernel ignores. A class receives the
# default random array and no generator, so each derives its values from that
# array, which is uniform over the whole of int16.
X_CLASSES = {
    "codes": _codes,
    "codes_wide": _codes_wide,
    "one_code": _one_code,
    "edges": _edges,
}


def oracle(x, bins):
    """How many elements of x equal each value from 0 to bins - 1; the others are ignored."""
    wide = x.astype(np.int64)
    return np.bincount(wide[(wide >= 0) & (wide < bins)], minlength=bins)


def bincount(x, bins):
    """The call a user writes, timed as NumPy. Every timing input is inside [0, bins)."""
    return np.bincount(x, minlength=bins)


def _uniform(rng, n):
    """Timing input: codes uniform in [0, BINS)."""
    return {"x": rng.integers(0, BINS, n).astype(DTYPE)}


def _few_values(rng, n):
    """Timing input: FEW_VALUES distinct codes, so neighbouring elements often share a counter."""
    return {"x": rng.integers(0, FEW_VALUES, n).astype(DTYPE)}


KERNEL = Kernel(
    name="bincount_i16",
    returns="void",
    args=[
        In("x", "i16", classes=X_CLASSES),
        Out("counts", "i64", capacity="bins"),
        Size("n"),
        Scalar("bins", "usize", BINS),
    ],
    oracle=oracle,
    numpy=bincount,
    scalar_sets=SCALAR_SETS,
    distributions={"uniform": _uniform, "few_values": _few_values},
)
