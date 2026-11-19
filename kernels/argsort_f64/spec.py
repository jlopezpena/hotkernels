"""Stable sort order: `np.argsort(x, kind="stable")` for float64.

`out[0..n)` receives the indices that sort `x`, equal elements in their
original order. NumPy orders NaN last and calls two elements equal when
neither is less than the other, so `-0.0` and `+0.0` are equal, and so are
any two NaNs, whatever their signs and payloads. A stable sort has one right
answer, and the kernel returns it for every input.

`out` has room for `3n` elements: the result, and `2n` of working memory that
the kernel leaves in any state. It is a scratch output, so whatever calls the
kernel owns the buffer and hands on only the first `n` elements.

The kernel is defined for `n` below 2**26.
"""
import numpy as np

from judge.spec import USED_RETURN, In, Kernel, Out, Size, Times

# Elements of working memory and result together, per element of input.
OUT_PER_ELEMENT = 3

# Value classes: how many distinct values `few_values` has, the range of the
# whole numbers of `whole`, and the share of each special value in the mixes.
FEW_VALUES = 3
WHOLE_RANGE = 1000
SPECIAL_SHARE = 0.4

# Bit patterns of NaNs that differ in sign and payload. NumPy calls them all
# equal, so a sort must leave them in their original order.
NAN_PATTERNS = np.array([0x7FF8000000000000, 0xFFF8000000000000, 0x7FF0000000000001,
                         0xFFF0000000000001, 0x7FFFFFFFFFFFFFFF, 0xFFF8000000000123], np.uint64)

# The values of `specials`: each is one end of the order or a tie across signs.
SPECIAL_VALUES = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1.0, -1.0,
                           np.finfo(np.float64).max, -np.finfo(np.float64).max,
                           np.finfo(np.float64).smallest_subnormal, -np.finfo(np.float64).smallest_subnormal])

# Timing: the scale of the classifier scores (see `_scores`).
SCORE_SHIFT = 1.5


def _nan_kinds(base):
    """Value class: numbers mixed with NaNs of several signs and payloads."""
    picks = np.abs(base * 1e6).astype(np.int64)
    nans = NAN_PATTERNS[picks % len(NAN_PATTERNS)].view(np.float64)
    return np.where(picks % 5 < 2, nans, base)


def _zero_signs(base):
    """Value class: zeros of both signs among a few other values, so every element has ties."""
    picks = np.abs(base * 1e6).astype(np.int64) % (FEW_VALUES + 2)
    return np.array([0.0, -0.0, -1.0, 1.0, 2.0])[picks]


def _specials(base):
    """Value class: the ends of the order and its ties across signs, mixed with numbers."""
    picks = np.abs(base * 1e6).astype(np.int64)
    special = SPECIAL_VALUES[picks % len(SPECIAL_VALUES)]
    return np.where((picks // len(SPECIAL_VALUES)) % 10 < 10 * SPECIAL_SHARE, special, base)


def _whole(base):
    """Value class: whole numbers, whose bits differ only high in the word."""
    return np.floor(np.abs(base) * WHOLE_RANGE % WHOLE_RANGE)


def _positive(base):
    """Value class: magnitudes, a narrower window of bits than numbers of both signs."""
    return np.abs(base)


def _nearly_sorted(base):
    """Value class: ascending, with the last few elements out of place."""
    out = np.sort(base)
    tail = max(1, len(out) // 20)
    out[len(out) - tail:] = base[:tail]
    return out


X_CLASSES = {
    "nan_kinds": _nan_kinds,
    "zero_signs": _zero_signs,
    "specials": _specials,
    "whole": _whole,
    "positive": _positive,
    "nearly_sorted": _nearly_sorted,
}


def oracle(x):
    """The indices that sort x, equal elements in their original order."""
    return np.argsort(x, kind="stable")


def _uniform(rng, n):
    """Timing input: uniform in [0, 1)."""
    return {"x": rng.random(n)}


def _scores(rng, n):
    """Timing input: the scores of a binary classifier, as scikit-learn's metrics sort them."""
    labels = rng.integers(0, 2, n)
    return {"x": 1 / (1 + np.exp(-(rng.standard_normal(n) + SCORE_SHIFT * (labels - 0.5))))}


def _normal(rng, n):
    """Timing input: standard normal, numbers of both signs."""
    return {"x": rng.standard_normal(n)}


KERNEL = Kernel(
    name="argsort_f64",
    returns="usize",  # number of indices written: n
    args=[
        In("x", "f64", classes=X_CLASSES),
        Out("out", "i64", capacity=Times(OUT_PER_ELEMENT), used=USED_RETURN, scratch=True),
        Size("n"),
    ],
    oracle=oracle,
    distributions={"uniform": _uniform, "scores": _scores, "normal": _normal},
)
