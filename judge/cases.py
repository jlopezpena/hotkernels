"""Generates the inputs a kernel is checked against.

A "case" is one call to the kernel: an element count, a placement in memory, a
set of values for the scalar arguments, and a "value class" for each input
argument. A value class is a recipe for an array, such as `nan_at` (random data
with a NaN at one index) or `all_true`.

Every case is fully determined by its description, so a failure can always be
rerun from the text the judge prints.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

from .memory import ARRANGEMENTS
from .spec import DEFAULT_SCALARS, In, InOut, Kernel, element_count, np_dtype

# --- Which sizes are tested -------------------------------------------------

# Every size from min_n up to this is tested, and "positional" classes are
# tried at every index. 65 is one past 64, the widest block a kernel is likely
# to process per step, so every leftover count is covered.
EXHAUSTIVE_MAX_N = 65
# Above the exhaustive range, powers of two from 2**7 to 2**22 (about four
# million) are tested, each with one less and one more.
MIN_POWER = 7
MAX_POWER = 22
# Arrays larger than this are slow to generate, so they get fewer cases.
LARGE_MIN_N = 65536
# Kernels with two inputs get every pairing of value classes up to this size.
CROSS_PRODUCT_MAX_N = 4096
# The field of a case description that names its scalar set.
SCALARS_FIELD = "scalars"

# --- Fresh random cases -----------------------------------------------------

# How many extra cases are generated from a new seed on every run.
FRESH_CASE_COUNT = 400
# Most fresh cases are small; a share of them are larger.
FRESH_SMALL_MAX_N = 300
FRESH_LARGE_MAX_N = 5000
FRESH_LARGE_SHARE = 0.1

# --- Values used by the value classes ---------------------------------------

FLOAT_EQUAL_VALUE = 1.5      # every element, in the "equal" class
INT_EQUAL_VALUE = 7          # every element, in the integer "equal" class
TIE_DISTINCT_VALUES = 4      # few distinct values, so duplicates are common
DENORMAL_MULTIPLES = 50      # denormals are small multiples of the smallest one
INF_SHARE = 0.1              # share of +inf, and of -inf, in "inf_mix"
NAN_SHARE = 0.2              # share of NaN in "nan_some"
SPIKE_BACKGROUND = 1         # background is clipped to [-1, 1] ...
SPIKE_VALUE = 10             # ... so one element at +/-10 clearly stands out
BOOL_SPARSE_SHARE = 0.05     # share of True in "sparse"
BOOL_DENSE_SHARE = 0.95      # share of True in "dense"
EVEN_SHARE = 0.5             # a fair coin

RANDOM = "random"


@dataclass
class _Recipe:
    """What a value-class function needs to build its array."""

    # NumPy dtype of the array to build.
    dt: np.dtype
    # Element count.
    n: int
    # Index of the interesting element, for positional classes. 0 otherwise.
    p: int
    # Random generator seeded from the case description.
    rng: np.random.Generator
    # The argument's default random array, which many classes modify.
    base: np.ndarray


# Value classes, keyed by (kind, class name), in definition order.
# kind is "f" for floats, "b" for booleans, "i" for integers.
_MAKERS: dict[tuple[str, str], Callable[[_Recipe], np.ndarray]] = {}
# Names of the classes that place one interesting element at a chosen index.
POSITIONAL: set[str] = set()


def _value_class(kind: str, name: str, positional: bool = False):
    """Register a function as the value class `name` for one kind of data."""
    def register(fn):
        """Store the function in the registry and return it unchanged."""
        _MAKERS[(kind, name)] = fn
        if positional:
            POSITIONAL.add(name)
        return fn
    return register


def _spike_background(r: _Recipe) -> np.ndarray:
    """Return the base array clipped to a narrow range, ready to receive a spike."""
    return np.clip(r.base, -SPIKE_BACKGROUND, SPIKE_BACKGROUND)


def _with_value_at(r: _Recipe, value) -> np.ndarray:
    """Return a copy of the base array with one element replaced."""
    a = r.base.copy()
    a[r.p] = value
    return a


# --- Float value classes ----------------------------------------------------

@_value_class("f", RANDOM)
def _f_random(r):
    """Normally distributed values."""
    return r.base


@_value_class("f", "equal")
def _f_equal(r):
    """Every element the same."""
    return np.full(r.n, FLOAT_EQUAL_VALUE, r.dt)


@_value_class("f", "asc")
def _f_asc(r):
    """Sorted ascending."""
    return np.sort(r.base)


@_value_class("f", "desc")
def _f_desc(r):
    """Sorted descending."""
    return np.sort(r.base)[::-1].copy()


@_value_class("f", "ties")
def _f_ties(r):
    """Only a few distinct values, so most elements have duplicates."""
    return r.rng.integers(0, TIE_DISTINCT_VALUES, r.n).astype(r.dt)


@_value_class("f", "zeros")
def _f_zeros(r):
    """A mix of positive and negative zero."""
    return np.where(r.rng.random(r.n) < EVEN_SHARE, 0.0, -0.0).astype(r.dt)


@_value_class("f", "denormal")
def _f_denormal(r):
    """Denormal (subnormal) values of both signs."""
    multiples = r.rng.integers(-DENORMAL_MULTIPLES, DENORMAL_MULTIPLES, r.n)
    return (multiples * np.finfo(r.dt).smallest_subnormal).astype(r.dt)


@_value_class("f", "inf_mix")
def _f_inf_mix(r):
    """Random values with some positive and negative infinities."""
    a = r.base.copy()
    draw = r.rng.random(r.n)
    a[draw < INF_SHARE] = np.inf
    a[draw > 1 - INF_SHARE] = -np.inf
    return a


@_value_class("f", "nan_all")
def _f_nan_all(r):
    """Every element NaN."""
    return np.full(r.n, np.nan, r.dt)


@_value_class("f", "nan_some")
def _f_nan_some(r):
    """Random values with NaN scattered through them."""
    a = r.base.copy()
    a[r.rng.random(r.n) < NAN_SHARE] = np.nan
    return a


@_value_class("f", "extremes")
def _f_extremes(r):
    """The largest and smallest representable magnitudes, and zero."""
    info = np.finfo(r.dt)
    return r.rng.choice(np.array([info.max, -info.max, info.tiny, -info.tiny, 0.0], r.dt), r.n)


@_value_class("f", "nan_at", positional=True)
def _f_nan_at(r):
    """Random values with a single NaN at index p."""
    return _with_value_at(r, np.nan)


@_value_class("f", "max_at", positional=True)
def _f_max_at(r):
    """A clear maximum at index p."""
    a = _spike_background(r)
    a[r.p] = SPIKE_VALUE
    return a


@_value_class("f", "min_at", positional=True)
def _f_min_at(r):
    """A clear minimum at index p."""
    a = _spike_background(r)
    a[r.p] = -SPIKE_VALUE
    return a


@_value_class("f", "posinf_at", positional=True)
def _f_posinf_at(r):
    """Random values with positive infinity at index p."""
    return _with_value_at(r, np.inf)


@_value_class("f", "neginf_at", positional=True)
def _f_neginf_at(r):
    """Random values with negative infinity at index p."""
    return _with_value_at(r, -np.inf)


@_value_class("f", "two_max_at", positional=True)
def _f_two_max_at(r):
    """The same maximum at index p and at its mirror image, to test tie-breaking."""
    a = _spike_background(r)
    a[r.p] = SPIKE_VALUE
    a[r.n - 1 - r.p] = SPIKE_VALUE
    return a


# --- Boolean value classes --------------------------------------------------

@_value_class("b", RANDOM)
def _b_random(r):
    """True and False with equal probability."""
    return r.base


@_value_class("b", "all_false")
def _b_all_false(r):
    """Every element False."""
    return np.zeros(r.n, bool)


@_value_class("b", "all_true")
def _b_all_true(r):
    """Every element True."""
    return np.ones(r.n, bool)


@_value_class("b", "sparse")
def _b_sparse(r):
    """Mostly False."""
    return r.rng.random(r.n) < BOOL_SPARSE_SHARE


@_value_class("b", "dense")
def _b_dense(r):
    """Mostly True."""
    return r.rng.random(r.n) < BOOL_DENSE_SHARE


@_value_class("b", "alternating")
def _b_alternating(r):
    """False, True, False, True, ..."""
    return (np.arange(r.n) % 2).astype(bool)


@_value_class("b", "one_true_at", positional=True)
def _b_one_true_at(r):
    """All False except index p."""
    a = np.zeros(r.n, bool)
    a[r.p] = True
    return a


@_value_class("b", "one_false_at", positional=True)
def _b_one_false_at(r):
    """All True except index p."""
    a = np.ones(r.n, bool)
    a[r.p] = False
    return a


# --- Integer value classes --------------------------------------------------

@_value_class("i", RANDOM)
def _i_random(r):
    """Uniform over the whole range of the type."""
    return r.base


@_value_class("i", "small")
def _i_small(r):
    """Only a few distinct values, so most elements have duplicates."""
    return r.rng.integers(0, TIE_DISTINCT_VALUES, r.n).astype(r.dt)


@_value_class("i", "equal")
def _i_equal(r):
    """Every element the same."""
    return np.full(r.n, INT_EQUAL_VALUE, r.dt)


@_value_class("i", "asc")
def _i_asc(r):
    """Sorted ascending."""
    return np.sort(r.base)


@_value_class("i", "desc")
def _i_desc(r):
    """Sorted descending."""
    return np.sort(r.base)[::-1].copy()


@_value_class("i", "extremes")
def _i_extremes(r):
    """The minimum and maximum of the type, and zero."""
    info = np.iinfo(r.dt)
    return r.rng.choice(np.array([info.min, info.max, 0], r.dt), r.n)


@_value_class("i", "max_at", positional=True)
def _i_max_at(r):
    """The type's maximum at index p, over a background of half-range values."""
    a = (r.base // 2).astype(r.dt)
    a[r.p] = np.iinfo(r.dt).max
    return a


@_value_class("i", "min_at", positional=True)
def _i_min_at(r):
    """The type's minimum at index p, over a background of half-range values."""
    a = (r.base // 2).astype(r.dt)
    a[r.p] = np.iinfo(r.dt).min
    return a


# --- Building inputs --------------------------------------------------------

def _kind(dtype: np.dtype) -> str:
    """Map a NumPy dtype to the kind used to look up value classes."""
    return "i" if dtype.kind in "iu" else dtype.kind


def classes(dtype_name: str) -> tuple[list[str], list[str]]:
    """Return the (plain, positional) value-class names for a spec type."""
    kind = _kind(np_dtype(dtype_name))
    names = [name for (k, name) in _MAKERS if k == kind]
    return ([n for n in names if n not in POSITIONAL], [n for n in names if n in POSITIONAL])


def _crc(text: str) -> int:
    """Hash a string to an integer that is stable across runs, for seeding."""
    return zlib.crc32(text.encode())


def base(dtype_name: str, n: int, seed: int, arg: str) -> np.ndarray:
    """Return the default random array for an argument.

    The same (type, size, seed, argument name) always gives the same array.
    """
    dt = np_dtype(dtype_name)
    rng = np.random.default_rng([seed, n, _crc(arg)])
    kind = _kind(dt)
    if kind == "f":
        return rng.standard_normal(n).astype(dt)
    if kind == "b":
        return rng.random(n) < EVEN_SHARE
    info = np.iinfo(dt)
    return rng.integers(info.min, info.max, n, dtype=dt, endpoint=True)


def make(dtype_name: str, cls: str, p: int, n: int, seed: int, arg: str,
         base_arr: np.ndarray) -> np.ndarray:
    """Build the array for one argument from its value class."""
    dt = np_dtype(dtype_name)
    rng = np.random.default_rng([seed, n, _crc(arg), _crc(cls)])
    return _MAKERS[(_kind(dt), cls)](_Recipe(dt, n, p, rng, base_arr))


@dataclass(frozen=True)
class Case:
    """One call to the kernel, described completely enough to reproduce it."""

    # Element count.
    n: int
    # Where the buffers sit in memory: one of memory.ARRANGEMENTS.
    arr: str
    # Seed all random data in the case is derived from.
    seed: int
    # One (argument name, value class, index) per input argument. The index
    # only matters for positional classes.
    picks: tuple[tuple[str, str, int], ...]
    # Which of the kernel's scalar sets supplies the scalar arguments.
    scalars: str = DEFAULT_SCALARS

    def __str__(self) -> str:
        """Format as the text the judge prints and `--case` accepts.

        The scalar set is left out when it is the default one, so a kernel
        without scalar sets prints exactly what it always did.
        """
        parts = [f"n={self.n}", f"arr={self.arr}", f"seed={self.seed}"]
        if self.scalars != DEFAULT_SCALARS:
            parts.append(f"{SCALARS_FIELD}={self.scalars}")
        for arg, cls, p in self.picks:
            parts.append(f"{arg}={cls}@{p}" if cls in POSITIONAL else f"{arg}={cls}")
        return " ".join(parts)

    @staticmethod
    def parse(text: str) -> "Case":
        """Rebuild a case from the text produced by `__str__`."""
        fields = dict(part.split("=", 1) for part in text.split())
        n, arr, seed = int(fields.pop("n")), fields.pop("arr"), int(fields.pop("seed"))
        scalars = fields.pop(SCALARS_FIELD, DEFAULT_SCALARS)
        picks = []
        for arg, value in fields.items():
            cls, _, p = value.partition("@")
            picks.append((arg, cls, int(p or 0)))
        return Case(n, arr, seed, tuple(picks), scalars)


def build_inputs(k: Kernel, case: Case, cache: dict | None = None) -> dict[str, np.ndarray]:
    """Build the input arrays for a case, keyed by argument name.

    `cache` holds each argument's base random array, so the many cases that
    share a size do not regenerate it.
    """
    picks = {arg: (cls, p) for arg, cls, p in case.picks}
    inputs = {}
    for a in k.ins:
        count = element_count(a, case.n)
        key = (a.name, count, case.seed)
        if cache is not None and key in cache:
            base_arr = cache[key]
        else:
            base_arr = base(a.dtype, count, case.seed, a.name)
            if cache is not None:
                cache[key] = base_arr
        cls, p = picks.get(a.name, (RANDOM, 0))
        # A positional class picks its index from the case's element count; a
        # buffer of a fixed length may be shorter than that.
        p = p % count if count else 0
        made = make(a.dtype, cls, p, count, case.seed, a.name, base_arr)
        inputs[a.name] = finish_input(a, made)
    return inputs


def finish_input(arg: In | InOut, values: np.ndarray) -> np.ndarray:
    """Give an input array the layout and order its argument requires.

    Args:
        arg: The argument the array is for.
        values: The array as generated.

    Returns:
        A contiguous array of the argument's type, passed through the
        argument's `prepare` function when the spec gives one.
    """
    array = np.ascontiguousarray(values, np_dtype(arg.dtype))
    if arg.prepare is None:
        return array
    return np.ascontiguousarray(arg.prepare(array), np_dtype(arg.dtype))


def default_input(arg: In | InOut, n: int, seed: int) -> np.ndarray:
    """Build the default random array for an argument, as used for timing.

    Args:
        arg: The argument the array is for.
        n: The value of the kernel's Size argument.
        seed: Seed of the random data.

    Returns:
        Random data of the argument's length, prepared as the spec asks.
    """
    return finish_input(arg, base(arg.dtype, element_count(arg, n), seed, arg.name))


# --- Enumerating cases ------------------------------------------------------

def sizes(min_n: int, max_n: int | None = None) -> list[int]:
    """Return the element counts to test: every small size, then powers of two with neighbours.

    Args:
        min_n: Smallest count to include.
        max_n: Largest count to include, or None for no limit.

    Returns:
        The counts, ascending.
    """
    out = set(range(min_n, EXHAUSTIVE_MAX_N + 1))
    for power in range(MIN_POWER, MAX_POWER + 1):
        out |= {2**power - 1, 2**power, 2**power + 1}
    return sorted(s for s in out if s >= min_n and (max_n is None or s <= max_n))


def _positions(n: int, thin: bool = False) -> list[int]:
    """Return the indices at which positional classes place their element.

    Args:
        n: Element count of the case.
        thin: Use the first, middle and last index even at small sizes, where
            every index is tried otherwise.

    Returns:
        The indices, ascending and without repeats.
    """
    if n <= EXHAUSTIVE_MAX_N and not thin:
        return list(range(n))
    if n > LARGE_MIN_N:
        return [0, n - 1]
    return sorted({0, n // 2, n - 1}) if n else []


def _single_argument_cases(k: Kernel, n: int, seed: int, scalars: str, thin: bool) -> Iterator[Case]:
    """Yield the cases of one size in which one argument at a time leaves random data.

    Args:
        k: The kernel's spec.
        n: Element count.
        seed: Seed of the cases' random data.
        scalars: Name of the scalar set the cases use.
        thin: Yield the reduced sweep: one arrangement per case instead of
            every one, and three indices per positional class.

    Returns:
        An iterator over the cases.
    """
    ins = k.ins
    exhaustive, large = n <= EXHAUSTIVE_MAX_N, n > LARGE_MIN_N
    # Small sizes are tried in every arrangement; larger ones only flush
    # against each guard page.
    arrangements = ARRANGEMENTS if exhaustive else ARRANGEMENTS[:2]
    emitted = 0
    for a in ins:
        plain, positional = classes(a.dtype)
        if large:
            # Keep one class of each sort; large arrays are slow to generate.
            plain, positional = plain[:1], positional[:1]
        if large and thin:
            positional = []
        for cls in plain:
            # The thin sweep rotates through the arrangements instead of trying all.
            chosen = [arrangements[(emitted + n) % len(arrangements)]] if thin else arrangements
            for arr in chosen:
                yield _case(ins, n, arr, seed, scalars, {a.name: (cls, 0)})
                emitted += 1
        for cls in positional:
            for i, p in enumerate(_positions(n, thin)):
                # Rotate through arrangements so each index meets several.
                arr = arrangements[(p + n) % len(arrangements)] if exhaustive \
                    else arrangements[i % len(arrangements)]
                yield _case(ins, n, arr, seed, scalars, {a.name: (cls, p)})


def _case(ins: list, n: int, arr: str, seed: int, scalars: str, chosen: dict) -> Case:
    """Build a case in which the arguments not in `chosen` get random data.

    Args:
        ins: The kernel's input arguments.
        n: Element count.
        arr: Memory arrangement.
        seed: Seed of the case's random data.
        scalars: Name of the scalar set.
        chosen: Argument name -> (value class, index) for the arguments that
            do not get random data.

    Returns:
        The case.
    """
    picks = tuple((a.name, *chosen.get(a.name, (RANDOM, 0))) for a in ins)
    return Case(n, arr, seed, picks, scalars)


def fixed_cases(k: Kernel, seed: int = 1) -> Iterator[Case]:
    """Yield the fixed set of cases, identical on every run.

    For each size and each input argument, every value class is tried while the
    other arguments hold random data. Kernels with two inputs also get every
    pairing of plain classes at small and medium sizes.

    That full sweep runs with the default scalar values. Every further scalar
    set gets a thinner one: the same value classes at the same sizes (less the
    exact powers of two above 65,536), but one memory arrangement per case,
    three indices per positional class, and no pairings.
    A scalar chooses a path through the kernel; what has to be shown is that
    each path handles every size and every kind of value, not that it does so
    at every index a second time.
    """
    ins = k.ins
    for scalars in k.scalar_set_names:
        thin = scalars != DEFAULT_SCALARS
        for n in sizes(k.min_n, k.check_max_n):
            if thin and n > LARGE_MIN_N and n % 2 == 0:
                # Large sizes come in threes: a power of two, one less and one
                # more. Generating them is most of what a scalar set costs, so
                # the thin sweep keeps the two odd ones, which leave a tail.
                continue
            yield from _single_argument_cases(k, n, seed, scalars, thin)
            if len(ins) == 2 and n <= CROSS_PRODUCT_MAX_N and not thin:
                first, second = ins
                for c1 in classes(first.dtype)[0]:
                    for c2 in classes(second.dtype)[0]:
                        yield _case(ins, n, ARRANGEMENTS[n % 2], seed, scalars,
                                    {first.name: (c1, 0), second.name: (c2, 0)})


def fresh_cases(k: Kernel, seed: int, count: int = FRESH_CASE_COUNT) -> Iterator[Case]:
    """Yield random cases from a seed that changes on every run.

    These stop a kernel from passing merely by fitting the fixed cases. Each
    one draws its scalar set at random from all the kernel has.
    """
    rng = np.random.default_rng(seed)
    scalar_sets = k.scalar_set_names
    for _ in range(count):
        top = FRESH_LARGE_MAX_N if rng.random() < FRESH_LARGE_SHARE else FRESH_SMALL_MAX_N
        n = int(rng.integers(k.min_n, top))
        picks = []
        for a in k.ins:
            plain, positional = classes(a.dtype)
            # Positional classes need at least one element to place.
            pool = plain + (positional if n > 0 else [])
            cls = str(rng.choice(pool))
            p = int(rng.integers(0, n)) if cls in POSITIONAL else 0
            picks.append((a.name, cls, p))
        arr = str(rng.choice(ARRANGEMENTS))
        # A kernel with one scalar set draws nothing here, so its fresh cases
        # are the ones the same seed always gave.
        scalars = str(rng.choice(scalar_sets)) if len(scalar_sets) > 1 else DEFAULT_SCALARS
        yield Case(n, arr, seed, tuple(picks), scalars)
