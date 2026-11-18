"""The calls `delivery check` makes: every hooked operation, across element
types, sizes, memory layouts, array kinds, argument spellings and values.

A case is one Python call. It says how to build fresh arguments, how to make
the call, and whether a kernel should take it. The binding decides for every
call whether a kernel's conditions hold, so a case that must fall back to
NumPy is as much a test as one that must not.

Values come from the judge's value classes (`judge/cases.py`), so the binding
sees the same NaNs, infinities, signed zeros and ties the kernels were judged
on. Scalar arguments come from each kernel's own spec.
"""
from __future__ import annotations

import atexit
import itertools
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from judge import cases as judge_cases
from judge.spec import DEFAULT_SCALARS, Kernel, load_spec, np_dtype

ROOT = Path(__file__).resolve().parent.parent
KERNELS_DIR = ROOT / "kernels"

# Expectation meaning "a kernel may or may not take this call". NumPy
# normalises some inputs before a hook sees them (it copies a strided array
# before argmax, for example), and whether it does is NumPy's business. The
# results must match either way.
ANY = "*"

# Seed of all generated data. A case builds the same arguments every time.
SEED = 1

# Sizes for the sweep over values: every power of two a kernel changes path
# at, and its neighbours, up to a size no kernel treats as small.
VALUE_SIZES = (1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129,
               255, 256, 257, 511, 512, 513, 1000, 1023, 1024, 1025, 2047, 2048, 2049,
               4096, 4097, 8191, 8192, 8193, 20000, 100003)
# Sizes for the sweep over layouts and spellings: one below every threshold
# in the binding, one in the middle, and one above the size from which a hook
# releases the interpreter lock.
LAYOUT_SIZES = (16, 1000, 20000)
# A few sizes for sweeps that multiply by many scalar values.
FEW_SIZES = (1, 16, 100, 1000, 8193)
# Columns of the two-dimensional layouts.
COLUMNS = 2
# Bytes other than 1 that a boolean array can hold for True. NumPy accepts
# them; the mask kernels must never see them.
BAD_MASK_BYTES = (2, 255)
# Element count of the one-off cases that need a single ordinary size.
ORDINARY_N = 1000
# Shape of the three-dimensional layout of ORDINARY_N elements.
CUBE_SHAPE = (-1, 5, 4)
# Element count of a mask whose only bad byte is its last: more than two of
# the blocks in which the binding checks a mask before summing it.
SUM_BOOL_LATE_N = 40000
# Every how many bytes the self-overlapping mask case sets one to True.
OVERLAP_EVERY = 3

# Value classes whose sums can overflow, underflow or meet an infinity. A
# float cumsum kernel drops its output for NumPy's when that happens.
MAY_RAISE = frozenset({"denormal", "extremes", "inf_mix", "posinf_at", "neginf_at"})

# How a spelling of a call relates to its kernel.
TAKES = "takes"      # a plain array must be handled by the kernel
NEVER = "never"      # the kernel must not handle it, whatever the array
MAYBE = "maybe"      # NumPy decides; only the results are compared


@dataclass(frozen=True)
class Case:
    """One call, made once on plain NumPy and once with hotkernels installed."""

    # Operation the call belongs to: a key of the binding's pass counters.
    operation: str
    # What the call is, in enough words to find it again.
    label: str
    # Builds the arguments. Called afresh for each of the two runs and must
    # return equal arguments both times.
    make: Callable[[], tuple]
    # Makes the call on the built arguments and returns its result.
    call: Callable[..., object]
    # Name of the kernel that must take the call, None when none of the
    # operation's kernels may, or ANY.
    expect: str | None
    # Spec of the kernel when its result is compared within the spec's
    # tolerance; None for a bitwise comparison.
    tolerance: Kernel | None = None


@dataclass(frozen=True)
class Variant:
    """One way an array can be laid out or wrapped."""

    # Short name used in case labels.
    name: str
    # Builds the variant from a contiguous one-dimensional array.
    build: Callable[[np.ndarray], object]
    # Whether the result is what a kernel needs: exactly an ndarray, native
    # byte order, C-contiguous, at least one dimension.
    plain: bool
    # Number of dimensions of what `build` returns.
    dims: int = 1
    # Whether the result can be written to.
    writeable: bool = True
    # False for a wrapper whose own methods decide what reaches NumPy, so
    # that no expectation about kernels can be stated for it.
    predictable: bool = True
    # Smallest element count the variant can be built from.
    min_n: int = 0
    # False for a variant that is the same as the contiguous one when
    # elements are one byte wide.
    for_single_bytes: bool = True


@dataclass(frozen=True)
class Spelling:
    """One way of writing a call to an operation, taking the array(s) it works on."""

    # The call as text, for labels.
    text: str
    # Makes the call.
    call: Callable[..., object]
    # TAKES, NEVER or MAYBE.
    kind: str = TAKES
    # Element count from which the kernel takes the call; below it NumPy must.
    min_n: int = 0
    # True when NumPy may hand the hook a normalised copy of an array that is
    # not plain, so that such an array may still reach the kernel.
    normalises: bool = False
    # Whether a contiguous array of two or more dimensions is taken too.
    any_shape: bool = True
    # Whether the sweep over every value class at every size runs this
    # spelling. False for a spelling that reaches the same code as another
    # one after its arguments are read, which the layouts still cover.
    sweep_values: bool = True


class _Subclass(np.ndarray):
    """A do-nothing subclass of ndarray. The binding must leave it to NumPy."""


# Files behind the memory-mapped variants. Removed when the process exits.
_MEMMAP_DIR = tempfile.TemporaryDirectory(prefix="delivery-memmap-")
atexit.register(_MEMMAP_DIR.cleanup)
_memmap_counter = itertools.count()


def _strided(a: np.ndarray) -> np.ndarray:
    """Return `a`'s values as every second element of an array twice as long."""
    return np.repeat(a, 2)[::2]


def _reversed(a: np.ndarray) -> np.ndarray:
    """Return `a`'s values in order, stored back to front with a negative step."""
    return a[::-1].copy()[::-1]


def _misaligned(a: np.ndarray) -> np.ndarray:
    """Return a writable copy of `a` that starts one byte off any alignment."""
    buffer = bytearray(1 + a.nbytes)
    buffer[1:] = a.tobytes()
    return np.frombuffer(buffer, dtype=a.dtype, count=len(a), offset=1)


def _readonly(a: np.ndarray) -> np.ndarray:
    """Return a copy of `a` that refuses writes."""
    b = a.copy()
    b.flags.writeable = False
    return b


def _byteswapped(a: np.ndarray) -> np.ndarray:
    """Return `a`'s values stored in the other byte order."""
    return a.astype(a.dtype.newbyteorder())


def _memmap(a: np.ndarray) -> np.ndarray:
    """Return `a`'s values in a memory-mapped file."""
    path = Path(_MEMMAP_DIR.name) / f"{next(_memmap_counter)}.bin"
    mapped = np.memmap(path, dtype=a.dtype, mode="w+", shape=a.shape)
    mapped[:] = a
    return mapped


def _rows(a: np.ndarray) -> np.ndarray:
    """Return the first whole rows of `a` as a C-ordered two-dimensional array."""
    return a[: len(a) // COLUMNS * COLUMNS].reshape(-1, COLUMNS).copy()


VARIANTS: tuple[Variant, ...] = (
    Variant("contiguous", np.copy, plain=True),
    Variant("strided", _strided, plain=False),
    Variant("reversed", _reversed, plain=False),
    Variant("misaligned", _misaligned, plain=True),
    Variant("readonly", _readonly, plain=True, writeable=False),
    Variant("byteswapped", _byteswapped, plain=False, for_single_bytes=False),
    Variant("subclass", lambda a: a.copy().view(_Subclass), plain=False),
    Variant("masked_array", lambda a: np.ma.array(a.copy()), plain=False, predictable=False),
    Variant("memmap", _memmap, plain=False, min_n=1),
    Variant("list", lambda a: a.tolist(), plain=False),
    Variant("rows", _rows, plain=True, dims=2, min_n=2 * COLUMNS),
    Variant("fortran_rows", lambda a: np.asfortranarray(_rows(a)), plain=False, dims=2, min_n=2 * COLUMNS),
    Variant("zero_d", lambda a: np.array(a[0]), plain=False, dims=0, min_n=1),
)
CONTIGUOUS = VARIANTS[0]


def spec(kernel: str) -> Kernel:
    """Load the spec of a kernel by name."""
    return load_spec(KERNELS_DIR / kernel)


def values(dtype: str, n: int, cls: str = judge_cases.RANDOM, position: int = 0, arg: str = "x") -> np.ndarray:
    """Build one array from a value class of the judge.

    Args:
        dtype: Element type as a spec name, such as "f64".
        n: Element count.
        cls: Value class name, such as "nan_some".
        position: Index of the interesting element, for positional classes.
        arg: Argument name. Different names give different random data.

    Returns:
        A contiguous one-dimensional array. The same arguments always give
        the same array.
    """
    base = judge_cases.base(dtype, n, SEED, arg)
    return judge_cases.make(dtype, cls, position % n if n else 0, n, SEED, arg, base)


def value_arrays(dtype: str, sizes: tuple[int, ...] = VALUE_SIZES, arg: str = "x") -> Iterator[tuple[int, str, str, Callable[[], np.ndarray]]]:
    """Yield every value class of an element type at every size.

    Args:
        dtype: Element type as a spec name.
        sizes: Element counts to build.
        arg: Argument name, used in labels and to vary the random data.

    Yields:
        (element count, value class, label, function that builds the array).
        Positional classes are built with their element first, in the middle
        and last.
    """
    plain, positional = judge_cases.classes(dtype)
    for n in sizes:
        for cls in plain:
            yield n, cls, f"n={n} {arg}={cls}", partial(values, dtype, n, cls, 0, arg)
        for cls in positional:
            for position in sorted({0, n // 2, n - 1}):
                yield n, cls, f"n={n} {arg}={cls}@{position}", partial(values, dtype, n, cls, position, arg)


def variants_for(dtype: str, n: int) -> Iterator[Variant]:
    """Yield the variants that can be built from `n` elements of `dtype`."""
    single_bytes = np_dtype(dtype).itemsize == 1
    for variant in VARIANTS:
        if n >= variant.min_n and (variant.for_single_bytes or not single_bytes):
            yield variant


def _built(dtype: str, n: int, variant: Variant, arg: str = "x") -> tuple:
    """Build the one argument of a unary case: random values in a variant."""
    return (variant.build(values(dtype, n, arg=arg)),)


def _made(maker: Callable[[], np.ndarray]) -> tuple:
    """Build the one argument of a unary case from an array maker."""
    return (maker(),)


def expectation(kernel: str | None, spelling: Spelling, variant: Variant, n: int) -> str | None:
    """Say whether a kernel should take a call.

    Args:
        kernel: The kernel for the array's element type, or None if the
            operation has none for it.
        spelling: How the call is written.
        variant: How the array is laid out.
        n: Element count.

    Returns:
        The kernel name, None, or ANY.
    """
    # A list has no element type of its own: NumPy converts it, and the
    # result may be an array some kernel takes.
    if variant.name == "list" and spelling.normalises and spelling.kind != NEVER:
        return ANY
    if kernel is None or spelling.kind == NEVER:
        return None
    if spelling.kind == MAYBE or not variant.predictable:
        return ANY
    shape_ok = variant.dims == 1 or (variant.dims > 1 and spelling.any_shape)
    if variant.plain and shape_ok:
        return kernel if n >= spelling.min_n else None
    # NumPy flags every empty array as contiguous, whatever its strides say.
    return ANY if spelling.normalises or n == 0 else None


def unary_cases(operation: str, kernels: dict[str, str], neighbours: tuple[str, ...],
                spellings: tuple[Spelling, ...], empty_takes: bool = True,
                tolerance: bool = False) -> Iterator[Case]:
    """Yield the cases of an operation that takes one array.

    Three sweeps: every value class at every size on a plain array, for the
    spellings that take the kernel; every layout and every spelling at a few
    sizes, for the kernel's element types and their neighbours; and an empty
    array.

    Args:
        operation: Operation name.
        kernels: Element type (spec name) -> kernel name.
        neighbours: Element types the operation has no kernel for.
        spellings: The ways of writing the call.
        empty_takes: Whether the kernel takes an empty plain array.
        tolerance: Whether results are compared within the kernel's tolerance.

    Yields:
        The cases.
    """
    for dtype, kernel in kernels.items():
        within = spec(kernel) if tolerance else None
        for spelling in spellings:
            if spelling.kind != TAKES or not spelling.sweep_values:
                continue
            for n, cls, label, maker in value_arrays(dtype):
                expect = expectation(kernel, spelling, CONTIGUOUS, n)
                # A kernel compared within a tolerance hands a call back to
                # NumPy when its arithmetic raises a floating-point exception.
                if tolerance and cls in MAY_RAISE:
                    expect = ANY
                yield Case(operation, f"{spelling.text} {dtype} {label}", partial(_made, maker), spelling.call,
                           expect, within)
    for dtype in (*kernels, *neighbours):
        kernel = kernels.get(dtype)
        within = spec(kernel) if tolerance and kernel else None
        for n, spelling in itertools.product(LAYOUT_SIZES, spellings):
            for variant in variants_for(dtype, n):
                yield Case(operation, f"{spelling.text} {dtype} n={n} layout={variant.name}",
                           partial(_built, dtype, n, variant), spelling.call,
                           expectation(kernel, spelling, variant, n), within)
        for spelling in spellings:
            expect = expectation(kernel, spelling, CONTIGUOUS, 0) if empty_takes else None
            yield Case(operation, f"{spelling.text} {dtype} n=0", partial(_built, dtype, 0, CONTIGUOUS),
                       spelling.call, expect, within)


# ------------------------------------------------------------- operations

# Bit patterns of NaNs the judge's value classes do not produce: a signalling
# NaN, and a negative quiet NaN with a payload, per element size in bytes.
# Operations that copy or compare elements must keep them bit for bit, and
# the clip kernels treat signalling NaNs on a path of their own.
_UNUSUAL_NANS = {
    4: (np.uint32, 0x7FA00001, 0xFFC00123),
    8: (np.uint64, 0x7FF4000000000001, 0xFFF8000000000123),
}
# Every how many elements each of the two patterns is placed.
_SIGNALLING_EVERY = 5
_NEGATIVE_EVERY = 7


def unusual_nans(dtype: str, n: int) -> np.ndarray:
    """Build random floats with signalling and negative NaNs spread through them.

    Args:
        dtype: "f32" or "f64".
        n: Element count.

    Returns:
        A contiguous array whose every fifth element is a signalling NaN and
        every seventh, from the third, a negative quiet NaN with a payload.
    """
    x = values(dtype, n)
    unsigned, signalling, negative = _UNUSUAL_NANS[x.dtype.itemsize]
    bits = x.view(unsigned)
    bits[::_SIGNALLING_EVERY] = signalling
    bits[2::_NEGATIVE_EVERY] = negative
    return x


def unusual_nan_cases(operation: str, kernels: dict[str, str], text: str, call: Callable,
                      min_n: int = 0) -> Iterator[Case]:
    """Yield one case per size on data that holds signalling and negative NaNs.

    Args:
        operation: Operation name.
        kernels: Element type ("f32", "f64") -> kernel name.
        text: The call as text, for labels.
        call: Makes the call on the array.
        min_n: Element count from which the kernel takes the call.

    Yields:
        The cases.
    """
    for dtype, kernel in kernels.items():
        for n in VALUE_SIZES:
            yield Case(operation, f"{text} {dtype} n={n} x=signalling and negative NaNs",
                       partial(_made, partial(unusual_nans, dtype, n)), call, kernel if n >= min_n else None)


def argext_cases(name: str) -> Iterator[Case]:
    """Yield the cases of `argmax` or `argmin`.

    Args:
        name: "argmax" or "argmin".

    Yields:
        The cases. The function hook takes `np.argmax(x)` and the method hook
        `x.argmax()` and its spellings with every argument None, on a plain
        array of any size; NumPy's wrapper sends `np.argmax(x, axis=None)`
        to the method. The type slot behind every other spelling takes runs
        from 128 elements (HK_SLOT_MIN_N in the binding), and NumPy may
        copy an array into shape before the slot sees it.
    """
    function, nan_function = getattr(np, name), getattr(np, f"nan{name}")

    def method(*args, **kwargs) -> Callable[[object], object]:
        """Return a call of the method on its one argument, with these arguments.

        Args:
            *args: Positional arguments of the method call.
            **kwargs: Keyword arguments of the method call.

        Returns:
            A function of the array that makes the call.
        """
        return lambda x: getattr(x, name)(*args, **kwargs)

    spellings = (
        Spelling(f"np.{name}(x)", function, normalises=True),
        Spelling(f"x.{name}()", method(), normalises=True),
        Spelling(f"x.{name}(None)", method(None), normalises=True, sweep_values=False),
        Spelling(f"x.{name}(axis=None, out=None)", method(axis=None, out=None), normalises=True, sweep_values=False),
        Spelling(f"np.{name}(x, axis=None)", lambda x: function(x, axis=None), normalises=True, sweep_values=False),
        Spelling(f"np.{name}(a=x)", lambda x: function(a=x), normalises=True, sweep_values=False),
        Spelling(f"np.{name}(x, axis=0)", lambda x: function(x, axis=0), MAYBE),
        Spelling(f"np.{name}(x, axis=-1)", lambda x: function(x, axis=-1), MAYBE),
        Spelling(f"x.{name}(0)", method(0), MAYBE),
        Spelling(f"np.{name}(x, keepdims=True)", lambda x: function(x, keepdims=True), MAYBE),
        Spelling(f"x.{name}(keepdims=True)", method(keepdims=True), MAYBE),
        Spelling(f"x.{name}(keepdims=False)", method(keepdims=False), MAYBE),
        Spelling(f"np.{name}(x, out=o)", lambda x: function(x, out=np.empty((), np.intp)), MAYBE),
        Spelling(f"x.{name}(out=o)", lambda x: getattr(x, name)(out=np.empty((), np.intp)), MAYBE),
        Spelling(f"np.nan{name}(x)", nan_function, MAYBE),
        Spelling(f"x.{name}(bogus=1)", method(bogus=1), NEVER),
        Spelling(f"x.{name}(None, None, None, None)", method(None, None, None, None), NEVER),
    )
    kernels = {"f32": f"{name}_f32", "f64": f"{name}_f64"}
    # NumPy raises on an empty array, so no kernel takes one.
    yield from unary_cases(name, kernels, ("i64", "i32", "u8"), spellings, empty_takes=False)
    yield from unusual_nan_cases(name, kernels, f"np.{name}(x)", function)
    yield from unusual_nan_cases(name, kernels, f"x.{name}()", spellings[1].call)


# Layouts NumPy passes to the clip loop as they are, so that an in-place
# call arrives with the output equal to the input.
IN_PLACE_UNBUFFERED = frozenset({"contiguous", "rows", "readonly"})


def _clip_bounds(kernel: str) -> Iterator[tuple[str, float, float]]:
    """Yield every pair of bounds a clip kernel's spec lists, default first."""
    k = spec(kernel)
    for set_name in k.scalar_set_names:
        scalars = k.scalar_values(set_name)
        yield set_name, scalars["lo"], scalars["hi"]


def _clip_in_place(x, lo, hi):
    """Clip `x` into itself and return it."""
    return np.clip(x, lo, hi, out=x)


def _clip_method_in_place(x, lo, hi):
    """Clip `x` into itself with the method and return it."""
    return x.clip(lo, hi, out=x)


def _clip_into_fresh(x, lo, hi):
    """Clip `x` into a new array passed as `out`."""
    return np.clip(x, lo, hi, out=np.empty_like(x))


def _clip_where(x, lo, hi):
    """Clip `x` into an array of zeros where `x` is positive, and return that array."""
    return np.clip(x, lo, hi, out=np.zeros_like(x), where=np.greater(x, 0))


# Bounds a float32 array cannot hold exactly, which NumPy rounds to float32.
CLIP_ROUNDED = (0.1, 0.9)
# Python ints as bounds, which NumPy converts to the array's element type.
CLIP_INTS = (-1, 1)
# A lower bound above the upper one as doubles that rounds to the same
# float32: reversed for a float64 array, equal bounds for a float32 one.
CLIP_EQUAL_IN_F32 = (0.1 + 1e-9, 0.1)
# A Python int a float32 cannot hold, one beyond what a double holds exactly,
# and one beyond the range of a float32. NumPy rounds all three, and warns
# for the last on a float32 array.
CLIP_INT_NOT_F32, CLIP_INT_NOT_F64, CLIP_INT_TOO_LARGE = 2**24 + 1, 2**60, 2**200


def _clip_bound_cases(dtype: str, kernel: str, default: tuple[str, float, float]) -> Iterator[Case]:
    """Yield the sweep over values for bounds of each kind NumPy converts.

    Args:
        dtype: "f32" or "f64".
        kernel: The array's clip kernel.
        default: The spec's default set, as (name, lo, hi).

    Yields:
        One case per value class at FEW_SIZES for each kind of bound: the
        method with the default bounds, Python floats a float32 cannot hold,
        Python ints, NumPy scalars of the array's type, and, for float32,
        bounds that are reversed as doubles and equal as float32. For float64
        the same bounds are reversed, and no kernel may take the call.
    """
    scalar = np_dtype(dtype).type
    same_type = (scalar(default[1]), scalar(default[2]))
    for _, _, label, maker in value_arrays(dtype, FEW_SIZES):
        yield Case("clip", f"x.clip(lo, hi) {dtype} {label}", partial(_made, maker),
                   partial(_clip_method, default[1], default[2]), kernel)
        for lo, hi in (CLIP_ROUNDED, CLIP_INTS, same_type):
            yield Case("clip", f"np.clip(x, {lo!r}, {hi!r}) {dtype} {label}", partial(_made, maker),
                       partial(_clip3, np.clip, lo, hi), kernel)
        lo, hi = CLIP_EQUAL_IN_F32
        if dtype == "f32":
            yield Case("clip", f"np.clip(x, {lo!r}, {hi!r}) {dtype} {label}", partial(_made, maker),
                       partial(_clip3, np.clip, lo, hi), kernel)
    # Reversed for float64. Built from plain random values: the makers of
    # some value classes call np.clip themselves, which a kernel takes.
    if dtype == "f64":
        lo, hi = CLIP_EQUAL_IN_F32
        for n in FEW_SIZES:
            yield Case("clip", f"np.clip(x, {lo!r}, {hi!r}) {dtype} n={n}", partial(_built, dtype, n, CONTIGUOUS),
                       partial(_clip3, np.clip, lo, hi), None)


def clip_cases() -> Iterator[Case]:
    """Yield the cases of `clip`.

    Three hooks serve clip. The function hook on `np.clip` and the method
    hook on `x.clip` call the kernel directly for a contiguous float32 or
    float64 array with no `out` and two bounds that keep the result in the
    array's element type (Python floats and ints, NumPy scalars of the
    array's type), lo <= hi. Every other call goes on to the `clip` ufunc,
    whose loop hook takes contiguous data with scalar bounds, lo <= hi, when
    the output is not the input. NumPy buffers arrays that are misaligned or
    in the other byte order and hands the loop contiguous chunks, so those
    may reach the kernel too. The counters do not say which of the hooks took
    a call; `installation_checks` proves the direct path on its own.
    """
    kernels = {"f32": "clip_f32", "f64": "clip_f64"}
    for dtype, kernel in kernels.items():
        default = next(_clip_bounds(kernel))
        for set_name, lo, hi in _clip_bounds(kernel):
            sizes = VALUE_SIZES if set_name in (DEFAULT_SCALARS, "zero_lo") else FEW_SIZES
            for _, _, label, maker in value_arrays(dtype, sizes):
                yield Case("clip", f"np.clip(x, {lo!r}, {hi!r}) {dtype} {label}", partial(_made, maker),
                           partial(_clip3, np.clip, lo, hi), kernel)
        yield from _clip_bound_cases(dtype, kernel, default)
    yield from unusual_nan_cases("clip", kernels, "np.clip(x, -0.5, 0.5)", partial(_clip3, np.clip, -0.5, 0.5))
    yield from unusual_nan_cases("clip", kernels, "np.clip(x, 0.0, 1.0)", partial(_clip3, np.clip, 0.0, 1.0))
    yield from unusual_nan_cases("clip", kernels, "x.clip(-0.5, 0.5)", partial(_clip_method, -0.5, 0.5))
    lo, hi, nan = -0.5, 0.5, float("nan")
    spellings = (
        Spelling("np.clip(x, lo, hi)", partial(_clip3, np.clip, lo, hi), normalises=True),
        Spelling("x.clip(lo, hi)", partial(_clip_method, lo, hi), normalises=True),
        Spelling("np.clip(x, min=lo, max=hi)", lambda x: np.clip(x, min=lo, max=hi), normalises=True),
        Spelling("np.clip(x, max=hi, min=lo)", lambda x: np.clip(x, max=hi, min=lo), normalises=True),
        Spelling("np.clip(x, a_min=lo, a_max=hi)", lambda x: np.clip(x, a_min=lo, a_max=hi), normalises=True),
        Spelling("np.clip(a=x, a_min=lo, a_max=hi)", lambda x: np.clip(a=x, a_min=lo, a_max=hi), normalises=True),
        Spelling("np.clip(x, lo, hi, None)", lambda x: np.clip(x, lo, hi, None), normalises=True),
        Spelling("np.clip(x, lo, hi, out=None)", lambda x: np.clip(x, lo, hi, out=None), normalises=True),
        Spelling("x.clip(min=lo, max=hi)", lambda x: x.clip(min=lo, max=hi), normalises=True),
        Spelling("x.clip(lo, hi, None)", lambda x: x.clip(lo, hi, None), normalises=True),
        Spelling("x.clip(lo, hi, out=None)", lambda x: x.clip(lo, hi, out=None), normalises=True),
        Spelling("np.clip(x, -1, 1)", lambda x: np.clip(x, -1, 1), normalises=True),
        Spelling("np.clip(x, 0.1, 0.9)", lambda x: np.clip(x, 0.1, 0.9), normalises=True),
        Spelling("np.clip(x, -inf, inf)", lambda x: np.clip(x, -np.inf, np.inf), normalises=True),
        Spelling("np.clip(x, lo, inf)", lambda x: np.clip(x, lo, np.inf), normalises=True),
        Spelling("np.clip(x, -1e300, 1e300)", lambda x: np.clip(x, -1e300, 1e300), normalises=True),
        Spelling("np.clip(x, -1e-300, 1e-300)", lambda x: np.clip(x, -1e-300, 1e-300), normalises=True),
        Spelling("np.clip(x, -1e-40, 1e-40)", lambda x: np.clip(x, -1e-40, 1e-40), normalises=True),
        Spelling("np.clip(x, 2**24 + 1, 2**25 + 2)", lambda x: np.clip(x, CLIP_INT_NOT_F32, 2 * CLIP_INT_NOT_F32), normalises=True),
        Spelling("np.clip(x, -2**60, 2**60)", lambda x: np.clip(x, -CLIP_INT_NOT_F64, CLIP_INT_NOT_F64), normalises=True),
        Spelling("np.clip(x, -2**200, 2**200)", lambda x: np.clip(x, -CLIP_INT_TOO_LARGE, CLIP_INT_TOO_LARGE), normalises=True),
        Spelling("np.clip(x, False, True)", lambda x: np.clip(x, False, True), normalises=True),
        Spelling("np.clip(x, np.float32(lo), np.float32(hi))", lambda x: np.clip(x, np.float32(lo), np.float32(hi)), normalises=True),
        Spelling("np.clip(x, lo, np.float32(hi))", lambda x: np.clip(x, lo, np.float32(hi)), normalises=True),
        Spelling("x.clip(np.float32(lo), np.float32(hi))", lambda x: x.clip(np.float32(lo), np.float32(hi)), normalises=True),
        Spelling("np.clip(x, lo, hi, out=fresh)", lambda x: _clip_into_fresh(x, lo, hi), normalises=True),
        Spelling("np.clip(x, np.float64(lo), np.float64(hi))", lambda x: np.clip(x, np.float64(lo), np.float64(hi)), MAYBE),
        Spelling("np.clip(x, lo, np.float64(hi))", lambda x: np.clip(x, lo, np.float64(hi)), MAYBE),
        Spelling("x.clip(np.float64(lo), np.float64(hi))", lambda x: x.clip(np.float64(lo), np.float64(hi)), MAYBE),
        Spelling("np.clip(x, np.array(lo), np.array(hi))", lambda x: np.clip(x, np.array(lo), np.array(hi)), MAYBE),
        Spelling("np.clip(x, lo_array, hi_array)", lambda x: np.clip(x, np.full(np.shape(x), lo), np.full(np.shape(x), hi)), MAYBE),
        Spelling("np.clip(x, lo, hi, dtype=np.float64)", lambda x: np.clip(x, lo, hi, dtype=np.float64), MAYBE),
        Spelling("np.clip(x, lo, hi, casting='unsafe')", lambda x: np.clip(x, lo, hi, casting="unsafe"), MAYBE),
        Spelling("np.clip(x, lo, hi, out=zeros, where=x > 0)", lambda x: _clip_where(x, lo, hi), MAYBE),
        Spelling("x.clip(lo, hi, order='F')", lambda x: x.clip(lo, hi, order="F"), MAYBE),
        Spelling("np.clip(x, lo, hi, out=x)", lambda x: _clip_in_place(x, lo, hi), NEVER, normalises=True),
        Spelling("x.clip(lo, hi, out=x)", lambda x: _clip_method_in_place(x, lo, hi), NEVER, normalises=True),
        Spelling("np.clip(x, lo, None)", lambda x: np.clip(x, lo, None), NEVER),
        Spelling("np.clip(x, None, hi)", lambda x: np.clip(x, None, hi), NEVER),
        Spelling("np.clip(x, min=lo)", lambda x: np.clip(x, min=lo), NEVER),
        Spelling("np.clip(x, max=hi)", lambda x: np.clip(x, max=hi), NEVER),
        Spelling("np.clip(x, lo)", lambda x: np.clip(x, lo), NEVER),
        Spelling("np.clip(x, a_min=lo, max=hi)", lambda x: np.clip(x, a_min=lo, max=hi), NEVER),
        Spelling("np.clip(x, lo, hi, min=lo)", lambda x: np.clip(x, lo, hi, min=lo), NEVER),
        Spelling("np.clip(x, lo, hi, bogus=1)", lambda x: np.clip(x, lo, hi, bogus=1), NEVER),
        Spelling("np.clip(x, lo, hi, None, None)", lambda x: np.clip(x, lo, hi, None, None), NEVER),
        Spelling("x.clip(lo)", lambda x: x.clip(lo), NEVER),
        Spelling("x.clip(max=hi)", lambda x: x.clip(max=hi), NEVER),
        Spelling("x.clip()", lambda x: x.clip(), NEVER),
        Spelling("x.clip(lo, hi, None, None)", lambda x: x.clip(lo, hi, None, None), NEVER),
        Spelling("x.clip(a_min=lo, a_max=hi)", lambda x: x.clip(a_min=lo, a_max=hi), NEVER),
        Spelling("np.clip(x, hi, lo) reversed", lambda x: np.clip(x, hi, lo), NEVER),
        Spelling("x.clip(hi, lo) reversed", lambda x: x.clip(hi, lo), NEVER),
        Spelling("np.clip(x, nan, hi)", lambda x: np.clip(x, nan, hi), NEVER),
        Spelling("np.clip(x, lo, nan)", lambda x: np.clip(x, lo, nan), NEVER),
        Spelling("x.clip(nan, hi)", lambda x: x.clip(nan, hi), NEVER),
        Spelling("np.clip(x, np.float32(nan), np.float32(hi))", lambda x: np.clip(x, np.float32(nan), np.float32(hi)), NEVER),
    )
    for dtype in ("f32", "f64", "i64"):
        kernel = kernels.get(dtype)
        for n, spelling in itertools.product((0, *LAYOUT_SIZES), spellings):
            for variant in variants_for(dtype, n):
                expect = expectation(kernel, spelling, variant, n)
                # An in-place call on an array NumPy has to buffer reaches the
                # loop with separate buffers, which the kernel may take.
                if spelling.kind == NEVER and spelling.normalises and kernel and variant.name not in IN_PLACE_UNBUFFERED:
                    expect = ANY
                # NumPy converts integers to float64 for float bounds, and the
                # float64 loop may then take the converted data.
                if dtype == "i64" and spelling.kind != NEVER:
                    expect = ANY
                # The loop is not called at all for an empty array.
                if n == 0 and expect == kernel:
                    expect = None
                yield Case("clip", f"{spelling.text} {dtype} n={n} layout={variant.name}",
                           partial(_built, dtype, n, variant), spelling.call, expect)


def _clip3(function, lo, hi, x):
    """Call `function(x, lo, hi)`."""
    return function(x, lo, hi)


def _clip_method(lo, hi, x):
    """Call `x.clip(lo, hi)`."""
    return x.clip(lo, hi)


def _bad_mask(n: int, byte: int) -> np.ndarray:
    """Return a boolean array whose True elements are stored as `byte`, not 1.

    NumPy treats any non-zero byte as True. The mask kernels do not, so the
    binding must keep such a mask away from them.
    """
    raw = values("bool", n, arg="mask").astype(np.uint8) * byte
    return raw.view(np.bool_)


def _x_and_mask(dtype: str, n: int, x_variant: Variant, mask_maker: Callable[[], object]) -> tuple:
    """Build the two arguments of a mask case."""
    return (x_variant.build(values(dtype, n)), mask_maker())


def _mask_makers(n: int) -> Iterator[tuple[str, Callable[[], np.ndarray]]]:
    """Yield every boolean value class at one size, as (label, maker)."""
    for _, _, label, maker in value_arrays("bool", (n,), arg="mask"):
        yield label.split(" ", 1)[1], maker


def _compress_into(x, mask):
    """Run `np.compress(mask, x, out=o)` into an array of exactly the right size and return `o`."""
    out = np.empty(np.count_nonzero(mask), np.asarray(x).dtype)
    return np.compress(mask, x, out=out)


def _reshaped(maker: Callable[[], np.ndarray], order: str = "C") -> Callable[[], np.ndarray]:
    """Return a maker of the same values as `maker`'s, in COLUMNS columns of the given order.

    Args:
        maker: Builds a one-dimensional array whose length COLUMNS divides.
        order: "C" or "F", the memory order of the two-dimensional result.

    Returns:
        A function that builds the two-dimensional array.
    """
    return lambda: np.asarray(maker().reshape(-1, COLUMNS), order=order)


def _cube(maker: Callable[[], np.ndarray]) -> Callable[[], np.ndarray]:
    """Return a maker of the same values as `maker`'s, in three dimensions.

    Args:
        maker: Builds a one-dimensional array whose length CUBE_SHAPE's
            last two lengths divide.

    Returns:
        A function that builds the three-dimensional array, in C order.
    """
    return lambda: maker().reshape(CUBE_SHAPE).copy()


def _compress_shape_cases(dtype: str, kernel: str | None) -> Iterator[Case]:
    """Yield the cases of `np.compress` and `np.extract` with arrays of two dimensions.

    Without an axis both flatten `x` in C order, so a contiguous `x` of any
    shape is taken; `np.extract` flattens the mask too, and `np.compress`
    raises for one of two dimensions.

    Args:
        dtype: Element type of `x`, as a spec name.
        kernel: The compress kernel for that type, or None.

    Yields:
        The cases.
    """
    n = ORDINARY_N
    x_flat = partial(values, dtype, n)
    mask_flat = partial(values, "bool", n, arg="mask")
    shapes = (
        ("x two-dimensional", _reshaped(x_flat), mask_flat, kernel, kernel),
        ("x two-dimensional, Fortran order", _reshaped(x_flat, "F"), mask_flat, None, None),
        ("both two-dimensional", _reshaped(x_flat), _reshaped(mask_flat), None, kernel),
        ("mask two-dimensional, Fortran order", x_flat, _reshaped(mask_flat, "F"), None, None),
        ("x zero-dimensional", lambda: np.array(x_flat()[0]), lambda: np.array(True), None, None),
    )
    for text, x_maker, mask_maker, by_compress, by_extract in shapes:
        make = lambda x_maker=x_maker, mask_maker=mask_maker: (x_maker(), mask_maker())
        yield Case("compress", f"np.compress(mask, x) {dtype} {text}", make, _compress_call, by_compress)
        yield Case("compress", f"np.extract(mask, x) {dtype} {text}", make, _extract_call, by_extract)


def _select_shape_cases(dtype: str, kernel: str | None) -> Iterator[Case]:
    """Yield the cases of `x[mask]` with arrays of more than one dimension.

    A mask of the shape of `x` selects elements in C order, which the kernel
    takes for contiguous arrays. A mask of fewer dimensions selects rows, and
    one of the same size and another shape is an error.

    Args:
        dtype: Element type of `x`, as a spec name.
        kernel: The compress kernel for that type, or None.

    Yields:
        The cases.
    """
    n = ORDINARY_N
    x_flat = partial(values, dtype, n)
    mask_flat = partial(values, "bool", n, arg="mask")
    row_mask = partial(values, "bool", n // COLUMNS, arg="mask")
    shapes = (
        ("both two-dimensional", _reshaped(x_flat), _reshaped(mask_flat), kernel),
        ("both three-dimensional", _cube(x_flat), _cube(mask_flat), kernel),
        ("both two-dimensional, Fortran order", _reshaped(x_flat, "F"), _reshaped(mask_flat, "F"), None),
        ("x two-dimensional, mask over its rows", _reshaped(x_flat), row_mask, None),
        ("x two-dimensional, mask of the same size and one dimension", _reshaped(x_flat), mask_flat, None),
        ("both two-dimensional, shapes transposed", _reshaped(x_flat), lambda: mask_flat().reshape(COLUMNS, -1), None),
        ("both empty and two-dimensional", lambda: np.zeros((0, COLUMNS), np_dtype(dtype)),
         lambda: np.zeros((0, COLUMNS), np.bool_), kernel),
    )
    for text, x_maker, mask_maker, expect in shapes:
        make = lambda x_maker=x_maker, mask_maker=mask_maker: (x_maker(), mask_maker())
        yield Case("compress", f"x[mask] {dtype} {text}", make, lambda x, mask: x[mask], expect)


def _compress_call(x, mask):
    """Return `np.compress(mask, x)`."""
    return np.compress(mask, x)


def _extract_call(x, mask):
    """Return `np.extract(mask, x)`."""
    return np.extract(mask, x)


# Value classes of `x` in the sweep of mask selection over every size and
# mask: the ones whose bit patterns a kernel that computed on its elements
# would change.
_COMPRESS_VALUE_CLASSES = {"f32": ("random", "nan_some"), "f64": ("random", "nan_some"), "i64": ("random", "extremes")}


def compress_cases() -> Iterator[Case]:
    """Yield the cases of `x[mask]`, `np.compress(mask, x)` and `np.extract(mask, x)`.

    int64 elements are selected by the float64 kernel, which moves 64-bit
    elements without computing on them.
    """
    kernels = {"f32": "compress_f32", "f64": "compress_f64", "i64": "compress_f64"}
    select = Spelling("x[mask]", lambda x, mask: x[mask], any_shape=False)
    functions = (
        Spelling("np.compress(mask, x)", _compress_call),
        Spelling("np.extract(mask, x)", _extract_call),
    )
    others = (
        *functions,
        Spelling("np.compress(mask, x, axis=None, out=None)", lambda x, mask: np.compress(mask, x, axis=None, out=None)),
        Spelling("np.compress(condition=mask, a=x)", lambda x, mask: np.compress(condition=mask, a=x)),
        Spelling("np.extract(condition=mask, arr=x)", lambda x, mask: np.extract(condition=mask, arr=x)),
        Spelling("np.compress(mask, x, axis=0)", lambda x, mask: np.compress(mask, x, axis=0), NEVER),
        Spelling("np.compress(mask, x, out=o)", _compress_into, NEVER),
        Spelling("np.compress(mask.astype(int), x)", lambda x, mask: np.compress(np.asarray(mask).astype(int), x), NEVER),
        Spelling("np.extract(mask.astype(int), x)", lambda x, mask: np.extract(np.asarray(mask).astype(int), x), NEVER),
        Spelling("x[mask,]", lambda x, mask: x[mask,], NEVER),
        Spelling("x[mask][:3]", lambda x, mask: x[mask][:3], any_shape=False),
    )
    for dtype, kernel in kernels.items():
        for n in VALUE_SIZES:
            for mask_label, mask_maker in _mask_makers(n):
                for x_class in _COMPRESS_VALUE_CLASSES[dtype]:
                    make = partial(_values_and_mask, dtype, n, x_class, mask_maker)
                    yield Case("compress", f"x[mask] {dtype} n={n} x={x_class} {mask_label}", make, select.call, kernel)
                for spelling in functions:
                    make = partial(_values_and_mask, dtype, n, "random", mask_maker)
                    yield Case("compress", f"{spelling.text} {dtype} n={n} {mask_label}", make, spelling.call, kernel)
    for dtype in ("f32", "f64", "i64", "u64", "i32", "u8"):
        kernel = kernels.get(dtype)
        for n in (0, *LAYOUT_SIZES):
            plain_mask = partial(values, "bool", n, arg="mask")
            for spelling in (select, *others):
                for variant in variants_for(dtype, n):
                    if variant.dims != 1:
                        continue
                    yield Case("compress", f"{spelling.text} {dtype} n={n} x={variant.name}",
                               partial(_x_and_mask, dtype, n, variant, plain_mask), spelling.call,
                               expectation(kernel, spelling, variant, n))
                for variant in variants_for("bool", n):
                    if variant.dims != 1:
                        continue
                    mask_maker = partial(_variant_of, "bool", n, variant, "mask")
                    yield Case("compress", f"{spelling.text} {dtype} n={n} mask={variant.name}",
                               partial(_x_and_mask, dtype, n, CONTIGUOUS, mask_maker), spelling.call,
                               expectation(kernel, spelling, variant, n))
            if n:
                for spelling in (select, *functions):
                    for byte in BAD_MASK_BYTES:
                        yield Case("compress", f"{spelling.text} {dtype} n={n} mask bytes are {byte}",
                                   partial(_x_and_mask, dtype, n, CONTIGUOUS, partial(_bad_mask, n, byte)),
                                   spelling.call, None)
                    for change, text in ((-1, "one short"), (1, "one long")):
                        other = partial(values, "bool", n + change, arg="mask")
                        yield Case("compress", f"{spelling.text} {dtype} n={n} mask {text}",
                                   partial(_x_and_mask, dtype, n, CONTIGUOUS, other), spelling.call, None)
        yield from _select_shape_cases(dtype, kernel)
        yield from _compress_shape_cases(dtype, kernel)


def _values_and_mask(dtype: str, n: int, x_class: str, mask_maker: Callable[[], np.ndarray]) -> tuple:
    """Build `x` from a value class and a mask from a maker."""
    return (values(dtype, n, x_class), mask_maker())


def _variant_of(dtype: str, n: int, variant: Variant, arg: str) -> object:
    """Build random values of `dtype` in a variant."""
    return variant.build(values(dtype, n, arg=arg))


def _assign(value, x, mask):
    """Run `x[mask] = value` and return `x`."""
    x[mask] = value
    return x


def _putmask(value, x, mask):
    """Run `np.putmask(x, mask, value)` and return `x`."""
    np.putmask(x, mask, value)
    return x


def _copyto(value, x, mask):
    """Run `np.copyto(x, value, where=mask)` and return `x`."""
    np.copyto(x, value, where=mask)
    return x


def _overlapping(n: int) -> tuple:
    """Build an array and a boolean mask that is a view of the array's own bytes."""
    x = np.zeros(n, np.float64)
    x.view(np.uint8)[:n:OVERLAP_EVERY] = 1
    return (x, x.view(np.bool_)[:n])


# Values assigned through a mask, with whether the kernel takes them: a
# Python float, or a Python int that converts to a double.
_FILL_VALUES: tuple[tuple[str, object, bool], ...] = (
    ("0.0", 0.0, True), ("1.5", 1.5, True), ("-0.0", -0.0, True), ("nan", float("nan"), True),
    ("-inf", float("-inf"), True), ("5e-324", 5e-324, True), ("0", 0, True), ("-3", -3, True),
    ("2**53", 2**53, True), ("2**53 + 1", 2**53 + 1, True), ("2**70", 2**70, True),
    ("10**400", 10**400, False), ("True", True, False), ("np.float64(2.0)", np.float64(2.0), False),
    ("np.float32(2.0)", np.float32(2.0), False), ("array", np.array(7.0), False), ("'1.5'", "1.5", False),
    ("None", None, False), ("1j", 1j, False),
)


def _copyto_casting(value, x, mask):
    """Run `np.copyto(x, value, casting='same_kind', where=mask)` and return `x`."""
    np.copyto(x, value, casting="same_kind", where=mask)
    return x


def _copyto_positional(value, x, mask):
    """Run `np.copyto(x, value, 'same_kind', mask)` and return `x`."""
    np.copyto(x, value, "same_kind", mask)
    return x


def _copyto_keywords(value, x, mask):
    """Run `np.copyto(dst=x, src=value, where=mask)` and return `x`."""
    np.copyto(dst=x, src=value, where=mask)
    return x


def _copyto_unmasked(value, x, mask):
    """Run `np.copyto(x, value)`, which has no mask, and return `x`."""
    np.copyto(x, value)
    return x


def _putmask_keywords(value, x, mask):
    """Run `np.putmask(x, mask=mask, values=value)` and return `x`."""
    np.putmask(x, mask=mask, values=value)
    return x


def _putmask_list(value, x, mask):
    """Run `np.putmask(x, mask, [value, value])`, which repeats a list of values, and return `x`."""
    np.putmask(x, mask, [value, value])
    return x


def _int_mask(maker: Callable[[], np.ndarray]) -> Callable[[], np.ndarray]:
    """Return a maker of the same mask as `maker`'s, stored as int64 0 and 1."""
    return lambda: maker().astype(np.int64)


# The ways to write a masked assignment of a scalar: as text, the function
# that makes it, and whether the kernel may take it. All but the first are
# function hooks.
_ASSIGNMENTS: tuple[tuple[str, Callable, bool], ...] = (
    ("x[mask] = v", _assign, True),
    ("np.putmask(x, mask, v)", _putmask, True),
    ("np.copyto(x, v, where=mask)", _copyto, True),
    ("np.putmask(x, mask=mask, values=v)", _putmask_keywords, True),
    ("np.copyto(dst=x, src=v, where=mask)", _copyto_keywords, True),
    ("np.copyto(x, v, 'same_kind', mask)", _copyto_positional, False),
    ("np.copyto(x, v, casting='same_kind', where=mask)", _copyto_casting, False),
    ("np.copyto(x, v)", _copyto_unmasked, False),
    ("np.putmask(x, mask, [v, v])", _putmask_list, False),
)
# The function forms among them that take the kernel under the same
# conditions as `x[mask] = v`.
_FUNCTION_ASSIGNMENTS = _ASSIGNMENTS[1:3]


def _maskfill_shape_cases(dtype: str, takes: str | None) -> Iterator[Case]:
    """Yield the masked assignments whose arrays have two dimensions or none.

    `np.putmask` needs only the sizes of `x` and the mask to match and reads
    both in C order. `np.copyto` broadcasts the mask against `x`, so the
    kernel takes it only for matching shapes. `x[mask] = v` is taken for a
    mask of the shape of `x`; a mask of another shape selects rows or is an
    error, and is NumPy's.

    Args:
        dtype: Element type of `x`, as a spec name.
        takes: The kernel for that type, or None.

    Yields:
        The cases.
    """
    n = ORDINARY_N
    x_flat = partial(values, dtype, n)
    mask_flat = partial(values, "bool", n, arg="mask")
    shapes = (
        # (text, x, mask, taken by x[mask] = v, by putmask, by copyto)
        ("both two-dimensional", _reshaped(x_flat), _reshaped(mask_flat), takes, takes, takes),
        ("both three-dimensional", _cube(x_flat), _cube(mask_flat), takes, takes, takes),
        ("both two-dimensional, shapes transposed", _reshaped(x_flat), lambda: mask_flat().reshape(COLUMNS, -1),
         None, takes, None),
        ("x two-dimensional, mask over its rows", _reshaped(x_flat), partial(values, "bool", n // COLUMNS, arg="mask"),
         None, None, None),
        ("x two-dimensional, mask one-dimensional", _reshaped(x_flat), mask_flat, None, takes, None),
        ("x one-dimensional, mask two-dimensional", x_flat, _reshaped(mask_flat), None, takes, None),
        ("x two-dimensional, Fortran order", _reshaped(x_flat, "F"), _reshaped(mask_flat, "F"), None, None, None),
        ("x zero-dimensional", lambda: np.array(x_flat()[0]), lambda: np.array(True), None, None, None),
        ("x empty two-dimensional", lambda: np.zeros((0, COLUMNS), np_dtype(dtype)),
         lambda: np.zeros((0, COLUMNS), np.bool_), takes, takes, takes),
    )
    for text, x_maker, mask_maker, *expects in shapes:
        make = lambda x_maker=x_maker, mask_maker=mask_maker: (x_maker(), mask_maker())
        for (spelling, function, _), expect in zip(_ASSIGNMENTS[:3], expects):
            yield Case("maskfill", f"{spelling} with v=1.5 {dtype} {text}", make, partial(function, 1.5), expect)


def maskfill_cases() -> Iterator[Case]:
    """Yield the cases of `x[mask] = value`, `np.putmask` and `np.copyto(..., where=mask)`."""
    kernel = "maskfill_f64"
    for n in VALUE_SIZES:
        for mask_label, mask_maker in _mask_makers(n):
            for text, value, _ in _FILL_VALUES[:4]:
                yield Case("maskfill", f"x[mask] = {text} f64 n={n} {mask_label}",
                           partial(_values_and_mask, "f64", n, "random", mask_maker), partial(_assign, value), kernel)
            for spelling, function, _ in _FUNCTION_ASSIGNMENTS:
                yield Case("maskfill", f"{spelling} with v=1.5 f64 n={n} {mask_label}",
                           partial(_values_and_mask, "f64", n, "random", mask_maker), partial(function, 1.5), kernel)
    for dtype in ("f64", "f32", "i64"):
        takes = kernel if dtype == "f64" else None
        for n in (0, *LAYOUT_SIZES):
            plain_mask = partial(values, "bool", n, arg="mask")
            for spelling, function, by_kernel in _ASSIGNMENTS:
                for text, value, taken in _FILL_VALUES:
                    yield Case("maskfill", f"{spelling} with v={text} {dtype} n={n}",
                               partial(_x_and_mask, dtype, n, CONTIGUOUS, plain_mask), partial(function, value),
                               takes if taken and by_kernel else None)
            for spelling, function, _ in _ASSIGNMENTS[:3]:
                for variant in variants_for(dtype, n):
                    if variant.dims != 1 or variant.name == "list":
                        continue
                    expect = takes if variant.plain and variant.writeable else None
                    if takes and (not variant.predictable or (n == 0 and not variant.plain)):
                        expect = ANY
                    yield Case("maskfill", f"{spelling} with v=1.5 {dtype} n={n} x={variant.name}",
                               partial(_x_and_mask, dtype, n, variant, plain_mask), partial(function, 1.5), expect)
                for variant in variants_for("bool", n):
                    if variant.dims != 1:
                        continue
                    expect = takes if variant.plain else (ANY if takes and n == 0 else None)
                    yield Case("maskfill", f"{spelling} with v=1.5 {dtype} n={n} mask={variant.name}",
                               partial(_x_and_mask, dtype, n, CONTIGUOUS, partial(_variant_of, "bool", n, variant, "mask")),
                               partial(function, 1.5), expect)
                yield Case("maskfill", f"{spelling} with v=1.5 {dtype} n={n} mask=int64",
                           partial(_x_and_mask, dtype, n, CONTIGUOUS, _int_mask(plain_mask)), partial(function, 1.5), None)
                if n:
                    for byte in BAD_MASK_BYTES:
                        yield Case("maskfill", f"{spelling} with v=1.5 {dtype} n={n} mask bytes are {byte}",
                                   partial(_x_and_mask, dtype, n, CONTIGUOUS, partial(_bad_mask, n, byte)),
                                   partial(function, 1.5), None)
                    short = partial(values, "bool", n - 1, arg="mask")
                    yield Case("maskfill", f"{spelling} with v=1.5 {dtype} n={n} mask one short",
                               partial(_x_and_mask, dtype, n, CONTIGUOUS, short), partial(function, 1.5), None)
        yield from _maskfill_shape_cases(dtype, takes)
    for spelling, function, _ in _ASSIGNMENTS[:3]:
        yield Case("maskfill", f"{spelling} with v=1.5 where mask is a view of x", partial(_overlapping, ORDINARY_N),
                   partial(function, 1.5), None)


def nonzero_cases() -> Iterator[Case]:
    """Yield the cases of `nonzero` on booleans."""
    spellings = (
        Spelling("mask.nonzero()", lambda m: m.nonzero(), any_shape=False),
        Spelling("np.nonzero(mask)", np.nonzero, any_shape=False, normalises=True),
        Spelling("np.flatnonzero(mask)", np.flatnonzero, normalises=True),
        Spelling("np.argwhere(mask)", np.argwhere, MAYBE),
        Spelling("np.where(mask)", np.where, any_shape=False),
        Spelling("np.where(mask, 1.0, 0.0)", lambda m: np.where(m, 1.0, 0.0), NEVER),
        Spelling("np.where(mask, mask, mask)", lambda m: np.where(m, m, m), NEVER),
    )
    # The kernel is not called for an empty mask's sake: it takes it and returns nothing.
    yield from unary_cases("nonzero", {"bool": "nonzero_bool"}, ("u8", "f64", "i64"), spellings)
    # The spellings that would take the kernel for a clean mask.
    taking = (spellings[0], spellings[1], spellings[4])
    for n in LAYOUT_SIZES:
        for byte in BAD_MASK_BYTES:
            for spelling in taking:
                yield Case("nonzero", f"{spelling.text} n={n} mask bytes are {byte}",
                           lambda n=n, byte=byte: (_bad_mask(n, byte),), spelling.call, None)


def cumsum_cases(inexact: bool) -> Iterator[Case]:
    """Yield the cases of `cumsum`.

    Args:
        inexact: False for the int64 and bool kernels and the float kernels' absence;
            True for the float kernels, which are compared within their
            specs' tolerance and only exist under install(inexact=True).

    Yields:
        The cases.
    """
    spellings = (
        Spelling("x.cumsum()", lambda x: x.cumsum()),
        Spelling("np.cumsum(x)", np.cumsum, normalises=True),
        Spelling("x.cumsum(None)", lambda x: x.cumsum(None)),
        Spelling("x.cumsum(axis=None, dtype=None)", lambda x: x.cumsum(axis=None, dtype=None)),
        Spelling("np.cumsum(x, axis=0)", lambda x: np.cumsum(x, axis=0), NEVER),
        Spelling("x.cumsum(0)", lambda x: x.cumsum(0), NEVER),
        Spelling("x.cumsum(dtype=np.float64)", lambda x: x.cumsum(dtype=np.float64), NEVER),
        Spelling("x.cumsum(out=o)", lambda x: x.cumsum(out=np.empty(np.size(x), np.asarray(x).dtype)), NEVER),
        Spelling("np.add.accumulate(x)", np.add.accumulate, NEVER),
    )
    if inexact:
        yield from unary_cases("cumsum", {"f32": "cumsum_f32", "f64": "cumsum_f64"}, (), spellings,
                               empty_takes=False, tolerance=True)
    else:
        yield from unary_cases("cumsum", {"i64": "cumsum_i64", "bool": "cumsum_bool"}, ("f32", "f64", "i32", "u8"),
                               spellings, empty_takes=False)
        # A mask holding a byte other than 0 or 1 counts once per element in NumPy.
        for n, byte, spelling in itertools.product(LAYOUT_SIZES, BAD_MASK_BYTES, spellings[:2]):
            yield Case("cumsum", f"{spelling.text} bool n={n} mask bytes are {byte}",
                       partial(_made, partial(_bad_mask, n, byte)), spelling.call, None)


def astype_cases() -> Iterator[Case]:
    """Yield the cases of `astype` from uint8 to float32."""
    spellings = (
        Spelling("x.astype(np.float32)", lambda x: x.astype(np.float32)),
        Spelling("x.astype('float32')", lambda x: x.astype("float32")),
        Spelling("x.astype('f4')", lambda x: x.astype("f4")),
        Spelling("x.astype(np.dtype('float32'))", lambda x: x.astype(np.dtype("float32"))),
        Spelling("x.astype(dtype=np.float32)", lambda x: x.astype(dtype=np.float32), NEVER),
        Spelling("x.astype(np.float32, copy=False)", lambda x: x.astype(np.float32, copy=False), NEVER),
        Spelling("x.astype(np.float32, order='F')", lambda x: x.astype(np.float32, order="F"), NEVER),
        Spelling("x.astype(np.float64)", lambda x: x.astype(np.float64), NEVER),
        Spelling("x.astype('>f4')", lambda x: x.astype(">f4"), NEVER),
        Spelling("x.astype(float)", lambda x: x.astype(float), NEVER),
        Spelling("np.asarray(x, dtype=np.float32)", lambda x: np.asarray(x, dtype=np.float32), NEVER),
    )
    yield from unary_cases("astype", {"u8": "cast_u8_f32"}, ("i8", "bool", "u16", "f64"), spellings)


def _late_bad_mask(n: int, byte: int) -> np.ndarray:
    """Return a boolean array of zeros whose last element is stored as `byte`.

    The binding checks a mask block by block; this one is clean until its end.
    """
    raw = np.zeros(n, np.uint8)
    raw[-1] = byte
    return raw.view(np.bool_)


def sum_cases() -> Iterator[Case]:
    """Yield the cases of `sum` on uint8 and on booleans, which the same kernel serves."""
    spellings = (
        Spelling("np.sum(x)", np.sum),
        Spelling("x.sum()", lambda x: x.sum()),
        Spelling("np.sum(x, axis=0)", lambda x: np.sum(x, axis=0), NEVER),
        Spelling("np.sum(x, axis=None)", lambda x: np.sum(x, axis=None), NEVER),
        Spelling("x.sum(dtype=np.uint8)", lambda x: x.sum(dtype=np.uint8), NEVER),
        Spelling("np.sum(x, initial=5)", lambda x: np.sum(x, initial=5), NEVER),
        Spelling("np.sum(x, keepdims=True)", lambda x: np.sum(x, keepdims=True), NEVER),
        Spelling("np.add.reduce(x)", np.add.reduce, NEVER),
        Spelling("np.sum(a=x)", lambda x: np.sum(a=x), NEVER),
    )
    yield from unary_cases("sum", {"u8": "sum_u8", "bool": "sum_u8"}, ("i8", "u16", "f64"), spellings)
    # A mask holding a byte other than 0 or 1 counts once per element in NumPy.
    for n, byte, spelling in itertools.product(LAYOUT_SIZES, BAD_MASK_BYTES, spellings[:2]):
        yield Case("sum", f"{spelling.text} bool n={n} mask bytes are {byte}",
                   partial(_made, partial(_bad_mask, n, byte)), spelling.call, None)
        yield Case("sum", f"{spelling.text} bool n={SUM_BOOL_LATE_N} last mask byte is {byte}",
                   partial(_made, partial(_late_bad_mask, SUM_BOOL_LATE_N, byte)), spelling.call, None)


def count_nonzero_cases() -> Iterator[Case]:
    """Yield the cases of `count_nonzero`."""
    spellings = (
        Spelling("np.count_nonzero(x)", np.count_nonzero),
        Spelling("np.count_nonzero(x, axis=0)", lambda x: np.count_nonzero(x, axis=0), NEVER),
        Spelling("np.count_nonzero(x, axis=None)", lambda x: np.count_nonzero(x, axis=None), NEVER),
        Spelling("np.count_nonzero(x, keepdims=True)", lambda x: np.count_nonzero(x, keepdims=True), NEVER),
        Spelling("np.count_nonzero(a=x)", lambda x: np.count_nonzero(a=x), NEVER),
    )
    kernels = {"i64": "count_nonzero_i64", "f64": "count_nonzero_f64"}
    yield from unary_cases("count_nonzero", kernels, ("i32", "f32", "bool", "u8"), spellings)
    yield from unusual_nan_cases("count_nonzero", {"f64": "count_nonzero_f64"}, "np.count_nonzero(x)", np.count_nonzero)


def _histogram_sets() -> Iterator[tuple[str, int, float, float]]:
    """Yield every bin count and range the histogram kernel's spec lists."""
    k = spec("histogram_f64")
    for set_name in k.scalar_set_names:
        scalars = k.scalar_values(set_name)
        yield set_name, int(scalars["nbins"]), scalars["lo"], scalars["hi"]


def histogram_cases() -> Iterator[Case]:
    """Yield the cases of `histogram`.

    The kernel takes a contiguous float64 array, an integer bin count and a
    range of two plain numbers, or no range. NumPy raises for a range that is
    not finite or whose bin edges are not strictly increasing, and widens an
    empty one; all of those must be left to it.
    """
    kernel = "histogram_f64"
    for set_name, bins, lo, hi in _histogram_sets():
        sizes = VALUE_SIZES if set_name == DEFAULT_SCALARS else FEW_SIZES
        for _, _, label, maker in value_arrays("f64", sizes):
            yield Case("histogram", f"np.histogram(x, {bins}, ({lo!r}, {hi!r})) {label}", partial(_made, maker),
                       lambda x, bins=bins, lo=lo, hi=hi: np.histogram(x, bins, (lo, hi)), kernel)
    finite = ("random", "equal", "asc", "desc", "ties", "zeros", "denormal")
    for n in FEW_SIZES:
        for cls in judge_cases.classes("f64")[0]:
            # Without a range NumPy uses the data's minimum and maximum. It
            # raises if they are not finite and widens the range if they are
            # equal, so only data with two distinct finite extremes is taken.
            data = values("f64", n, cls)
            usable = cls in finite and n > 1 and np.isfinite(data).all() and data.min() < data.max()
            for text, call in (("np.histogram(x)", np.histogram), ("np.histogram(x, 7)", lambda x: np.histogram(x, 7))):
                yield Case("histogram", f"{text} n={n} x={cls}", partial(_made, partial(values, "f64", n, cls)),
                           call, kernel if usable else None)
    spellings = (
        Spelling("np.histogram(x, 50, (-3.0, 3.0))", lambda x: np.histogram(x, 50, (-3.0, 3.0))),
        Spelling("np.histogram(x, bins=50, range=(-3.0, 3.0))", lambda x: np.histogram(x, bins=50, range=(-3.0, 3.0))),
        Spelling("np.histogram(a=x, bins=5, range=(0, 1))", lambda x: np.histogram(a=x, bins=5, range=(0, 1))),
        Spelling("np.histogram(x, range=(-1, 1))", lambda x: np.histogram(x, range=(-1, 1))),
        Spelling("np.histogram(x, 50, (-3.0, 3.0), None, None)", lambda x: np.histogram(x, 50, (-3.0, 3.0), None, None)),
        Spelling("np.histogram(x, 50, [-3.0, 3.0])", lambda x: np.histogram(x, 50, [-3.0, 3.0]), NEVER),
        Spelling("np.histogram(x, 50, (np.float64(-3), 3.0))", lambda x: np.histogram(x, 50, (np.float64(-3), 3.0)), NEVER),
        Spelling("np.histogram(x, np.int64(50), (-3.0, 3.0))", lambda x: np.histogram(x, np.int64(50), (-3.0, 3.0)), NEVER),
        Spelling("np.histogram(x, 50, (-3.0, 3.0), density=True)", lambda x: np.histogram(x, 50, (-3.0, 3.0), density=True), NEVER),
        Spelling("np.histogram(x, 50, (-3.0, 3.0), density=False)", lambda x: np.histogram(x, 50, (-3.0, 3.0), density=False), NEVER),
        Spelling("np.histogram(x, 50, (-3.0, 3.0), weights=w)", lambda x: np.histogram(x, 50, (-3.0, 3.0), weights=np.ones(np.shape(x))), NEVER),
        Spelling("np.histogram(x, 'auto')", lambda x: np.histogram(x, "auto"), NEVER),
        Spelling("np.histogram(x, [-1.0, 0.0, 2.0])", lambda x: np.histogram(x, [-1.0, 0.0, 2.0]), NEVER),
        Spelling("np.histogram(x, 0, (-3.0, 3.0))", lambda x: np.histogram(x, 0, (-3.0, 3.0)), NEVER),
        Spelling("np.histogram(x, -4, (-3.0, 3.0))", lambda x: np.histogram(x, -4, (-3.0, 3.0)), NEVER),
        Spelling("np.histogram(x, True, (-3.0, 3.0))", lambda x: np.histogram(x, True, (-3.0, 3.0)), NEVER),
        Spelling("np.histogram(x, 5, (3.0, -3.0)) reversed", lambda x: np.histogram(x, 5, (3.0, -3.0)), NEVER),
        Spelling("np.histogram(x, 5, (1.0, 1.0)) empty", lambda x: np.histogram(x, 5, (1.0, 1.0)), NEVER),
        Spelling("np.histogram(x, 5, (nan, 1.0))", lambda x: np.histogram(x, 5, (float("nan"), 1.0)), NEVER),
        Spelling("np.histogram(x, 5, (0.0, inf))", lambda x: np.histogram(x, 5, (0.0, float("inf"))), NEVER),
        Spelling("np.histogram(x, 5, (0, 2**70))", lambda x: np.histogram(x, 5, (0, 2**70)), NEVER),
        Spelling("np.histogram(x, 1000, (1e16, 1e16 + 2)) too many bins", lambda x: np.histogram(x, 1000, (1e16, 1e16 + 2)), NEVER),
        Spelling("np.histogram(x, 5, (0.0, 1.0, 2.0))", lambda x: np.histogram(x, 5, (0.0, 1.0, 2.0)), NEVER),
        Spelling("np.histogram(x, bins=5, bogus=1)", lambda x: np.histogram(x, bins=5, bogus=1), NEVER),
    )
    for dtype in ("f64", "f32", "i64"):
        for n, spelling in itertools.product((0, *LAYOUT_SIZES), spellings):
            for variant in variants_for(dtype, n):
                yield Case("histogram", f"{spelling.text} {dtype} n={n} layout={variant.name}",
                           partial(_built, dtype, n, variant), spelling.call,
                           expectation(kernel if dtype == "f64" else None, spelling, variant, n))


def bincount_cases() -> Iterator[Case]:
    """Yield the cases of `bincount` on uint8."""
    spellings = (
        Spelling("np.bincount(x)", np.bincount, any_shape=False),
        Spelling("np.bincount(x, minlength=0)", lambda x: np.bincount(x, minlength=0), any_shape=False),
        Spelling("np.bincount(x, minlength=10)", lambda x: np.bincount(x, minlength=10), any_shape=False),
        Spelling("np.bincount(x, minlength=256)", lambda x: np.bincount(x, minlength=256), any_shape=False),
        Spelling("np.bincount(x, minlength=300)", lambda x: np.bincount(x, minlength=300), any_shape=False),
        Spelling("np.bincount(x, None, 256)", lambda x: np.bincount(x, None, 256), any_shape=False),
        Spelling("np.bincount(x, weights=None)", lambda x: np.bincount(x, weights=None), any_shape=False),
        Spelling("np.bincount(x, weights=w)", lambda x: np.bincount(x, weights=np.ones(np.shape(x))), NEVER),
        Spelling("np.bincount(x, minlength=-1)", lambda x: np.bincount(x, minlength=-1), NEVER),
        Spelling("np.bincount(x, minlength=np.int64(5))", lambda x: np.bincount(x, minlength=np.int64(5)), NEVER),
        Spelling("np.bincount(x=x)", lambda x: np.bincount(x=x), NEVER),
        Spelling("np.bincount(x, None, 0, 1)", lambda x: np.bincount(x, None, 0, 1), NEVER),
    )
    yield from unary_cases("bincount", {"u8": "bincount_u8"}, ("i64", "u16", "i8", "bool"), spellings)
    for value in (0, 1, 254, 255):
        yield Case("bincount", f"np.bincount(x) every byte is {value}", lambda value=value: (np.full(ORDINARY_N, value, np.uint8),),
                   np.bincount, "bincount_u8")


def _isin_x(spread: str, n: int) -> np.ndarray:
    """Build the array searched by an isin case.

    Args:
        spread: "narrow" for values from 0 to 1,000, where a small set of test
            values matches often; "wide" for the whole int64 range.
        n: Element count.

    Returns:
        A contiguous int64 array.
    """
    rng = np.random.default_rng([SEED, n])
    if spread == "narrow":
        return rng.integers(0, 1000, n)
    info = np.iinfo(np.int64)
    x = rng.integers(info.min, info.max, n, endpoint=True)
    x[::3] = info.max
    x[1::7] = info.min
    return x


# Test values for isin, with whether the kernel takes them: one to eight
# int64 values, as an array or as a list or tuple of Python ints.
_INT64 = np.iinfo(np.int64)
_ISIN_TESTS: tuple[tuple[str, Callable[[], object], bool], ...] = (
    ("[5]", lambda: [5], True),
    ("[1, 2, 3]", lambda: [1, 2, 3], True),
    ("(1, 2, 3)", lambda: (1, 2, 3), True),
    ("[3, 3, 3]", lambda: [3, 3, 3], True),
    ("eight values", lambda: [1, 50, 999, 0, 7, 500, 501, 2], True),
    ("eight wide values", lambda: [_INT64.max, _INT64.min, 0, -1, 1, 2**40, -(2**40), 65535], True),
    ("[30000, 59990, 5, 100, 1000, 20000, 40000, -10]", lambda: [30000, 59990, 5, 100, 1000, 20000, 40000, -10], True),
    ("[-10, 59990, 5, 100, 1000, 20000, 40000, 30000]", lambda: [-10, 59990, 5, 100, 1000, 20000, 40000, 30000], True),
    ("array of 2", lambda: np.array([4, 999]), True),
    ("array of 8", lambda: np.arange(8) * 100, True),
    ("array of 9", lambda: np.arange(9) * 100, False),
    ("nine values", lambda: list(range(9)), False),
    ("[]", lambda: [], False),
    ("empty array", lambda: np.array([], np.int64), False),
    ("[1, 2.5]", lambda: [1, 2.5], False),
    ("[1, True]", lambda: [1, True], False),
    ("[2**63]", lambda: [2**63], False),
    ("[-2**63 - 1]", lambda: [-(2**63) - 1], False),
    ("int32 array", lambda: np.array([1, 2], np.int32), False),
    ("strided array", lambda: np.arange(16)[::4], False),
    ("two-dimensional array", lambda: np.array([[1, 2], [3, 4]]), False),
    ("scalar 5", lambda: 5, False),
    ("set", lambda: {1, 2}, False),
)


def isin_cases() -> Iterator[Case]:
    """Yield the cases of `isin` on int64."""
    kernel = "isin_i64"
    for n, spread in itertools.product(VALUE_SIZES, ("narrow", "wide")):
        for text, test, taken in _ISIN_TESTS:
            yield Case("isin", f"np.isin(x, {text}) n={n} x={spread}", lambda n=n, spread=spread, test=test: (_isin_x(spread, n), test()),
                       np.isin, kernel if taken else None)
    spellings = (
        Spelling("np.isin(x, [1, 2, 3])", lambda x: np.isin(x, [1, 2, 3])),
        Spelling("np.isin(x, [1, 2, 3], invert=True)", lambda x: np.isin(x, [1, 2, 3], invert=True), NEVER),
        Spelling("np.isin(x, [1, 2, 3], assume_unique=True)", lambda x: np.isin(x, [1, 2, 3], assume_unique=True), NEVER),
        Spelling("np.isin(x, [1, 2, 3], kind='sort')", lambda x: np.isin(x, [1, 2, 3], kind="sort"), NEVER),
        Spelling("np.isin(element=x, test_elements=[1, 2, 3])", lambda x: np.isin(element=x, test_elements=[1, 2, 3]), NEVER),
    )
    for dtype in ("i64", "i32", "u8", "f64"):
        for n, spelling in itertools.product((0, *LAYOUT_SIZES), spellings):
            for variant in variants_for(dtype, n):
                yield Case("isin", f"{spelling.text} {dtype} n={n} layout={variant.name}",
                           partial(_built, dtype, n, variant), spelling.call,
                           expectation(kernel if dtype == "i64" else None, spelling, variant, n))


# ---------------------------------------------------- searchsorted and interp

# Lengths of the sorted array, and numbers of needles, either side of every
# boundary at which the search kernels change path: 16, 64 and 128 needles at
# once, and one much longer array.
TABLE_SIZES = (0, 1, 2, 15, 16, 17, 63, 64, 65, 1000, 4097)
NEEDLE_COUNTS = (0, 1, 2, 15, 16, 17, 63, 64, 65, 127, 128, 129, 1000)
# Length from which `searchsorted_f64` takes its path for arrays beyond the
# cache, and a length above it.
LARGE_TABLE_N = 1_048_576
BEYOND_LARGE_TABLE_N = LARGE_TABLE_N + 3
# Numbers of needles tried against the two long arrays: one, and either side
# of the group of 128 that the path for long arrays works in.
LARGE_TABLE_NEEDLES = (1, 64, 127, 128, 129, 1000)
# (value class of the array, value class of the needles) tried against the
# two long arrays.
LARGE_TABLE_PAIRS = (("random", "nan_some"), ("random", "hits"), ("equal", "hits"))
# Number of needles in the cases that vary the spelling or the layout of a
# search: more than one group of 64, with a leftover.
SPELLING_NEEDLES = 100
# (sample points, queries) of the interp cases whose sample points hold a NaN.
NAN_POINT_SHAPES = ((1, SPELLING_NEEDLES), (2, SPELLING_NEEDLES), (ORDINARY_N, SPELLING_NEEDLES), (ORDINARY_N, ORDINARY_N))
# Value classes for the sorted array of a search: ordinary values, repeated
# values, infinities at both ends, and NaNs, which NumPy sorts last.
TABLE_CLASSES = ("random", "ties", "inf_mix", "nan_some")
# Value classes for the needles.
NEEDLE_CLASSES = ("random", "ties", "inf_mix", "nan_some", "nan_all", "extremes", "zeros")
# Value classes for the sample points of interp. A NaN is not among them:
# NumPy's answer then depends on which elements its search probes.
POINT_CLASSES = ("random", "ties", "inf_mix", "equal", "zeros")
# Value classes for the values at the sample points.
FP_CLASSES = ("random", "nan_some", "inf_mix", "extremes", "equal")
# Mirrors INTERP_MAX_POINTS_PER_QUERY in the binding: with more sample points
# than this per query, np.interp is left to NumPy.
INTERP_MAX_POINTS_PER_QUERY = 64


def _table(n: int, cls: str = judge_cases.RANDOM) -> np.ndarray:
    """Build a sorted float64 array of `n` elements from a value class, NaNs last."""
    return np.sort(values("f64", n, cls, arg="a"))


def _hits(table: np.ndarray, count: int) -> np.ndarray:
    """Build `count` needles that are elements of `table`, so every search is an exact hit.

    Args:
        table: The array the needles are drawn from.
        count: Number of needles.

    Returns:
        A float64 array of `count` elements; zeros when the table is empty.
    """
    if len(table) == 0:
        return np.zeros(count)
    picks = np.random.default_rng([SEED, len(table), count]).integers(0, len(table), count)
    return table[picks].copy()


def _search_arguments(n: int, table_class: str, count: int, needle_class: str) -> tuple:
    """Build the sorted array and the needles of a searchsorted case.

    Args:
        n: Length of the sorted array.
        table_class: Value class of the sorted array.
        count: Number of needles.
        needle_class: Value class of the needles, or "hits" for needles taken
            from the array itself.

    Returns:
        (sorted array, needles).
    """
    table = _table(n, table_class)
    if needle_class == "hits":
        return table, _hits(table, count)
    return table, values("f64", count, needle_class, arg="v")


def _search_layout_expectation(kernel: str, variant: Variant, n: int, any_shape: bool) -> str | None:
    """Say whether a search kernel takes a call in which one array is a variant.

    Args:
        kernel: The kernel the call would reach.
        variant: How the array in question is laid out.
        n: Its element count.
        any_shape: Whether the kernel accepts the array with two or more
            dimensions: true for needles and queries, false for the sorted
            array and the sample points.

    Returns:
        The kernel name, None, or ANY.
    """
    if not variant.predictable:
        return ANY
    if variant.plain and (variant.dims == 1 or (variant.dims > 1 and any_shape)):
        return kernel
    # NumPy flags every empty array as contiguous, whatever its strides say.
    return ANY if n == 0 else None


def _search_right(a, v):
    """Call `np.searchsorted(a, v, side='right')`.

    Args:
        a: The sorted array.
        v: The needles.

    Returns:
        What NumPy, or the kernel in its place, returns.
    """
    return np.searchsorted(a, v, side="right")


# The two sides of a search: (the call as a case names it, the call, the kernel
# that takes it). The sides differ only for a needle that equals an element
# and for a NaN needle, so the "hits" class is what tells them apart.
SEARCH_SIDES = (
    ("np.searchsorted(a, v)", np.searchsorted, "searchsorted_f64"),
    ("np.searchsorted(a, v, side='right')", _search_right, "searchsorted_right_f64"),
)


def _search_grid_cases(text: str, search: Callable, kernel: str) -> Iterator[Case]:
    """Yield the cases of one side of a search over sizes and value classes.

    Args:
        text: The call, as the cases name it.
        search: Makes the call from (sorted array, needles).
        kernel: The kernel expected to take every case.

    Returns:
        An iterator over the cases.
    """
    for n, count in itertools.product(TABLE_SIZES, NEEDLE_COUNTS):
        # Every class of needle on an ordinary array, then every class of array.
        pairs = [("random", needle_class) for needle_class in (*NEEDLE_CLASSES, "hits")]
        pairs += [(table_class, needle_class) for table_class in TABLE_CLASSES[1:] for needle_class in ("random", "hits")]
        for table_class, needle_class in pairs:
            yield Case("searchsorted", f"{text} a={table_class} n={n} v={needle_class} needles={count}",
                       partial(_search_arguments, n, table_class, count, needle_class), search, kernel)
    # The long arrays get needles with NaNs among them, and needles that are
    # elements, so that the path for long arrays is seen to compare the right
    # way through the binding as well as in the judge, which draws needles
    # from the array at every size. An array of one repeated value is there for the
    # first step of the search, which only shows a wrong comparison when
    # elements equal to the needle lie on both sides of a[p - 1], p being the
    # largest power of two not above the length.
    for n in (LARGE_TABLE_N, BEYOND_LARGE_TABLE_N):
        for count, (table_class, needle_class) in itertools.product(LARGE_TABLE_NEEDLES, LARGE_TABLE_PAIRS):
            yield Case("searchsorted", f"{text} a={table_class} n={n} v={needle_class} needles={count}",
                       partial(_search_arguments, n, table_class, count, needle_class), search, kernel)


def searchsorted_cases() -> Iterator[Case]:
    """Yield the cases of `searchsorted` on float64, for both sides."""
    kernel, right = "searchsorted_f64", "searchsorted_right_f64"
    for side in SEARCH_SIDES:
        yield from _search_grid_cases(*side)

    n, count = ORDINARY_N, SPELLING_NEEDLES
    arguments = partial(_search_arguments, n, "random", count, "random")
    spellings = (
        ("a.searchsorted(v)", lambda a, v: a.searchsorted(v), kernel),
        ("np.searchsorted(a, v, 'left')", lambda a, v: np.searchsorted(a, v, "left"), kernel),
        ("np.searchsorted(a, v, side='left', sorter=None)", lambda a, v: np.searchsorted(a, v, side="left", sorter=None), kernel),
        ("a.searchsorted(v=v)", lambda a, v: a.searchsorted(v=v), kernel),
        ("np.searchsorted(a, v, side='right')", lambda a, v: np.searchsorted(a, v, side="right"), right),
        ("np.searchsorted(a, v, 'right')", lambda a, v: np.searchsorted(a, v, "right"), right),
        ("a.searchsorted(v, 'right')", lambda a, v: a.searchsorted(v, "right"), right),
        ("a.searchsorted(v=v, side='right')", lambda a, v: a.searchsorted(v=v, side="right"), right),
        ("np.searchsorted(a, v, side='right', sorter=None)", lambda a, v: np.searchsorted(a, v, side="right", sorter=None), right),
        ("np.searchsorted(a, v, sorter=np.arange(len(a)))", lambda a, v: np.searchsorted(a, v, sorter=np.arange(len(a))), None),
        ("np.searchsorted(a, v, side='right', sorter=np.arange(len(a)))", lambda a, v: np.searchsorted(a, v, side="right", sorter=np.arange(len(a))), None),
        ("np.digitize(v, a)", lambda a, v: np.digitize(v, a), right),
        ("np.digitize(v, a, right=True)", lambda a, v: np.digitize(v, a, right=True), kernel),
        # Needles that are bin edges: the two sides of digitize differ at every one.
        ("np.digitize(a, a)", lambda a, v: np.digitize(a, a), right),
        ("np.digitize(a, a, right=True)", lambda a, v: np.digitize(a, a, right=True), kernel),
        # Decreasing bins: digitize searches them reversed, which is a view
        # that is not contiguous...
        ("np.digitize(v, a[::-1].copy())", lambda a, v: np.digitize(v, a[::-1].copy()), None),
        ("np.digitize(v, a[::-1].copy(), right=True)", lambda a, v: np.digitize(v, a[::-1].copy(), right=True), None),
        # ...unless the bins were a reversed view to begin with: reversed
        # again they are the contiguous array.
        ("np.digitize(v, a[::-1])", lambda a, v: np.digitize(v, a[::-1]), right),
        ("np.digitize(v, a[::-1], right=True)", lambda a, v: np.digitize(v, a[::-1], right=True), kernel),
        ("np.searchsorted(a, 0.25, side='right')", lambda a, v: np.searchsorted(a, 0.25, side="right"), right),
        ("np.searchsorted(a, a[3], side='right')", lambda a, v: np.searchsorted(a, float(a[3]), side="right"), right),
        ("np.searchsorted(a, float('nan'), side='right')", lambda a, v: np.searchsorted(a, float("nan"), side="right"), right),
        ("np.searchsorted(a, a, side='right')", lambda a, v: np.searchsorted(a, a, side="right"), right),
        ("np.searchsorted(a, 0.25)", lambda a, v: np.searchsorted(a, 0.25), kernel),
        ("np.searchsorted(a, float('nan'))", lambda a, v: np.searchsorted(a, float("nan")), kernel),
        ("np.searchsorted(a, 1)", lambda a, v: np.searchsorted(a, 1), kernel),
        ("np.searchsorted(a, 2**60 + 1)", lambda a, v: np.searchsorted(a, 2**60 + 1), None),
        ("np.searchsorted(a, True)", lambda a, v: np.searchsorted(a, True), None),
        ("np.searchsorted(a, np.float64(0.25))", lambda a, v: np.searchsorted(a, np.float64(0.25)), None),
        ("np.searchsorted(a, np.array(0.25))", lambda a, v: np.searchsorted(a, np.array(0.25)), None),
        ("np.searchsorted(a, v.tolist())", lambda a, v: np.searchsorted(a, v.tolist()), None),
        ("np.searchsorted(a, v.astype(np.float32))", lambda a, v: np.searchsorted(a, v.astype(np.float32)), None),
        ("np.searchsorted(a, v.astype(np.int64))", lambda a, v: np.searchsorted(a, v.astype(np.int64)), None),
        ("np.searchsorted(a.astype(np.float32), v)", lambda a, v: np.searchsorted(a.astype(np.float32), v), None),
        ("np.searchsorted(a.astype(np.int64), v)", lambda a, v: np.searchsorted(a.astype(np.int64), v), None),
        ("np.searchsorted(a, a)", lambda a, v: np.searchsorted(a, a), kernel),
        ("np.searchsorted(a, v, side='up')", lambda a, v: np.searchsorted(a, v, side="up"), None),
        ("np.searchsorted(a, v, side=None)", lambda a, v: np.searchsorted(a, v, side=None), None),
    )
    for text, call, expect in spellings:
        yield Case("searchsorted", f"{text} n={n} needles={count}", arguments, call, expect)

    for (text, search, kernel), size in itertools.product(SEARCH_SIDES, (0, *LAYOUT_SIZES)):
        for variant in variants_for("f64", size):
            # A list has no method of its own: NumPy converts it and calls the
            # method of the array it made.
            expect = ANY if variant.name == "list" else _search_layout_expectation(kernel, variant, size, any_shape=False)
            yield Case("searchsorted", f"{text} n={size} needles={count} a layout={variant.name}",
                       lambda size=size, variant=variant: (variant.build(_table(size)), values("f64", count, arg="v")),
                       search, expect)
            expect = None if variant.name == "list" else _search_layout_expectation(kernel, variant, size, any_shape=True)
            yield Case("searchsorted", f"{text} n={n} needles={size} v layout={variant.name}",
                       lambda size=size, variant=variant: (_table(n), variant.build(values("f64", size, arg="v"))),
                       search, expect)


def _interp_arguments(n: int, count: int, x_class: str, xp_class: str, fp_class: str) -> tuple:
    """Build the queries, sample points and values of an interp case.

    Args:
        n: Number of sample points.
        count: Number of queries.
        x_class: Value class of the queries, or "hits" for queries that are
            sample points.
        xp_class: Value class of the sample points, which are sorted.
        fp_class: Value class of the values at the sample points.

    Returns:
        (x, xp, fp).
    """
    xp = np.sort(values("f64", n, xp_class, arg="xp"))
    x = _hits(xp, count) if x_class == "hits" else values("f64", count, x_class, arg="x")
    return x, xp, values("f64", n, fp_class, arg="fp")


def _interp_with_nan(n: int, count: int, position: int) -> tuple:
    """Build an interp call whose sample points hold a NaN at `position`."""
    x, xp, fp = _interp_arguments(n, count, "random", "random", "random")
    xp[position] = np.nan
    return x, xp, fp


def _interp_takes(n: int, count: int) -> bool:
    """Say whether the binding gives an interp call to the kernel by its two lengths.

    Args:
        n: Number of sample points.
        count: Number of queries.

    Returns:
        True when there are few enough sample points per query for the pass
        over them to be worth making.
    """
    return n // INTERP_MAX_POINTS_PER_QUERY <= count


def interp_cases() -> Iterator[Case]:
    """Yield the cases of `interp` on float64."""
    kernel = "interp_f64"
    for n, count in itertools.product(TABLE_SIZES[1:], NEEDLE_COUNTS):
        expect = kernel if _interp_takes(n, count) else None
        triples = [(x_class, "random", "random") for x_class in (*NEEDLE_CLASSES, "hits")]
        triples += [(x_class, xp_class, "random") for xp_class in POINT_CLASSES[1:] for x_class in ("random", "hits")]
        triples += [("random", "random", fp_class) for fp_class in FP_CLASSES[1:]]
        triples += [("hits", "ties", "inf_mix"), ("inf_mix", "inf_mix", "inf_mix"), ("nan_some", "ties", "nan_some")]
        for x_class, xp_class, fp_class in triples:
            yield Case("interp", f"np.interp(x, xp, fp) n={n} queries={count} x={x_class} xp={xp_class} fp={fp_class}",
                       partial(_interp_arguments, n, count, x_class, xp_class, fp_class), np.interp, expect)
    for n, count in NAN_POINT_SHAPES:
        for position in sorted({0, n // 2, n - 1}):
            yield Case("interp", f"np.interp(x, xp, fp) n={n} queries={count} xp has a NaN at {position}",
                       partial(_interp_with_nan, n, count, position), np.interp, None)

    n, count = ORDINARY_N, SPELLING_NEEDLES
    arguments = partial(_interp_arguments, n, count, "random", "random", "random")
    spellings = (
        ("np.interp(x, xp, fp, None, None, None)", lambda x, xp, fp: np.interp(x, xp, fp, None, None, None), kernel),
        ("np.interp(x, xp, fp, left=None, right=None, period=None)", lambda x, xp, fp: np.interp(x, xp, fp, left=None, right=None, period=None), kernel),
        ("np.interp(x=x, xp=xp, fp=fp)", lambda x, xp, fp: np.interp(x=x, xp=xp, fp=fp), kernel),
        ("np.interp(x, xp, fp, left=0.0)", lambda x, xp, fp: np.interp(x, xp, fp, left=0.0), None),
        ("np.interp(x, xp, fp, right=0.0)", lambda x, xp, fp: np.interp(x, xp, fp, right=0.0), None),
        ("np.interp(x, xp, fp, period=1.0)", lambda x, xp, fp: np.interp(x, xp, fp, period=1.0), None),
        ("np.interp(0.25, xp, fp)", lambda x, xp, fp: np.interp(0.25, xp, fp), None),
        ("np.interp(x.tolist(), xp, fp)", lambda x, xp, fp: np.interp(x.tolist(), xp, fp), None),
        ("np.interp(x[:5], xp, fp)", lambda x, xp, fp: np.interp(x[:5], xp, fp), None),
        ("np.interp(x[:0], xp, fp)", lambda x, xp, fp: np.interp(x[:0], xp, fp), None),
        ("np.interp(x, xp, fp[:-1])", lambda x, xp, fp: np.interp(x, xp, fp[:-1]), None),
        ("np.interp(x, xp[:0], fp[:0])", lambda x, xp, fp: np.interp(x, xp[:0], fp[:0]), None),
        ("np.interp(x, xp, fp.astype(np.float32))", lambda x, xp, fp: np.interp(x, xp, fp.astype(np.float32)), None),
        ("np.interp(x, xp, fp.astype(np.complex128))", lambda x, xp, fp: np.interp(x, xp, fp.astype(np.complex128)), None),
        ("np.interp(x, xp.astype(np.float32), fp)", lambda x, xp, fp: np.interp(x, xp.astype(np.float32), fp), None),
        ("np.interp(x.astype(np.float32), xp, fp)", lambda x, xp, fp: np.interp(x.astype(np.float32), xp, fp), None),
        ("np.interp(x.astype(np.int64), xp, fp)", lambda x, xp, fp: np.interp(x.astype(np.int64), xp, fp), None),
        ("np.interp(x, xp, xp)", lambda x, xp, fp: np.interp(x, xp, xp), kernel),
        ("np.interp(xp, xp, fp)", lambda x, xp, fp: np.interp(xp, xp, fp), kernel),
    )
    for text, call, expect in spellings:
        yield Case("interp", f"{text} n={n} queries={count}", arguments, call, expect)

    for size in (0, *LAYOUT_SIZES):
        for variant in variants_for("f64", size):
            expect = None if variant.name == "list" else _search_layout_expectation(kernel, variant, size, any_shape=True)
            yield Case("interp", f"np.interp(x, xp, fp) n={n} queries={size} x layout={variant.name}",
                       lambda size=size, variant=variant: (variant.build(values("f64", size, arg="x")), _table(n), values("f64", n, arg="fp")),
                       np.interp, expect if _interp_takes(n, size) else None)
            if size == 0:
                continue
            for which in ("xp", "fp"):
                expect = None if variant.name == "list" else _search_layout_expectation(kernel, variant, size, any_shape=False)
                yield Case("interp", f"np.interp(x, xp, fp) n={size} queries={size} {which} layout={variant.name}",
                           lambda size=size, variant=variant, which=which: (
                               values("f64", size, arg="x"),
                               variant.build(_table(size)) if which == "xp" else _table(size),
                               variant.build(values("f64", size, arg="fp")) if which == "fp" else values("f64", size, arg="fp")),
                           np.interp, expect)


# ------------------------------------------------------------------- where

# Element types np.where(c, a, b) is taken for when `a` and `b` are both
# arrays of it, with the kernel that takes it: int64 elements are moved by the
# float64 kernel as bit patterns.
_WHERE_KERNELS = {"f64": "where_f64", "i64": "where_f64"}
# Element types of two arrays that np.where leaves to NumPy.
_WHERE_NEIGHBOURS = ("f32", "u64", "i32", "u8", "bool")
# Argument name of the condition, which gives it random data of its own.
_CONDITION = "mask"


def _signalling_nan() -> np.float64:
    """Return an np.float64 holding a signalling NaN with a payload, which the result must keep bit for bit."""
    unsigned, signalling, _ = _UNUSUAL_NANS[8]
    return np.array([signalling], unsigned).view(np.float64)[0]


# Scalars given as `a` or `b` beside a float64 array: (text, maker, taken).
# Taken are those that leave NumPy's result float64 and that it stores
# exactly: a Python float, a Python int a double holds exactly, an
# np.float64. The rest change the result's type or value, or raise.
_WHERE_SCALARS: tuple[tuple[str, Callable[[], object], bool], ...] = (
    ("1.5", lambda: 1.5, True),
    ("nan", lambda: float("nan"), True),
    ("-0.0", lambda: -0.0, True),
    ("-inf", lambda: float("-inf"), True),
    ("0", lambda: 0, True),
    ("-2**53", lambda: -(2**53), True),
    ("np.float64(-0.0)", lambda: np.float64(-0.0), True),
    ("np.float64(signalling NaN)", _signalling_nan, True),
    ("2**53 + 1", lambda: 2**53 + 1, False),
    ("10**400", lambda: 10**400, False),
    ("True", lambda: True, False),
    ("np.float32(1.5)", lambda: np.float32(1.5), False),
    ("np.int64(3)", lambda: np.int64(3), False),
    ("1j", lambda: 1j, False),
    ("np.array(1.5)", lambda: np.array(1.5), False),
    ("None", lambda: None, False),
)


def _where(c, a, b):
    """Return `np.where(c, a, b)`."""
    return np.where(c, a, b)


def _built_each(*makers: Callable[[], object]) -> tuple:
    """Build the arguments of a case, one maker per argument, in order."""
    return tuple(maker() for maker in makers)


def _given(value: object) -> object:
    """Return `value`: a maker of an argument that is the same object every time, such as a constant."""
    return value


def _unusual_nans_reversed(n: int) -> np.ndarray:
    """Return the float64 array of `unusual_nans`, back to front, so that its NaNs fall elsewhere."""
    return unusual_nans("f64", n)[::-1].copy()


def _where_value_cases() -> Iterator[Case]:
    """Yield `np.where(c, a, b)` over every value class at every size, all three plain arrays.

    Both sides get the same value class (with data of their own), so the
    sweep covers each class on each side. Then every class of the condition,
    the NaNs the judge does not make, and each scalar taken, on each side.
    """
    for dtype, kernel in _WHERE_KERNELS.items():
        for n in VALUE_SIZES:
            condition = partial(values, "bool", n, arg=_CONDITION)
            sides = zip(value_arrays(dtype, (n,), "a"), value_arrays(dtype, (n,), "b"))
            for (_, _, label, a_maker), (_, _, _, b_maker) in sides:
                yield Case("where", f"np.where(c, a, b) {dtype} {label} and b alike",
                           partial(_built_each, condition, a_maker, b_maker), _where, kernel)
            a_maker = partial(values, dtype, n, arg="a")
            b_maker = partial(values, dtype, n, arg="b")
            for mask_label, mask_maker in _mask_makers(n):
                yield Case("where", f"np.where(c, a, b) {dtype} n={n} c={mask_label.split('=', 1)[1]}",
                           partial(_built_each, mask_maker, a_maker, b_maker), _where, kernel)
    for n in VALUE_SIZES:
        condition = partial(values, "bool", n, arg=_CONDITION)
        yield Case("where", f"np.where(c, a, b) f64 n={n} a and b=signalling and negative NaNs",
                   partial(_built_each, condition, partial(unusual_nans, "f64", n), partial(_unusual_nans_reversed, n)),
                   _where, "where_f64")
        array = partial(values, "f64", n, arg="a")
        for text, scalar, taken in _WHERE_SCALARS:
            if not taken:
                continue
            yield Case("where", f"np.where(c, a, {text}) f64 n={n}", partial(_built_each, condition, array, scalar),
                       _where, "where_f64")
            yield Case("where", f"np.where(c, {text}, b) f64 n={n}", partial(_built_each, condition, scalar, array),
                       _where, "where_f64")


def _where_layout_cases() -> Iterator[Case]:
    """Yield `np.where(c, a, b)` with each of the three arguments in every layout, the others plain.

    Only plain one-dimensional arrays are taken. A two-dimensional array beside
    one-dimensional ones is NumPy's to broadcast or refuse, and a
    zero-dimensional condition is broadcast.
    """
    spelling = Spelling("np.where(c, a, b)", _where, any_shape=False)
    for dtype in (*_WHERE_KERNELS, *_WHERE_NEIGHBOURS):
        kernel = _WHERE_KERNELS.get(dtype)
        for n in (0, *LAYOUT_SIZES):
            plain = {
                "c": partial(values, "bool", n, arg=_CONDITION),
                "a": partial(values, dtype, n, arg="a"),
                "b": partial(values, dtype, n, arg="b"),
            }
            for arg, arg_dtype, name in (("c", "bool", _CONDITION), ("a", dtype, "a"), ("b", dtype, "b")):
                for variant in variants_for(arg_dtype, n):
                    makers = {**plain, arg: partial(_variant_of, arg_dtype, n, variant, name)}
                    yield Case("where", f"np.where(c, a, b) {dtype} n={n} {arg}={variant.name}",
                               partial(_built_each, makers["c"], makers["a"], makers["b"]), _where,
                               expectation(kernel, spelling, variant, n))


def _where_shape_cases(dtype: str, kernel: str) -> Iterator[Case]:
    """Yield `np.where(c, a, b)` with arrays of more than one dimension, and shapes that broadcast.

    Arrays of one shape and C order are taken whatever their number of
    dimensions. Shapes that differ are NumPy's to broadcast or refuse.

    Args:
        dtype: Element type of `a` and `b`, as a spec name.
        kernel: The kernel that takes the call for that type.

    Yields:
        The cases.
    """
    n = ORDINARY_N
    c = partial(values, "bool", n, arg=_CONDITION)
    a = partial(values, dtype, n, arg="a")
    b = partial(values, dtype, n, arg="b")
    shapes = (
        ("all two-dimensional", _reshaped(c), _reshaped(a), _reshaped(b), kernel),
        ("all three-dimensional", _cube(c), _cube(a), _cube(b), kernel),
        ("all two-dimensional, Fortran order", _reshaped(c, "F"), _reshaped(a, "F"), _reshaped(b, "F"), None),
        ("c two-dimensional, Fortran order", _reshaped(c, "F"), _reshaped(a), _reshaped(b), None),
        ("b over the rows of two-dimensional c and a", _reshaped(c), _reshaped(a), lambda: b()[:COLUMNS].copy(), None),
        ("c of one column against two-dimensional a and b", lambda: c()[: n // COLUMNS].reshape(-1, 1).copy(),
         _reshaped(a), _reshaped(b), None),
        ("c of one row against two-dimensional a and b", lambda: c()[:COLUMNS].copy(), _reshaped(a), _reshaped(b), None),
        ("a and b of one element", c, lambda: a()[:1].copy(), lambda: b()[:1].copy(), None),
        ("c two-dimensional, a and b of the same size and one dimension", _reshaped(c), a, b, None),
        ("all empty and two-dimensional", lambda: np.zeros((0, COLUMNS), np.bool_),
         lambda: np.zeros((0, COLUMNS), np_dtype(dtype)), lambda: np.zeros((0, COLUMNS), np_dtype(dtype)), kernel),
    )
    for text, c_maker, a_maker, b_maker, expect in shapes:
        yield Case("where", f"np.where(c, a, b) {dtype} {text}", partial(_built_each, c_maker, a_maker, b_maker),
                   _where, expect)
    if dtype == "f64":
        yield Case("where", "np.where(c, a, nan) f64 c and a two-dimensional",
                   partial(_built_each, _reshaped(c), _reshaped(a), partial(_given, float("nan"))), _where, kernel)
        yield Case("where", "np.where(c, 0, b) f64 c and b three-dimensional",
                   partial(_built_each, _cube(c), partial(_given, 0), _cube(b)), _where, kernel)
        yield Case("where", "np.where(c, a, nan) f64 c two-dimensional, a of one dimension",
                   partial(_built_each, _reshaped(c), a, partial(_given, float("nan"))), _where, None)


# Calls of np.where on plain float64 arrays, written other ways: (text,
# call, taken). The call receives the condition and two float64 arrays.
_WHERE_SPELLINGS: tuple[tuple[str, Callable, bool], ...] = (
    ("np.where(c, a, a)", lambda c, a, b: np.where(c, a, a), True),
    ("np.where(c, b, a)", lambda c, a, b: np.where(c, b, a), True),
    ("np.where(c, x=a, y=b)", lambda c, a, b: np.where(c, x=a, y=b), False),
    ("np.where(c, a)", lambda c, a, b: np.where(c, a), False),
    ("np.where(c, a, b, b)", lambda c, a, b: np.where(c, a, b, b), False),
    ("np.where(c.astype(int), a, b)", lambda c, a, b: np.where(c.astype(np.int64), a, b), False),
    ("np.where(c.view(np.uint8), a, b)", lambda c, a, b: np.where(c.view(np.uint8), a, b), False),
    ("np.where(c, c, c)", lambda c, a, b: np.where(c, c, c), False),
    ("np.where(c, 1.0, 0.0)", lambda c, a, b: np.where(c, 1.0, 0.0), False),
    ("np.where(c, 0, nan)", lambda c, a, b: np.where(c, 0, float("nan")), False),
    ("np.where(c, a, b.astype(np.float32))", lambda c, a, b: np.where(c, a, b.astype(np.float32)), False),
    ("np.where(c, a.astype(np.float32), b)", lambda c, a, b: np.where(c, a.astype(np.float32), b), False),
    ("np.where(c, a.astype(np.int64), b)", lambda c, a, b: np.where(c, a.astype(np.int64), b), False),
    ("np.where(c, a.astype(np.int64), 0)", lambda c, a, b: np.where(c, a.astype(np.int64), 0), False),
    ("np.where(c, 1.5, b.astype(np.int64))", lambda c, a, b: np.where(c, 1.5, b.astype(np.int64)), False),
    ("np.where(c, a, b.astype('>f8'))", lambda c, a, b: np.where(c, a, b.astype(">f8")), False),
)


def where_cases() -> Iterator[Case]:
    """Yield the cases of `np.where(c, a, b)`.

    float64 and int64 pairs are taken, the int64 ones by the float64 kernel
    moving bit patterns, and so are float64 arrays beside a scalar that keeps
    the result float64. Everything else, and a condition holding a byte other
    than 0 or 1, is NumPy's.
    """
    yield from _where_value_cases()
    yield from _where_layout_cases()
    for dtype, kernel in _WHERE_KERNELS.items():
        yield from _where_shape_cases(dtype, kernel)
    for n in (0, *LAYOUT_SIZES):
        condition = partial(values, "bool", n, arg=_CONDITION)
        floats = partial(values, "f64", n, arg="a")
        others = partial(values, "f64", n, arg="b")
        for text, call, taken in _WHERE_SPELLINGS:
            yield Case("where", f"{text} f64 n={n}", partial(_built_each, condition, floats, others), call,
                       "where_f64" if taken else None)
        for dtype in _WHERE_KERNELS:
            array = partial(values, dtype, n, arg="a")
            for text, scalar, taken in _WHERE_SCALARS:
                # Beside an int64 array no scalar is taken: a float changes the
                # result's type, and an int is NumPy's to fit.
                expect = "where_f64" if taken and dtype == "f64" else None
                yield Case("where", f"np.where(c, a, {text}) {dtype} n={n}",
                           partial(_built_each, condition, array, scalar), _where, expect)
                yield Case("where", f"np.where(c, {text}, b) {dtype} n={n}",
                           partial(_built_each, condition, scalar, array), _where, expect)
        if n:
            for byte in BAD_MASK_BYTES:
                bad = partial(_bad_mask, n, byte)
                for dtype in _WHERE_KERNELS:
                    yield Case("where", f"np.where(c, a, b) {dtype} n={n} c bytes are {byte}",
                               partial(_built_each, bad, partial(values, dtype, n, arg="a"),
                                       partial(values, dtype, n, arg="b")), _where, None)
                yield Case("where", f"np.where(c, a, nan) f64 n={n} c bytes are {byte}",
                           partial(_built_each, bad, floats, partial(_given, float("nan"))), _where, None)


# Every operation that has a hook, with its exact cases.
OPERATIONS: dict[str, Callable[[], Iterator[Case]]] = {
    "argmax": partial(argext_cases, "argmax"),
    "argmin": partial(argext_cases, "argmin"),
    "clip": clip_cases,
    "compress": compress_cases,
    "maskfill": maskfill_cases,
    "nonzero": nonzero_cases,
    "cumsum": partial(cumsum_cases, False),
    "astype": astype_cases,
    "sum": sum_cases,
    "count_nonzero": count_nonzero_cases,
    "histogram": histogram_cases,
    "bincount": bincount_cases,
    "isin": isin_cases,
    "searchsorted": searchsorted_cases,
    "interp": interp_cases,
    "where": where_cases,
}

# Operations that have kernels which are not bit-identical to NumPy, with the
# cases that run under install(inexact=True).
INEXACT_OPERATIONS: dict[str, Callable[[], Iterator[Case]]] = {
    "cumsum": partial(cumsum_cases, True),
}
