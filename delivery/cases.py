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
# Element count below which the type slot leaves argmax to NumPy. Mirrors
# HK_SLOT_MIN_N in the binding.
SLOT_MIN_N = 128
# Columns of the two-dimensional layouts.
COLUMNS = 2
# Bytes other than 1 that a boolean array can hold for True. NumPy accepts
# them; the mask kernels must never see them.
BAD_MASK_BYTES = (2, 255)
# Element count of the one-off cases that need a single ordinary size.
ORDINARY_N = 1000
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
            if spelling.kind != TAKES:
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
        The cases. The function hook takes a plain array of any size; the
        type slot behind every other spelling takes runs from SLOT_MIN_N
        elements, and NumPy may copy an array into shape before the slot
        sees it.
    """
    function, nan_function = getattr(np, name), getattr(np, f"nan{name}")
    spellings = (
        Spelling(f"np.{name}(x)", function, normalises=True),
        Spelling(f"x.{name}()", lambda x: getattr(x, name)(), min_n=SLOT_MIN_N, normalises=True),
        Spelling(f"np.{name}(x, axis=0)", lambda x: function(x, axis=0), MAYBE),
        Spelling(f"np.{name}(x, axis=-1)", lambda x: function(x, axis=-1), MAYBE),
        Spelling(f"np.{name}(a=x)", lambda x: function(a=x), MAYBE),
        Spelling(f"np.{name}(x, keepdims=True)", lambda x: function(x, keepdims=True), MAYBE),
        Spelling(f"np.{name}(x, out=o)", lambda x: function(x, out=np.empty((), np.intp)), MAYBE),
        Spelling(f"np.nan{name}(x)", nan_function, MAYBE),
    )
    kernels = {"f32": f"{name}_f32", "f64": f"{name}_f64"}
    # NumPy raises on an empty array, so no kernel takes one.
    yield from unary_cases(name, kernels, ("i64", "i32", "u8"), spellings, empty_takes=False)
    yield from unusual_nan_cases(name, kernels, f"np.{name}(x)", function)
    yield from unusual_nan_cases(name, kernels, f"x.{name}()", spellings[1].call, SLOT_MIN_N)


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


def _clip_into_fresh(x, lo, hi):
    """Clip `x` into a new array passed as `out`."""
    return np.clip(x, lo, hi, out=np.empty_like(x))


def clip_cases() -> Iterator[Case]:
    """Yield the cases of `clip`.

    The kernels take contiguous data with scalar bounds, lo <= hi, when the
    output is not the input. NumPy buffers arrays that are misaligned or in
    the other byte order and hands the loop contiguous chunks, so those may
    reach the kernel too.
    """
    kernels = {"f32": "clip_f32", "f64": "clip_f64"}
    for dtype, kernel in kernels.items():
        default = next(_clip_bounds(kernel))
        for set_name, lo, hi in _clip_bounds(kernel):
            sizes = VALUE_SIZES if set_name in (DEFAULT_SCALARS, "zero_lo") else FEW_SIZES
            for _, _, label, maker in value_arrays(dtype, sizes):
                yield Case("clip", f"np.clip(x, {lo!r}, {hi!r}) {dtype} {label}", partial(_made, maker),
                           partial(_clip3, np.clip, lo, hi), kernel)
        for _, _, label, maker in value_arrays(dtype, FEW_SIZES):
            yield Case("clip", f"x.clip(lo, hi) {dtype} {label}", partial(_made, maker),
                       partial(_clip_method, default[1], default[2]), kernel)
    yield from unusual_nan_cases("clip", kernels, "np.clip(x, -0.5, 0.5)", partial(_clip3, np.clip, -0.5, 0.5))
    yield from unusual_nan_cases("clip", kernels, "np.clip(x, 0.0, 1.0)", partial(_clip3, np.clip, 0.0, 1.0))
    lo, hi = -0.5, 0.5
    spellings = (
        Spelling("np.clip(x, lo, hi)", partial(_clip3, np.clip, lo, hi), normalises=True),
        Spelling("x.clip(lo, hi)", partial(_clip_method, lo, hi), normalises=True),
        Spelling("np.clip(x, lo, hi, out=fresh)", lambda x: _clip_into_fresh(x, lo, hi), normalises=True),
        Spelling("np.clip(x, min=lo, max=hi)", lambda x: np.clip(x, min=lo, max=hi), normalises=True),
        Spelling("np.clip(x, np.float64(lo), np.float64(hi))", lambda x: np.clip(x, np.float64(lo), np.float64(hi)), MAYBE),
        Spelling("np.clip(x, lo, hi, out=x)", lambda x: _clip_in_place(x, lo, hi), NEVER, normalises=True),
        Spelling("np.clip(x, lo_array, hi_array)", lambda x: np.clip(x, np.full(np.shape(x), lo), np.full(np.shape(x), hi)), MAYBE),
        Spelling("np.clip(x, lo, None)", lambda x: np.clip(x, lo, None), NEVER),
        Spelling("np.clip(x, None, hi)", lambda x: np.clip(x, None, hi), NEVER),
        Spelling("np.clip(x, hi, lo) reversed", lambda x: np.clip(x, hi, lo), NEVER),
        Spelling("np.clip(x, nan, hi)", lambda x: np.clip(x, float("nan"), hi), NEVER),
        Spelling("np.clip(x, lo, nan)", lambda x: np.clip(x, lo, float("nan")), NEVER),
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


def compress_cases() -> Iterator[Case]:
    """Yield the cases of `x[mask]`."""
    kernels = {"f32": "compress_f32", "f64": "compress_f64"}
    select = Spelling("x[mask]", lambda x, mask: x[mask], any_shape=False)
    others = (
        Spelling("np.compress(mask, x)", lambda x, mask: np.compress(mask, x), NEVER),
        Spelling("np.extract(mask, x)", lambda x, mask: np.extract(mask, x), NEVER),
        Spelling("x[mask,]", lambda x, mask: x[mask,], NEVER),
        Spelling("x[mask][:3]", lambda x, mask: x[mask][:3], any_shape=False),
    )
    for dtype, kernel in kernels.items():
        for n in VALUE_SIZES:
            for mask_label, mask_maker in _mask_makers(n):
                for x_class in ("random", "nan_some"):
                    make = partial(_values_and_mask, dtype, n, x_class, mask_maker)
                    yield Case("compress", f"x[mask] {dtype} n={n} x={x_class} {mask_label}", make, select.call, kernel)
    for dtype in ("f32", "f64", "i64", "u8"):
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
                for byte in BAD_MASK_BYTES:
                    yield Case("compress", f"x[mask] {dtype} n={n} mask bytes are {byte}",
                               partial(_x_and_mask, dtype, n, CONTIGUOUS, partial(_bad_mask, n, byte)), select.call, None)
                short = partial(values, "bool", n - 1, arg="mask")
                yield Case("compress", f"x[mask] {dtype} n={n} mask one short",
                           partial(_x_and_mask, dtype, n, CONTIGUOUS, short), select.call, None)
        rows = next(v for v in VARIANTS if v.name == "rows")
        yield Case("compress", f"x[mask] {dtype} two-dimensional",
                   partial(_x_and_mask, dtype, ORDINARY_N, rows, lambda: _rows(values("bool", ORDINARY_N, arg="mask"))),
                   select.call, None)


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


def maskfill_cases() -> Iterator[Case]:
    """Yield the cases of `x[mask] = value`."""
    kernel = "maskfill_f64"
    for n in VALUE_SIZES:
        for mask_label, mask_maker in _mask_makers(n):
            for text, value, _ in _FILL_VALUES[:4]:
                yield Case("maskfill", f"x[mask] = {text} f64 n={n} {mask_label}",
                           partial(_values_and_mask, "f64", n, "random", mask_maker), partial(_assign, value), kernel)
    for dtype in ("f64", "f32", "i64"):
        for n in (0, *LAYOUT_SIZES):
            plain_mask = partial(values, "bool", n, arg="mask")
            takes = kernel if dtype == "f64" else None
            for text, value, taken in _FILL_VALUES:
                yield Case("maskfill", f"x[mask] = {text} {dtype} n={n}",
                           partial(_x_and_mask, dtype, n, CONTIGUOUS, plain_mask), partial(_assign, value),
                           takes if taken else None)
            for function in (_putmask, _copyto):
                yield Case("maskfill", f"{function.__name__[1:]} {dtype} n={n}",
                           partial(_x_and_mask, dtype, n, CONTIGUOUS, plain_mask), partial(function, 1.5), None)
            for variant in variants_for(dtype, n):
                if variant.dims != 1 or variant.name == "list":
                    continue
                expect = takes if variant.plain and variant.writeable else None
                if takes and (not variant.predictable or (n == 0 and not variant.plain)):
                    expect = ANY
                yield Case("maskfill", f"x[mask] = 1.5 {dtype} n={n} x={variant.name}",
                           partial(_x_and_mask, dtype, n, variant, plain_mask), partial(_assign, 1.5), expect)
            for variant in variants_for("bool", n):
                if variant.dims != 1:
                    continue
                yield Case("maskfill", f"x[mask] = 1.5 {dtype} n={n} mask={variant.name}",
                           partial(_x_and_mask, dtype, n, CONTIGUOUS, partial(_variant_of, "bool", n, variant, "mask")),
                           partial(_assign, 1.5),
                           takes if variant.plain else (ANY if takes and n == 0 else None))
            if n:
                for byte in BAD_MASK_BYTES:
                    yield Case("maskfill", f"x[mask] = 1.5 {dtype} n={n} mask bytes are {byte}",
                               partial(_x_and_mask, dtype, n, CONTIGUOUS, partial(_bad_mask, n, byte)),
                               partial(_assign, 1.5), None)
    yield Case("maskfill", "x[mask] = 1.5 where mask is a view of x", partial(_overlapping, ORDINARY_N), partial(_assign, 1.5), None)


def nonzero_cases() -> Iterator[Case]:
    """Yield the cases of `nonzero` on booleans."""
    spellings = (
        Spelling("mask.nonzero()", lambda m: m.nonzero(), any_shape=False),
        Spelling("np.nonzero(mask)", np.nonzero, any_shape=False, normalises=True),
        Spelling("np.flatnonzero(mask)", np.flatnonzero, normalises=True),
        Spelling("np.argwhere(mask)", np.argwhere, MAYBE),
        Spelling("np.where(mask)", np.where, NEVER),
    )
    # The kernel is not called for an empty mask's sake: it takes it and returns nothing.
    yield from unary_cases("nonzero", {"bool": "nonzero_bool"}, ("u8", "f64", "i64"), spellings)
    for n in LAYOUT_SIZES:
        for byte in BAD_MASK_BYTES:
            yield Case("nonzero", f"mask.nonzero() n={n} mask bytes are {byte}",
                       lambda n=n, byte=byte: (_bad_mask(n, byte),), spellings[0].call, None)


def cumsum_cases(inexact: bool) -> Iterator[Case]:
    """Yield the cases of `cumsum`.

    Args:
        inexact: False for the int64 kernel and the float kernels' absence;
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
        yield from unary_cases("cumsum", {"i64": "cumsum_i64"}, ("f32", "f64", "i32", "u8"), spellings,
                               empty_takes=False)


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


def sum_cases() -> Iterator[Case]:
    """Yield the cases of `sum` on uint8."""
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
    yield from unary_cases("sum", {"u8": "sum_u8"}, ("i8", "u16", "bool", "f64"), spellings)


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


def searchsorted_cases() -> Iterator[Case]:
    """Yield the cases of `searchsorted` on float64."""
    kernel = "searchsorted_f64"
    search = np.searchsorted
    for n, count in itertools.product(TABLE_SIZES, NEEDLE_COUNTS):
        # Every class of needle on an ordinary array, then every class of array.
        pairs = [("random", needle_class) for needle_class in (*NEEDLE_CLASSES, "hits")]
        pairs += [(table_class, needle_class) for table_class in TABLE_CLASSES[1:] for needle_class in ("random", "hits")]
        for table_class, needle_class in pairs:
            yield Case("searchsorted", f"np.searchsorted(a, v) a={table_class} n={n} v={needle_class} needles={count}",
                       partial(_search_arguments, n, table_class, count, needle_class), search, kernel)
    for n in (LARGE_TABLE_N, BEYOND_LARGE_TABLE_N):
        for count in (1, 64, 127, 128, 129, 1000):
            yield Case("searchsorted", f"np.searchsorted(a, v) a=random n={n} v=nan_some needles={count}",
                       partial(_search_arguments, n, "random", count, "nan_some"), search, kernel)

    n, count = ORDINARY_N, 100
    arguments = partial(_search_arguments, n, "random", count, "random")
    spellings = (
        ("a.searchsorted(v)", lambda a, v: a.searchsorted(v), kernel),
        ("np.searchsorted(a, v, 'left')", lambda a, v: np.searchsorted(a, v, "left"), kernel),
        ("np.searchsorted(a, v, side='left', sorter=None)", lambda a, v: np.searchsorted(a, v, side="left", sorter=None), kernel),
        ("a.searchsorted(v=v)", lambda a, v: a.searchsorted(v=v), kernel),
        ("np.searchsorted(a, v, side='right')", lambda a, v: np.searchsorted(a, v, side="right"), None),
        ("np.searchsorted(a, v, sorter=np.arange(len(a)))", lambda a, v: np.searchsorted(a, v, sorter=np.arange(len(a))), None),
        ("np.digitize(v, a)", lambda a, v: np.digitize(v, a), None),
        ("np.digitize(v, a, right=True)", lambda a, v: np.digitize(v, a, right=True), kernel),
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
    )
    for text, call, expect in spellings:
        yield Case("searchsorted", f"{text} n={n} needles={count}", arguments, call, expect)

    for size in (0, *LAYOUT_SIZES):
        for variant in variants_for("f64", size):
            # A list has no method of its own: NumPy converts it and calls the
            # method of the array it made.
            expect = ANY if variant.name == "list" else _search_layout_expectation(kernel, variant, size, any_shape=False)
            yield Case("searchsorted", f"np.searchsorted(a, v) n={size} needles={count} a layout={variant.name}",
                       lambda size=size, variant=variant: (variant.build(_table(size)), values("f64", count, arg="v")),
                       search, expect)
            expect = None if variant.name == "list" else _search_layout_expectation(kernel, variant, size, any_shape=True)
            yield Case("searchsorted", f"np.searchsorted(a, v) n={n} needles={size} v layout={variant.name}",
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
    for n, count in ((1, 100), (2, 100), (ORDINARY_N, 100), (ORDINARY_N, ORDINARY_N)):
        for position in sorted({0, n // 2, n - 1}):
            yield Case("interp", f"np.interp(x, xp, fp) n={n} queries={count} xp has a NaN at {position}",
                       partial(_interp_with_nan, n, count, position), np.interp, None)

    n, count = ORDINARY_N, 100
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
}

# Operations that have kernels which are not bit-identical to NumPy, with the
# cases that run under install(inexact=True).
INEXACT_OPERATIONS: dict[str, Callable[[], Iterator[Case]]] = {
    "cumsum": partial(cumsum_cases, True),
}
