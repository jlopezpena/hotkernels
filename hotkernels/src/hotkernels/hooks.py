"""What hotkernels installs: the kernels, the hooks that deliver them, and a
self-test for each hook.

A hook is one pointer inside NumPy that the extension module can point at the
binding. This module says which hooks exist, which kernels each one serves,
and how to prove after applying a hook that the kernel really is reached. The
proof is a probe: one small NumPy call with a known answer. `install()` keeps
a hook only if every probe gives the right answer and the kernel's call
counter moved. A NumPy that routes a call differently from the one the hooks
were written for then leaves hotkernels switched off for that call, with
NumPy untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np

# Kinds of hook, as shown to the user.
TYPE_SLOT = "type slot"
INNER_LOOP = "inner loop"
FUNCTION = "function"
METHOD = "method"
INDEXING = "indexing"

# Element count of a probe array. Large enough for every size threshold in
# the binding, small enough that install() stays fast.
PROBE_N = 1000
# Element count of a probe that must be served by a function hook and not by
# the type slot behind it. The slot leaves runs this short to NumPy.
SMALL_PROBE_N = 16
# Position of the one element a probe expects a search to find.
PROBE_INDEX = 617
SMALL_PROBE_INDEX = 11
# Bounds the clip probes clamp to.
PROBE_LO, PROBE_HI = -0.5, 0.5
# How many of every `PROBE_STRIDE` elements a probe selects or sets.
PROBE_STRIDE = 3
# Number of bins and range of the histogram probe.
PROBE_BINS = 4
PROBE_RANGE = (0.0, 4.0)
# Byte value the bincount, sum and astype probes fill their array with.
PROBE_BYTE = 7
# Value the masked-assignment probe writes.
PROBE_FILL = 2.5

# Element count of the sorted array the search probes look in. It differs
# from PROBE_N so that a kernel which confused its two lengths would fail.
PROBE_TABLE_N = 37


@dataclass(frozen=True)
class Probe:
    """One self-test call for one kernel."""

    # Kernel whose call counter must move when `call` runs.
    kernel: str
    # Makes one NumPy call the kernel should take and returns whether the
    # answer was right.
    call: Callable[[], bool]
    # True for a kernel that is not bit-identical to NumPy. Its probe only
    # runs when such kernels were asked for.
    inexact: bool = False


@dataclass(frozen=True)
class Hook:
    """One pointer inside NumPy that can be pointed at the binding."""

    # Name of the hook in the extension module.
    name: str
    # Kind of hook: one of the constants above.
    kind: str
    # One probe per kernel the hook serves.
    probes: tuple[Probe, ...]


@dataclass(frozen=True)
class KernelInfo:
    """How one kernel appears to a user."""

    # Kernel name, as in `kernels/`.
    name: str
    # The NumPy call the kernel stands in for, as a user writes it.
    numpy_call: str
    # Element type of the arrays it handles, as a NumPy type name.
    elements: str
    # Name of the operation in the binding's counters of calls passed on to
    # NumPy, or None for a kernel without a hook.
    operation: str | None
    # Names of the hooks that deliver the kernel. Empty for one without a hook.
    hooks: tuple[str, ...]
    # Why the kernel has no hook, or None when it has one.
    unbound_reason: str | None = None


def _marked(dtype: type, n: int, index: int, value: float) -> np.ndarray:
    """Return `n` zeros of `dtype` with `value` at `index`."""
    x = np.zeros(n, dtype)
    x[index] = value
    return x


def _argext_method(dtype: type, method: str, value: float) -> bool:
    """Probe the type slot: `x.argmax()` or `x.argmin()` on a long array.

    Args:
        dtype: Element type of the probe array.
        method: "argmax" or "argmin".
        value: The one non-zero element: positive for argmax, negative for argmin.

    Returns:
        Whether the method found the marked element.
    """
    return int(getattr(_marked(dtype, PROBE_N, PROBE_INDEX, value), method)()) == PROBE_INDEX


def _argext_function(dtype: type, function: Callable, value: float) -> bool:
    """Probe the function hook: `np.argmax(x)` or `np.argmin(x)` on a short array.

    The array is shorter than the type slot serves, so only the function hook
    can have taken the call.

    Args:
        dtype: Element type of the probe array.
        function: `np.argmax` or `np.argmin`.
        value: The one non-zero element: positive for argmax, negative for argmin.

    Returns:
        Whether the function found the marked element.
    """
    return int(function(_marked(dtype, SMALL_PROBE_N, SMALL_PROBE_INDEX, value))) == SMALL_PROBE_INDEX


def _argext_small_method(dtype: type, method: str, value: float) -> bool:
    """Probe the method hook: `x.argmax()` or `x.argmin()` on a short array.

    The array is shorter than the type slot serves, so only the method hook
    can have taken the call.

    Args:
        dtype: Element type of the probe array.
        method: "argmax" or "argmin".
        value: The one non-zero element: positive for argmax, negative for argmin.

    Returns:
        Whether the method found the marked element.
    """
    x = _marked(dtype, SMALL_PROBE_N, SMALL_PROBE_INDEX, value)
    return int(getattr(x, method)()) == SMALL_PROBE_INDEX


def _clip_with(dtype: type, clip: Callable[[np.ndarray, float, float], np.ndarray]) -> bool:
    """Probe a clip hook: values from -1 to 1 clamped to the probe bounds.

    The loop hook takes every call the function and method hooks take, so
    the probe of those two proves that the result is right and that a clip
    kernel ran, not which hook ran it.

    Args:
        dtype: Element type of the probe array.
        clip: Clips its first argument to the bounds given by the other two.

    Returns:
        Whether the result has the array's element type and the right values.
    """
    x = np.linspace(-1.0, 1.0, PROBE_N).astype(dtype)
    result = clip(x, PROBE_LO, PROBE_HI)
    return result.dtype == dtype and bool(np.array_equal(result, np.minimum(np.maximum(x, PROBE_LO), PROBE_HI)))


def _clip_method(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Return `x.clip(lo, hi)`."""
    return x.clip(lo, hi)


def _clip(dtype: type) -> bool:
    """Probe the clip loop or the function hook: `np.clip(x, lo, hi)`."""
    return _clip_with(dtype, np.clip)


def _stride_mask() -> np.ndarray:
    """Return a boolean mask that is True at every PROBE_STRIDE-th position."""
    return np.arange(PROBE_N) % PROBE_STRIDE == 0


def _compress(dtype: type) -> bool:
    """Probe the indexing hook: `x[mask]`."""
    x = np.arange(PROBE_N).astype(dtype)
    return bool(np.array_equal(x[_stride_mask()], x[::PROBE_STRIDE]))


def _compress_function(dtype: type, function: Callable) -> bool:
    """Probe a function hook: `np.compress(mask, x)` or `np.extract(mask, x)`.

    Args:
        dtype: Element type of the probe array.
        function: `np.compress` or `np.extract`; both take the mask first.

    Returns:
        Whether the function selected every PROBE_STRIDE-th element.
    """
    x = np.arange(PROBE_N).astype(dtype)
    return bool(np.array_equal(function(_stride_mask(), x), x[::PROBE_STRIDE]))


def _assign(x: np.ndarray, mask: np.ndarray) -> None:
    """Run `x[mask] = PROBE_FILL`."""
    x[mask] = PROBE_FILL


def _putmask(x: np.ndarray, mask: np.ndarray) -> None:
    """Run `np.putmask(x, mask, PROBE_FILL)`."""
    np.putmask(x, mask, PROBE_FILL)


def _copyto(x: np.ndarray, mask: np.ndarray) -> None:
    """Run `np.copyto(x, PROBE_FILL, where=mask)`."""
    np.copyto(x, PROBE_FILL, where=mask)


def _maskfill(assign: Callable[[np.ndarray, np.ndarray], None]) -> bool:
    """Probe a masked assignment: `x[mask] = value` or one of its function forms.

    Args:
        assign: Sets the elements of its first argument where its second is
            True to PROBE_FILL, in place.

    Returns:
        Whether exactly the selected elements were set.
    """
    x = np.arange(PROBE_N, dtype=np.float64)
    expected = x.copy()
    expected[::PROBE_STRIDE] = PROBE_FILL
    assign(x, _stride_mask())
    return bool(np.array_equal(x, expected))


def _nonzero_method(mask: np.ndarray) -> tuple:
    """Return `mask.nonzero()`."""
    return mask.nonzero()


def _nonzero(function: Callable[[np.ndarray], tuple]) -> bool:
    """Probe `mask.nonzero()` or `np.where(mask)`.

    Args:
        function: Takes the mask and returns NumPy's tuple of index arrays.

    Returns:
        Whether the one index array holds every PROBE_STRIDE-th position.
    """
    (indices,) = function(_stride_mask())
    return bool(np.array_equal(indices, np.arange(0, PROBE_N, PROBE_STRIDE)))


def _where() -> bool:
    """Probe the function hook: `np.where(mask, a, b)` on float64, with a mask True at every PROBE_STRIDE-th position.

    The two arrays differ at every position, so an element taken from the
    wrong one fails the probe.
    """
    a = np.arange(PROBE_N, dtype=np.float64)
    b = -a - 1.0
    expected = b.copy()
    expected[::PROBE_STRIDE] = a[::PROBE_STRIDE]
    result = np.where(_stride_mask(), a, b)
    return result.dtype == np.float64 and bool(np.array_equal(result, expected))


def _cumsum(dtype: type) -> bool:
    """Probe the method hook: `x.cumsum()` on ones, whose running sum is exact in every type."""
    return bool(np.array_equal(np.ones(PROBE_N, dtype).cumsum(), np.arange(1, PROBE_N + 1).astype(dtype)))


def _cumsum_bool() -> bool:
    """Probe the method hook: `mask.cumsum()` on a mask that is True at every PROBE_STRIDE-th position."""
    mask = _stride_mask()
    result = mask.cumsum()
    return result.dtype == np.intp and bool(np.array_equal(result, np.arange(PROBE_N) // PROBE_STRIDE + 1))


def _astype() -> bool:
    """Probe the method hook: `x.astype(np.float32)` on bytes."""
    result = np.full(PROBE_N, PROBE_BYTE, np.uint8).astype(np.float32)
    return result.dtype == np.float32 and bool(np.all(result == PROBE_BYTE))


def _sum_method() -> bool:
    """Probe the method hook: `x.sum()` on bytes."""
    return int(np.full(PROBE_N, PROBE_BYTE, np.uint8).sum()) == PROBE_N * PROBE_BYTE


def _sum_function() -> bool:
    """Probe the function hook: `np.sum(x)` on bytes."""
    return int(np.sum(np.full(PROBE_N, PROBE_BYTE, np.uint8))) == PROBE_N * PROBE_BYTE


def _count_nonzero(dtype: type) -> bool:
    """Probe the function hook: `np.count_nonzero(x)`."""
    return int(np.count_nonzero(_stride_mask().astype(dtype))) == len(range(0, PROBE_N, PROBE_STRIDE))


def _histogram() -> bool:
    """Probe the function hook: `np.histogram` on values that fill each bin equally."""
    x = (np.arange(PROBE_N) % PROBE_BINS) + 0.5
    counts, edges = np.histogram(x, PROBE_BINS, PROBE_RANGE)
    return bool(np.all(counts == PROBE_N // PROBE_BINS)) and len(edges) == PROBE_BINS + 1


def _bincount() -> bool:
    """Probe the function hook: `np.bincount` on equal bytes."""
    counts = np.bincount(np.full(PROBE_N, PROBE_BYTE, np.uint8))
    return len(counts) == PROBE_BYTE + 1 and int(counts[PROBE_BYTE]) == PROBE_N


def _isin() -> bool:
    """Probe the function hook: `np.isin` with two test values."""
    x = np.arange(PROBE_N, dtype=np.int64)
    result = np.isin(x, [PROBE_INDEX, SMALL_PROBE_INDEX])
    return int(result.sum()) == 2 and bool(result[PROBE_INDEX]) and bool(result[SMALL_PROBE_INDEX])


def _searchsorted() -> bool:
    """Probe the method hook: `np.searchsorted` for needles between whole numbers."""
    table = np.arange(PROBE_TABLE_N, dtype=np.float64)
    needles = (np.arange(PROBE_N) % PROBE_TABLE_N) + 0.5
    expected = (np.arange(PROBE_N) % PROBE_TABLE_N) + 1
    return bool(np.array_equal(np.searchsorted(table, needles), expected))


def _searchsorted_right() -> bool:
    """Probe the method hook: `np.searchsorted` with `side='right'` for needles that are elements.

    Every needle equals one element, so the left side's answer is one less at
    each of them: a call that reached the wrong kernel fails.
    """
    table = np.arange(PROBE_TABLE_N, dtype=np.float64)
    needles = (np.arange(PROBE_N) % PROBE_TABLE_N).astype(np.float64)
    expected = (np.arange(PROBE_N) % PROBE_TABLE_N) + 1
    return bool(np.array_equal(np.searchsorted(table, needles, side="right"), expected))


def _interp() -> bool:
    """Probe the function hook: `np.interp` through points on the line y = 2x.

    Halves and doubles of small whole numbers are exact, so the answer is too.
    """
    points = np.arange(PROBE_TABLE_N, dtype=np.float64)
    queries = (np.arange(PROBE_N) % (PROBE_TABLE_N - 1)) + 0.5
    return bool(np.array_equal(np.interp(queries, points, 2.0 * points), 2.0 * queries))


# Every hook, in the order install() applies them: the deepest first.
HOOKS: tuple[Hook, ...] = (
    Hook("slot.argmax.f32", TYPE_SLOT, (Probe("argmax_f32", partial(_argext_method, np.float32, "argmax", 1.0)),)),
    Hook("slot.argmax.f64", TYPE_SLOT, (Probe("argmax_f64", partial(_argext_method, np.float64, "argmax", 1.0)),)),
    Hook("slot.argmin.f32", TYPE_SLOT, (Probe("argmin_f32", partial(_argext_method, np.float32, "argmin", -1.0)),)),
    Hook("slot.argmin.f64", TYPE_SLOT, (Probe("argmin_f64", partial(_argext_method, np.float64, "argmin", -1.0)),)),
    Hook("loop.clip.f32", INNER_LOOP, (Probe("clip_f32", partial(_clip, np.float32)),)),
    Hook("loop.clip.f64", INNER_LOOP, (Probe("clip_f64", partial(_clip, np.float64)),)),
    Hook("function.argmax", FUNCTION, (
        Probe("argmax_f32", partial(_argext_function, np.float32, np.argmax, 1.0)),
        Probe("argmax_f64", partial(_argext_function, np.float64, np.argmax, 1.0)),
    )),
    Hook("function.argmin", FUNCTION, (
        Probe("argmin_f32", partial(_argext_function, np.float32, np.argmin, -1.0)),
        Probe("argmin_f64", partial(_argext_function, np.float64, np.argmin, -1.0)),
    )),
    Hook("function.count_nonzero", FUNCTION, (
        Probe("count_nonzero_i64", partial(_count_nonzero, np.int64)),
        Probe("count_nonzero_f64", partial(_count_nonzero, np.float64)),
    )),
    Hook("function.sum", FUNCTION, (Probe("sum_u8", _sum_function),)),
    Hook("function.histogram", FUNCTION, (Probe("histogram_f64", _histogram),)),
    Hook("function.bincount", FUNCTION, (Probe("bincount_u8", _bincount),)),
    Hook("function.isin", FUNCTION, (Probe("isin_i64", _isin),)),
    Hook("function.interp", FUNCTION, (Probe("interp_f64", _interp),)),
    Hook("function.compress", FUNCTION, (
        Probe("compress_f32", partial(_compress_function, np.float32, np.compress)),
        Probe("compress_f64", partial(_compress_function, np.float64, np.compress)),
    )),
    Hook("function.extract", FUNCTION, (
        Probe("compress_f32", partial(_compress_function, np.float32, np.extract)),
        Probe("compress_f64", partial(_compress_function, np.float64, np.extract)),
    )),
    Hook("function.putmask", FUNCTION, (Probe("maskfill_f64", partial(_maskfill, _putmask)),)),
    Hook("function.copyto", FUNCTION, (Probe("maskfill_f64", partial(_maskfill, _copyto)),)),
    Hook("function.where", FUNCTION, (
        Probe("nonzero_bool", partial(_nonzero, np.where)),
        Probe("where_f64", _where),
    )),
    Hook("function.clip", FUNCTION, (
        Probe("clip_f32", partial(_clip, np.float32)),
        Probe("clip_f64", partial(_clip, np.float64)),
    )),
    Hook("method.astype", METHOD, (Probe("cast_u8_f32", _astype),)),
    Hook("method.sum", METHOD, (Probe("sum_u8", _sum_method),)),
    Hook("method.nonzero", METHOD, (Probe("nonzero_bool", partial(_nonzero, _nonzero_method)),)),
    Hook("method.cumsum", METHOD, (
        Probe("cumsum_i64", partial(_cumsum, np.int64)),
        Probe("cumsum_f32", partial(_cumsum, np.float32), inexact=True),
        Probe("cumsum_f64", partial(_cumsum, np.float64), inexact=True),
        Probe("cumsum_bool", _cumsum_bool),
    )),
    Hook("method.searchsorted", METHOD, (
        Probe("searchsorted_f64", _searchsorted),
        Probe("searchsorted_right_f64", _searchsorted_right),
    )),
    Hook("method.argmax", METHOD, (
        Probe("argmax_f32", partial(_argext_small_method, np.float32, "argmax", 1.0)),
        Probe("argmax_f64", partial(_argext_small_method, np.float64, "argmax", 1.0)),
    )),
    Hook("method.argmin", METHOD, (
        Probe("argmin_f32", partial(_argext_small_method, np.float32, "argmin", -1.0)),
        Probe("argmin_f64", partial(_argext_small_method, np.float64, "argmin", -1.0)),
    )),
    Hook("method.clip", METHOD, (
        Probe("clip_f32", partial(_clip_with, np.float32, _clip_method)),
        Probe("clip_f64", partial(_clip_with, np.float64, _clip_method)),
    )),
    Hook("index.get", INDEXING, (
        Probe("compress_f32", partial(_compress, np.float32)),
        Probe("compress_f64", partial(_compress, np.float64)),
    )),
    Hook("index.set", INDEXING, (Probe("maskfill_f64", partial(_maskfill, _assign)),)),
)

# The hooks that deliver the compress kernels: `x[mask]`, `np.compress` and `np.extract`.
COMPRESS_HOOKS = ("index.get", "function.compress", "function.extract")

# Every kernel, in the order the report lists them. Kernels of one operation
# are adjacent.
KERNELS: tuple[KernelInfo, ...] = (
    KernelInfo("argmax_f32", "np.argmax", "float32", "argmax", ("slot.argmax.f32", "function.argmax", "method.argmax")),
    KernelInfo("argmax_f64", "np.argmax", "float64", "argmax", ("slot.argmax.f64", "function.argmax", "method.argmax")),
    KernelInfo("argmin_f32", "np.argmin", "float32", "argmin", ("slot.argmin.f32", "function.argmin", "method.argmin")),
    KernelInfo("argmin_f64", "np.argmin", "float64", "argmin", ("slot.argmin.f64", "function.argmin", "method.argmin")),
    KernelInfo("clip_f32", "np.clip", "float32", "clip", ("loop.clip.f32", "function.clip", "method.clip")),
    KernelInfo("clip_f64", "np.clip", "float64", "clip", ("loop.clip.f64", "function.clip", "method.clip")),
    KernelInfo("compress_f32", "x[mask]", "float32", "compress", COMPRESS_HOOKS),
    KernelInfo("compress_f64", "x[mask]", "float64", "compress", COMPRESS_HOOKS),
    KernelInfo("maskfill_f64", "x[mask] = v", "float64", "maskfill", ("index.set", "function.putmask", "function.copyto")),
    KernelInfo("nonzero_bool", "np.nonzero", "bool", "nonzero", ("method.nonzero", "function.where")),
    KernelInfo("cumsum_i64", "np.cumsum", "int64", "cumsum", ("method.cumsum",)),
    KernelInfo("cumsum_f32", "np.cumsum", "float32", "cumsum", ("method.cumsum",)),
    KernelInfo("cumsum_f64", "np.cumsum", "float64", "cumsum", ("method.cumsum",)),
    KernelInfo("cumsum_bool", "np.cumsum", "bool", "cumsum", ("method.cumsum",)),
    KernelInfo("cast_u8_f32", "x.astype(float32)", "uint8", "astype", ("method.astype",)),
    KernelInfo("sum_u8", "np.sum", "uint8", "sum", ("function.sum", "method.sum")),
    KernelInfo("count_nonzero_i64", "np.count_nonzero", "int64", "count_nonzero", ("function.count_nonzero",)),
    KernelInfo("count_nonzero_f64", "np.count_nonzero", "float64", "count_nonzero", ("function.count_nonzero",)),
    KernelInfo("histogram_f64", "np.histogram", "float64", "histogram", ("function.histogram",)),
    KernelInfo("bincount_u8", "np.bincount", "uint8", "bincount", ("function.bincount",)),
    KernelInfo("isin_i64", "np.isin", "int64", "isin", ("function.isin",)),
    KernelInfo("searchsorted_f64", "np.searchsorted", "float64", "searchsorted", ("method.searchsorted",)),
    # Indented under the row above: the same call with one argument more, in a
    # column the report keeps to 19 characters.
    KernelInfo("searchsorted_right_f64", "  side='right'", "float64", "searchsorted", ("method.searchsorted",)),
    KernelInfo("interp_f64", "np.interp", "float64", "interp", ("function.interp",)),
    KernelInfo("where_f64", "np.where(c, a, b)", "float64", "where", ("function.where",)),
)
