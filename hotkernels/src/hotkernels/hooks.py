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

# Why the two search kernels have no hook.
NEEDS_SECOND_SIZE = "not bound: the kernel takes one length for both of its inputs"


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


def _clip(dtype: type) -> bool:
    """Probe the clip loop: values from -1 to 1 clamped to the probe bounds."""
    x = np.linspace(-1.0, 1.0, PROBE_N).astype(dtype)
    return bool(np.array_equal(np.clip(x, PROBE_LO, PROBE_HI), np.minimum(np.maximum(x, PROBE_LO), PROBE_HI)))


def _stride_mask() -> np.ndarray:
    """Return a boolean mask that is True at every PROBE_STRIDE-th position."""
    return np.arange(PROBE_N) % PROBE_STRIDE == 0


def _compress(dtype: type) -> bool:
    """Probe the indexing hook: `x[mask]`."""
    x = np.arange(PROBE_N).astype(dtype)
    return bool(np.array_equal(x[_stride_mask()], x[::PROBE_STRIDE]))


def _maskfill() -> bool:
    """Probe the assignment hook: `x[mask] = value`."""
    x = np.arange(PROBE_N, dtype=np.float64)
    expected = x.copy()
    expected[::PROBE_STRIDE] = PROBE_FILL
    x[_stride_mask()] = PROBE_FILL
    return bool(np.array_equal(x, expected))


def _nonzero() -> bool:
    """Probe the method hook: `mask.nonzero()`."""
    (indices,) = _stride_mask().nonzero()
    return bool(np.array_equal(indices, np.arange(0, PROBE_N, PROBE_STRIDE)))


def _cumsum(dtype: type) -> bool:
    """Probe the method hook: `x.cumsum()` on ones, whose running sum is exact in every type."""
    return bool(np.array_equal(np.ones(PROBE_N, dtype).cumsum(), np.arange(1, PROBE_N + 1).astype(dtype)))


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
    Hook("method.astype", METHOD, (Probe("cast_u8_f32", _astype),)),
    Hook("method.sum", METHOD, (Probe("sum_u8", _sum_method),)),
    Hook("method.nonzero", METHOD, (Probe("nonzero_bool", _nonzero),)),
    Hook("method.cumsum", METHOD, (
        Probe("cumsum_i64", partial(_cumsum, np.int64)),
        Probe("cumsum_f32", partial(_cumsum, np.float32), inexact=True),
        Probe("cumsum_f64", partial(_cumsum, np.float64), inexact=True),
    )),
    Hook("index.get", INDEXING, (
        Probe("compress_f32", partial(_compress, np.float32)),
        Probe("compress_f64", partial(_compress, np.float64)),
    )),
    Hook("index.set", INDEXING, (Probe("maskfill_f64", _maskfill),)),
)

# Every kernel, in the order the report lists them. Kernels of one operation
# are adjacent.
KERNELS: tuple[KernelInfo, ...] = (
    KernelInfo("argmax_f32", "np.argmax", "float32", "argmax", ("slot.argmax.f32", "function.argmax")),
    KernelInfo("argmax_f64", "np.argmax", "float64", "argmax", ("slot.argmax.f64", "function.argmax")),
    KernelInfo("argmin_f32", "np.argmin", "float32", "argmin", ("slot.argmin.f32", "function.argmin")),
    KernelInfo("argmin_f64", "np.argmin", "float64", "argmin", ("slot.argmin.f64", "function.argmin")),
    KernelInfo("clip_f32", "np.clip", "float32", "clip", ("loop.clip.f32",)),
    KernelInfo("clip_f64", "np.clip", "float64", "clip", ("loop.clip.f64",)),
    KernelInfo("compress_f32", "x[mask]", "float32", "compress", ("index.get",)),
    KernelInfo("compress_f64", "x[mask]", "float64", "compress", ("index.get",)),
    KernelInfo("maskfill_f64", "x[mask] = v", "float64", "maskfill", ("index.set",)),
    KernelInfo("nonzero_bool", "np.nonzero", "bool", "nonzero", ("method.nonzero",)),
    KernelInfo("cumsum_i64", "np.cumsum", "int64", "cumsum", ("method.cumsum",)),
    KernelInfo("cumsum_f32", "np.cumsum", "float32", "cumsum", ("method.cumsum",)),
    KernelInfo("cumsum_f64", "np.cumsum", "float64", "cumsum", ("method.cumsum",)),
    KernelInfo("cast_u8_f32", "x.astype(float32)", "uint8", "astype", ("method.astype",)),
    KernelInfo("sum_u8", "np.sum", "uint8", "sum", ("function.sum", "method.sum")),
    KernelInfo("count_nonzero_i64", "np.count_nonzero", "int64", "count_nonzero", ("function.count_nonzero",)),
    KernelInfo("count_nonzero_f64", "np.count_nonzero", "float64", "count_nonzero", ("function.count_nonzero",)),
    KernelInfo("histogram_f64", "np.histogram", "float64", "histogram", ("function.histogram",)),
    KernelInfo("bincount_u8", "np.bincount", "uint8", "bincount", ("function.bincount",)),
    KernelInfo("isin_i64", "np.isin", "int64", "isin", ("function.isin",)),
    KernelInfo("searchsorted_f64", "np.searchsorted", "float64", None, (), NEEDS_SECOND_SIZE),
    KernelInfo("interp_f64", "np.interp", "float64", None, (), NEEDS_SECOND_SIZE),
)
