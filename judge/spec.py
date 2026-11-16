"""The kernel spec format.

Each kernel directory has a `spec.py` that defines `KERNEL = Kernel(...)`. The
spec tells the judge how to call the kernel and what the right answer is.

The format is provisional: it is the smallest thing that covers the kernels we
have. Extend it when a kernel needs something it cannot express.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

# Element types a buffer can have: spec name -> (NumPy dtype, C type).
DTYPES: dict[str, tuple[type, str]] = {
    "f32": (np.float32, "float"),
    "f64": (np.float64, "double"),
    "i8": (np.int8, "int8_t"),
    "i16": (np.int16, "int16_t"),
    "i32": (np.int32, "int32_t"),
    "i64": (np.int64, "int64_t"),
    "u8": (np.uint8, "uint8_t"),
    "u16": (np.uint16, "uint16_t"),
    "u32": (np.uint32, "uint32_t"),
    "u64": (np.uint64, "uint64_t"),
    "bool": (np.bool_, "bool"),
}

# Types a kernel can return: spec name -> C type. Every element type, plus
# `usize` for counts and `void` for kernels that only write to a buffer.
RETURNS: dict[str, str] = {name: c for name, (_, c) in DTYPES.items()} | {
    "usize": "size_t",
    "void": "void",
}

EXACT = "exact"
USED_RETURN = "return"
USED_CAPACITY = "capacity"


def np_dtype(name: str) -> np.dtype:
    """Return the NumPy dtype for a spec type name such as `f32`."""
    return np.dtype(DTYPES[name][0])


# Scalar argument types: spec name -> C type. Integers travel in the integer
# argument registers and floats in the floating-point ones.
SCALARS: dict[str, str] = {"usize": "size_t", "i64": "int64_t", "f32": "float", "f64": "double"}
FLOAT_SCALARS = ("f32", "f64")

# The value of `len` and `capacity` that means "as many elements as the Size argument says".
SIZE_N = "n"


@dataclass(frozen=True)
class In:
    """A read-only input buffer."""

    # Argument name, as used in the C prototype and passed to the oracle.
    name: str
    # Element type: a key of DTYPES.
    dtype: str
    # Element count: SIZE_N for the kernel's Size argument, or a fixed number
    # for a buffer whose length does not depend on it (a set of test values,
    # for example).
    len: str | int = SIZE_N
    # Applied to every generated array before the call, for kernels that are
    # only defined on some inputs: `np.sort` for one that needs sorted data,
    # for example. Takes the array and returns one of the same length. Timing
    # inputs pass through it too, unless a distribution supplies them.
    prepare: Callable | None = None


@dataclass(frozen=True)
class Out:
    """An output buffer the kernel writes to."""

    # Argument name, as used in the C prototype.
    name: str
    # Element type: a key of DTYPES.
    dtype: str
    # How many elements the buffer can hold: SIZE_N for the kernel's Size
    # argument, or a fixed number (a table of counts, for example).
    capacity: str | int = SIZE_N
    # How much of the buffer the kernel fills: USED_CAPACITY means all of it,
    # USED_RETURN means the kernel returns the number of elements it wrote.
    used: str = USED_CAPACITY
    # True lets the kernel write anywhere inside the capacity, not only the
    # elements it reports. Only the reported elements are compared with the
    # oracle; the rest of the buffer may hold anything afterwards. Writing past
    # the capacity is still a failure. Needs used=USED_RETURN. A binding must
    # never show a caller the unreported part of such a buffer.
    scratch: bool = False


@dataclass(frozen=True)
class InOut:
    """A buffer the kernel reads and modifies in place.

    It starts as an input array and is compared, in full, with the oracle's
    result afterwards. The oracle receives the array like any other input and
    must return the modified copy without changing its argument.
    """

    # Argument name, as used in the C prototype and passed to the oracle.
    name: str
    # Element type: a key of DTYPES.
    dtype: str
    # An in-place buffer always has as many elements as the Size argument says,
    # is filled completely, and is never scratch. These mirror the fields of In
    # and Out so the judge can treat it as either.
    len = SIZE_N
    capacity = SIZE_N
    used = USED_CAPACITY
    scratch = False
    prepare = None


@dataclass(frozen=True)
class Size:
    """The element count, passed to the kernel as a `size_t`."""

    # Argument name, as used in the C prototype.
    name: str


@dataclass(frozen=True)
class Scalar:
    """A scalar argument that has the same value in every call.

    The judge has no value classes for scalars, so a kernel is checked and
    timed with this one value only. Whatever the kernel does for other values
    is not tested.
    """

    # Argument name, as used in the C prototype.
    name: str
    # Scalar type: a key of SCALARS.
    dtype: str
    # The value passed in every call.
    value: int | float


def element_count(arg: In | Out | InOut, n: int, capacity: bool = False) -> int:
    """Return how many elements a buffer argument has.

    Args:
        arg: A buffer argument.
        n: The value of the kernel's Size argument in this call.
        capacity: Read the buffer's `capacity` instead of its `len`. An In has
            only a length and an Out only a capacity; an InOut has both.

    Returns:
        The fixed element count if the spec gives one, otherwise `n`.
    """
    declared = arg.capacity if capacity or isinstance(arg, Out) else arg.len
    return n if declared == SIZE_N else declared


@dataclass
class Kernel:
    """Everything the judge needs to know about one kernel."""

    # Kernel name. Also the C function name and the directory name.
    name: str
    # Return type: a key of RETURNS.
    returns: str
    # Arguments in C order: a mix of In, Out and Size.
    args: list
    # Python function over the inputs (one NumPy array per In, in order) that
    # returns the correct answer.
    oracle: Callable
    # How output is compared with the oracle: EXACT for byte equality, or
    # {"tolerance_ulp": k} to allow k units in the last place on floats.
    mode: str | dict = EXACT
    # Smallest element count the kernel is ever called with.
    min_n: int = 0
    # Largest element count used for correctness, or None for the judge's full
    # range. For kernels whose NumPy oracle is too slow to run on millions of
    # elements once per case. Timing always uses the full sweep.
    check_max_n: int | None = None
    # Named input distributions used for timing: name -> fn(rng, n) -> {arg: array}.
    # When absent, timing uses uniformly random inputs.
    distributions: dict[str, Callable] | None = None
    # What is timed as NumPy, when that is not the oracle. Takes the same
    # arguments. Needed for in-place operations, where the oracle has to copy
    # its input and the statement a user would write does not.
    numpy: Callable | None = None

    def __post_init__(self) -> None:
        """Reject specs the judge cannot handle, with a message naming the kernel."""
        if self.returns not in RETURNS:
            raise ValueError(f"{self.name}: unknown return type {self.returns!r}")
        for a in self.args:
            if isinstance(a, (In, Out, InOut)) and a.dtype not in DTYPES:
                raise ValueError(f"{self.name}: unknown dtype {a.dtype!r} for {a.name}")
            if isinstance(a, Scalar) and a.dtype not in SCALARS:
                raise ValueError(f"{self.name}: unknown scalar type {a.dtype!r} for {a.name}")
        if len(self.outs) > 1:
            raise ValueError(f"{self.name}: at most one Out or InOut argument is supported")
        if self.out and self.out.used == USED_RETURN and self.returns != "usize":
            raise ValueError(f"{self.name}: used='return' needs returns='usize'")
        if self.out and self.out.scratch and self.out.used != USED_RETURN:
            raise ValueError(f"{self.name}: scratch=True needs used='return'; a buffer "
                             "that is filled completely has no unreported part")
        if self.mode != EXACT and self.tolerance_ulp is None:
            raise ValueError(f"{self.name}: mode must be 'exact' or {{'tolerance_ulp': k}}")

    @property
    def ins(self) -> list[In | InOut]:
        """The buffers the oracle receives, in argument order: inputs and in-place buffers."""
        return [a for a in self.args if isinstance(a, (In, InOut))]

    @property
    def outs(self) -> list[Out | InOut]:
        """The buffers the kernel writes, in argument order: outputs and in-place buffers."""
        return [a for a in self.args if isinstance(a, (Out, InOut))]

    @property
    def buffers(self) -> list[In | Out | InOut]:
        """Every pointer argument, in argument order."""
        return [a for a in self.args if isinstance(a, (In, Out, InOut))]

    @property
    def out(self) -> Out | InOut | None:
        """The single buffer the kernel writes, or None for kernels that only return a value."""
        outs = self.outs
        return outs[0] if outs else None

    @property
    def tolerance_ulp(self) -> int | None:
        """Allowed error in units in the last place, or None in exact mode."""
        if isinstance(self.mode, dict):
            return self.mode.get("tolerance_ulp")
        return None


def load_spec(kernel_dir: Path) -> Kernel:
    """Import `spec.py` from a kernel directory and return its `KERNEL`."""
    path = Path(kernel_dir) / "spec.py"
    if not path.exists():
        raise FileNotFoundError(f"no spec.py in {kernel_dir}")
    module_spec = importlib.util.spec_from_file_location(f"_hk_spec_{path.parent.name}", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    kernel = getattr(module, "KERNEL", None)
    if not isinstance(kernel, Kernel):
        raise ValueError(f"{path} must define KERNEL = Kernel(...)")
    return kernel
