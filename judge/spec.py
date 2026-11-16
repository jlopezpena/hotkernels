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


@dataclass(frozen=True)
class In:
    """A read-only input buffer."""

    # Argument name, as used in the C prototype and passed to the oracle.
    name: str
    # Element type: a key of DTYPES.
    dtype: str
    # Name of the Size argument that gives the element count.
    len: str = "n"


@dataclass(frozen=True)
class Out:
    """An output buffer the kernel writes to."""

    # Argument name, as used in the C prototype.
    name: str
    # Element type: a key of DTYPES.
    dtype: str
    # Name of the Size argument that gives how many elements the buffer can hold.
    capacity: str = "n"
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
class Size:
    """The element count, passed to the kernel as a `size_t`."""

    # Argument name, as used in the C prototype.
    name: str


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
    # Named input distributions used for timing: name -> fn(rng, n) -> {arg: array}.
    # When absent, timing uses uniformly random inputs.
    distributions: dict[str, Callable] | None = None

    def __post_init__(self) -> None:
        """Reject specs the judge cannot handle, with a message naming the kernel."""
        if self.returns not in RETURNS:
            raise ValueError(f"{self.name}: unknown return type {self.returns!r}")
        for a in self.args:
            if isinstance(a, (In, Out)) and a.dtype not in DTYPES:
                raise ValueError(f"{self.name}: unknown dtype {a.dtype!r} for {a.name}")
        if len(self.outs) > 1:
            raise ValueError(f"{self.name}: at most one Out argument is supported")
        if self.out and self.out.used == USED_RETURN and self.returns != "usize":
            raise ValueError(f"{self.name}: used='return' needs returns='usize'")
        if self.out and self.out.scratch and self.out.used != USED_RETURN:
            raise ValueError(f"{self.name}: scratch=True needs used='return'; a buffer "
                             "that is filled completely has no unreported part")
        if self.mode != EXACT and self.tolerance_ulp is None:
            raise ValueError(f"{self.name}: mode must be 'exact' or {{'tolerance_ulp': k}}")

    @property
    def ins(self) -> list[In]:
        """The input buffers, in argument order."""
        return [a for a in self.args if isinstance(a, In)]

    @property
    def outs(self) -> list[Out]:
        """The output buffers, in argument order."""
        return [a for a in self.args if isinstance(a, Out)]

    @property
    def out(self) -> Out | None:
        """The single output buffer, or None for kernels that only return a value."""
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
