"""Kernel spec format.

Provisional: this is the smallest format that covers the kernels we have.
Extend it when a kernel needs something it cannot express.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

# spec name -> (NumPy dtype, C type)
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

RETURNS: dict[str, str] = {name: c for name, (_, c) in DTYPES.items()} | {
    "usize": "size_t",
    "void": "void",
}


def np_dtype(name: str) -> np.dtype:
    return np.dtype(DTYPES[name][0])


@dataclass(frozen=True)
class In:
    """A read-only input buffer of `n` elements."""

    name: str
    dtype: str
    len: str = "n"


@dataclass(frozen=True)
class Out:
    """An output buffer with room for `n` elements.

    used="capacity": the kernel fills the whole buffer.
    used="return":   the kernel returns how many elements it wrote.
    """

    name: str
    dtype: str
    capacity: str = "n"
    used: str = "capacity"


@dataclass(frozen=True)
class Size:
    """The element count `n`."""

    name: str


@dataclass
class Kernel:
    name: str
    returns: str
    args: list
    oracle: Callable
    mode: str | dict = "exact"
    min_n: int = 0
    # Optional named input distributions for timing: name -> fn(rng, n) -> {arg: array}
    distributions: dict[str, Callable] | None = None

    def __post_init__(self) -> None:
        if self.returns not in RETURNS:
            raise ValueError(f"{self.name}: unknown return type {self.returns!r}")
        for a in self.args:
            if isinstance(a, (In, Out)) and a.dtype not in DTYPES:
                raise ValueError(f"{self.name}: unknown dtype {a.dtype!r} for {a.name}")
        if len(self.outs) > 1:
            raise ValueError(f"{self.name}: at most one Out argument is supported")
        if self.out and self.out.used == "return" and self.returns != "usize":
            raise ValueError(f"{self.name}: used='return' needs returns='usize'")
        if self.mode != "exact" and self.tolerance_ulp is None:
            raise ValueError(f"{self.name}: mode must be 'exact' or {{'tolerance_ulp': k}}")

    @property
    def ins(self) -> list[In]:
        return [a for a in self.args if isinstance(a, In)]

    @property
    def outs(self) -> list[Out]:
        return [a for a in self.args if isinstance(a, Out)]

    @property
    def out(self) -> Out | None:
        outs = self.outs
        return outs[0] if outs else None

    @property
    def tolerance_ulp(self) -> int | None:
        if isinstance(self.mode, dict):
            return self.mode.get("tolerance_ulp")
        return None


def load_spec(kernel_dir: Path) -> Kernel:
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
