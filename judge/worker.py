"""Runs correctness cases against one compiled kernel, in its own process.

The parent (`judge.check`) starts this module as a subprocess so that a kernel
which crashes or hangs cannot take the judge down with it. Two files connect
the processes:

- The progress file holds the description of the case about to run. If this
  process dies, the parent reads it to report which case was running.
- The result file receives the outcome as JSON when all cases finish, or when
  the first one fails.
"""
from __future__ import annotations

import ctypes
import itertools
import json
import os
import struct
from pathlib import Path

import numpy as np
import typer

from .cases import Case, build_inputs, fixed_cases, fresh_cases
from .memory import ArenaPool
from .spec import (FLOAT_SCALARS, USED_RETURN, In, InOut, Kernel, Scalar, Size, element_count,
                   load_spec, np_dtype)

# Failure categories reported to the parent.
WRONG_RESULT = "wrong result"
MEMORY_SAFETY = "memory safety"
CALLING_CONVENTION = "calling convention"

# Output buffers are filled with this byte before the call. Any byte beyond the
# elements the kernel reported using that no longer holds it was overwritten.
_FILL_BYTE = 0xA5

# The progress file is overwritten in place before every call. Padding each
# description to a fixed width means a shorter one fully replaces a longer one.
_PROGRESS_WIDTH = 400

# The ARM64 and x86-64 calling conventions both pass up to eight integer or
# pointer arguments, and up to eight floating-point arguments, in registers.
# The calling-convention wrapper takes one array of each.
_REGISTER_ARGS = 8
# The wrapper writes three values back: the integer return register, the
# floating-point return register (as raw bits), and a mask of which preserved
# registers the kernel changed.
_RESULT_SLOTS = 3
_RESULT_INT, _RESULT_FLOAT_BITS, _RESULT_VIOLATIONS = range(_RESULT_SLOTS)

RegisterArgs = ctypes.c_uint64 * _REGISTER_ARGS
WrapperResult = ctypes.c_uint64 * _RESULT_SLOTS

# Meaning of each bit in the violation mask, in bit order. Must match the
# layout documented in judge/c/abi_arm64.s.
_PRESERVED_REGISTERS = (
    [f"x{r}" for r in range(19, 29)]
    + [f"d{r}" for r in range(8, 16)]
    + ["the stack pointer", "the frame pointer (x29)"]
)

# Bit widths of the integer return types, for truncating the 64-bit register.
_INT_RETURN_BITS = {
    "i8": 8, "i16": 16, "i32": 32, "i64": 64,
    "u8": 8, "u16": 16, "u32": 32, "u64": 64,
    "usize": 64, "bool": 8,
}
_FLOAT_RETURNS = ("f32", "f64")
_F32_BITS_MASK = 0xFFFFFFFF
# An integer register holds 64 bits; negative values are passed in two's complement.
_REGISTER_MASK = 0xFFFFFFFFFFFFFFFF

app = typer.Typer(add_completion=False)


def _show(value) -> str:
    """Format a value for a failure message, unwrapping NumPy scalars."""
    return repr(value.item() if isinstance(value, np.generic) else value)


def _matches(got: np.ndarray, expected: np.ndarray, ulp: int | None,
             scale: np.ndarray | None = None) -> np.ndarray:
    """Return an elementwise mask of where `got` matches `expected`.

    In exact mode, and always for non-float data, elements match when their
    bytes are equal. That distinguishes +0.0 from -0.0 and treats two NaNs
    with the same bits as equal.

    In tolerance mode, finite floats match when they differ by at most `ulp`
    units in the last place. NaN and infinity must still appear in exactly the
    same positions.

    Args:
        got: What the kernel produced.
        expected: The oracle's answer, same shape and dtype as `got`.
        ulp: Allowed error in units in the last place, or None for exact mode.
        scale: Per element, the magnitude at which the unit in the last place
            is taken, or None to take it at the larger of the two values
            compared. Only used in tolerance mode.

    Returns:
        A boolean array, True where the element is accepted.
    """
    if ulp is None or got.dtype.kind != "f":
        unsigned = f"u{got.dtype.itemsize}"
        return np.ascontiguousarray(got).view(unsigned) == np.ascontiguousarray(expected).view(unsigned)
    special = ~np.isfinite(expected) | ~np.isfinite(got)
    identical = (got == expected) | (np.isnan(got) & np.isnan(expected))
    with np.errstate(invalid="ignore", over="ignore"):
        if scale is None:
            magnitude = np.maximum(np.abs(got), np.abs(expected))
        else:
            # A scale beyond the largest finite value would have no unit in the
            # last place, so it is capped there. A NaN scale accepts nothing.
            largest = np.finfo(got.dtype).max
            magnitude = np.minimum(np.abs(np.asarray(scale, dtype=np.float64)), largest).astype(got.dtype)
        close = np.abs(got - expected) <= ulp * np.spacing(magnitude)
    return np.where(special, identical, close)


def _float_bits(scalar: Scalar) -> int:
    """Return the register contents for a floating-point scalar argument.

    Args:
        scalar: A Scalar of type f32 or f64.

    Returns:
        The raw bits as an integer. A float occupies the low 32 bits of its
        64-bit register, which is where the kernel reads it as `s0`..`s7`.
    """
    if scalar.dtype == "f32":
        return struct.unpack("<I", struct.pack("<f", scalar.value))[0]
    return struct.unpack("<Q", struct.pack("<d", scalar.value))[0]


class Runner:
    """Runs cases against one kernel and reports the first failure."""

    def __init__(self, k: Kernel, fn: int, harness: ctypes.CDLL, progress_fd: int) -> None:
        """Prepare to call the kernel at address `fn` through the harness wrapper."""
        self.k = k
        self.fn = fn
        self.progress_fd = progress_fd
        self.arenas = ArenaPool()
        # Base random arrays for the element count currently being tested.
        self.cache: dict = {}
        self.cache_n: int | None = None
        self.wrapper = harness.hk_abi_call
        self.wrapper.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(RegisterArgs),
            ctypes.POINTER(RegisterArgs),
            ctypes.POINTER(WrapperResult),
        ]
        self.wrapper.restype = None

    def run(self, case: Case) -> dict | None:
        """Run one case. Returns None if the kernel behaved, or a failure description."""
        k = self.k
        if case.n != self.cache_n:
            self.cache, self.cache_n = {}, case.n
        inputs = build_inputs(k, case, self.cache)
        oracle_args = [inputs[a.name] for a in k.ins]
        expected = k.oracle(*oracle_args)
        scale = k.tolerance_scale(*oracle_args) if k.tolerance_scale else None

        int_args, float_args, out_addr, out_bytes, read_only = self._place_arguments(case, inputs)
        result = WrapperResult()

        # Record the case before calling, in case the kernel kills this process.
        os.pwrite(self.progress_fd, str(case).ljust(_PROGRESS_WIDTH).encode(), 0)
        self.wrapper(self.fn, ctypes.byref(int_args), ctypes.byref(float_args), ctypes.byref(result))
        for arena in read_only:
            arena.protect(read_only=False)

        violations = result[_RESULT_VIOLATIONS]
        if violations:
            changed = [name for bit, name in enumerate(_PRESERVED_REGISTERS) if violations >> bit & 1]
            return self._failure(CALLING_CONVENTION, f"did not preserve {', '.join(changed)}", case)

        returned = self._decode_return(result)
        if k.out is None:
            return self._check_return_value(case, returned, expected, scale)
        return self._check_output(case, returned, expected, scale, out_addr, out_bytes)

    def _place_arguments(self, case: Case, inputs: dict) -> tuple:
        """Copy inputs into guarded memory and build the kernel's argument list.

        Returns (integer argument registers, floating-point argument registers,
        output address, output size in bytes, arenas made read-only). Inputs
        are made read-only so that a kernel writing to one crashes. An output
        buffer is filled with a known byte so that stray writes can be detected
        afterwards; an in-place buffer starts as its input and stays writable.

        The calling convention numbers integer and floating-point arguments
        separately: the third integer argument goes in the third integer
        register however many floating-point arguments precede it.
        """
        int_args, float_args = RegisterArgs(), RegisterArgs()
        int_slot = float_slot = 0
        read_only = []
        out_addr = out_bytes = 0
        for a in self.k.args:
            if isinstance(a, Scalar) and a.dtype in FLOAT_SCALARS:
                float_args[float_slot] = _float_bits(a)
                float_slot += 1
                continue
            if isinstance(a, (Size, Scalar)):
                int_args[int_slot] = (case.n if isinstance(a, Size) else int(a.value)) & _REGISTER_MASK
                int_slot += 1
                continue
            itemsize = np_dtype(a.dtype).itemsize
            nbytes = element_count(a, case.n) * itemsize
            arena = self.arenas.get(a.name, nbytes)
            addr = arena.place(nbytes, case.arr, itemsize)
            int_args[int_slot] = addr
            int_slot += 1
            if isinstance(a, (In, InOut)) and nbytes:
                ctypes.memmove(addr, inputs[a.name].ctypes.data, nbytes)
            if isinstance(a, In):
                arena.protect(read_only=True)
                read_only.append(arena)
                continue
            out_addr, out_bytes = addr, nbytes
            if not isinstance(a, InOut):
                ctypes.memset(addr, _FILL_BYTE, nbytes)
        return int_args, float_args, out_addr, out_bytes, read_only

    def _decode_return(self, result) -> object:
        """Convert the raw return registers into a Python value of the kernel's return type."""
        returns = self.k.returns
        if returns in _INT_RETURN_BITS:
            bits = _INT_RETURN_BITS[returns]
            value = result[_RESULT_INT] & ((1 << bits) - 1)
            is_signed = returns.startswith("i")
            if is_signed and value >> (bits - 1):
                value -= 1 << bits  # reinterpret the top bit as the sign
            return bool(value) if returns == "bool" else value
        if returns == "f32":
            raw = struct.pack("<I", result[_RESULT_FLOAT_BITS] & _F32_BITS_MASK)
            return np.float32(struct.unpack("<f", raw)[0])
        if returns == "f64":
            raw = struct.pack("<Q", result[_RESULT_FLOAT_BITS])
            return np.float64(struct.unpack("<d", raw)[0])
        return None  # void

    def _check_return_value(self, case: Case, returned, expected, scale) -> dict | None:
        """Compare the return value of a kernel that has no output buffer.

        Args:
            case: The case that was run, for the failure message.
            returned: The decoded return value.
            expected: The oracle's answer.
            scale: The spec's tolerance scale for this case, or None.

        Returns:
            None when the value is accepted, otherwise a failure description.
        """
        k = self.k
        if k.returns in _FLOAT_RETURNS:
            want = np.asarray(expected, dtype=returned.dtype)
            ok = bool(_matches(np.asarray(returned), want, k.tolerance_ulp, scale).all())
        else:
            want = bool(expected) if k.returns == "bool" else int(expected)
            ok = returned == want
        if ok:
            return None
        return self._failure(WRONG_RESULT, "wrong return value", case,
                             expected=_show(want), got=_show(returned))

    def _check_output(self, case: Case, returned, expected, scale,
                      out_addr: int, out_bytes: int) -> dict | None:
        """Compare what the kernel wrote to its output buffer with the oracle.

        Checks, in order: the reported element count, the elements themselves,
        and that nothing beyond the reported count was touched. The last check
        is skipped for a scratch buffer, where the kernel may leave anything
        between the reported elements and the capacity. A write past the
        capacity never reaches this function: it faults on the guard page.

        Args:
            case: The case that was run, for the failure message.
            returned: The decoded return value, which is the element count
                when the output reports how much of it was used.
            expected: The oracle's answer.
            scale: The spec's tolerance scale for this case, one magnitude per
                expected element, or None.
            out_addr: Address of the output buffer.
            out_bytes: Size of the output buffer in bytes.

        Returns:
            None when the output is accepted, otherwise a failure description.
        """
        k, out = self.k, self.k.out
        dt = np_dtype(out.dtype)
        want = np.ascontiguousarray(np.asarray(expected), dtype=dt).ravel()
        capacity = element_count(out, case.n, capacity=True)
        used = capacity
        if out.used == USED_RETURN:
            if returned != len(want):
                return self._failure(WRONG_RESULT, "wrong number of elements written", case,
                                     expected=str(len(want)), got=str(returned))
            used = returned
        elif len(want) != capacity:
            raise ValueError(f"oracle returned {len(want)} elements, expected {capacity}")

        raw = ctypes.string_at(out_addr, out_bytes) if out_bytes else b""
        got = np.frombuffer(raw, dtype=dt)
        if scale is not None:
            scale = np.asarray(scale).ravel()
        matches = _matches(got[:used], want, k.tolerance_ulp, scale)
        if not matches.all():
            i = int(np.argmin(matches))  # first mismatch
            return self._failure(WRONG_RESULT, f"wrong value in {out.name}[{i}]", case,
                                 expected=_show(want[i]), got=_show(got[i]))
        if out.scratch:
            return None
        untouched = raw[used * dt.itemsize:]
        if untouched.count(_FILL_BYTE) != len(untouched):
            return self._failure(MEMORY_SAFETY,
                                 f"wrote past the {used} used elements of {out.name}", case)
        return None

    @staticmethod
    def _failure(category: str, message: str, case: Case, **extra) -> dict:
        """Build the failure description sent back to the parent."""
        return {"category": category, "message": message, "case": str(case), **extra}


@app.command()
def run(
    kernel_dir: Path = typer.Argument(..., help="Directory holding the kernel's spec.py."),
    dylib: Path = typer.Argument(..., help="Compiled library containing the kernel."),
    harness: Path = typer.Argument(..., help="Compiled harness library for this kernel."),
    progress: Path = typer.Argument(..., help="File updated with the case about to run."),
    result: Path = typer.Argument(..., help="File that receives the outcome as JSON."),
    fresh_seed: int = typer.Argument(..., help="Seed for the fresh random cases."),
    case: str = typer.Option("", help="Run only this case, in the judge's printed form."),
) -> None:
    """Run every correctness case against one kernel and write the outcome to a file."""
    k = load_spec(kernel_dir)
    lib = ctypes.CDLL(str(dylib))
    fn = ctypes.cast(getattr(lib, k.name), ctypes.c_void_p).value
    progress_fd = os.open(progress, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    runner = Runner(k, fn, ctypes.CDLL(str(harness)), progress_fd)

    cases = [Case.parse(case)] if case else itertools.chain(fixed_cases(k), fresh_cases(k, fresh_seed))
    outcome = {"ok": True}
    count = 0
    for count, current in enumerate(cases, start=1):
        failure = runner.run(current)
        if failure:
            outcome = {"ok": False, **failure}
            break
    result.write_text(json.dumps({**outcome, "cases": count}))


if __name__ == "__main__":
    app()
