"""Runs correctness cases against one compiled kernel.

Runs in its own process so that a kernel that crashes or hangs cannot take the
judge down. Before each call the case description is written to a progress
file; if the process dies, the parent reads it to report which case was running.
"""
from __future__ import annotations

import ctypes
import itertools
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

from .cases import Case, build_inputs, fixed_cases, fresh_cases
from .memory import ArenaPool
from .spec import In, Kernel, Out, Size, load_spec, np_dtype

_PATTERN = 0xA5
_INT_BITS = {"i8": 8, "i16": 16, "i32": 32, "i64": 64, "u8": 8, "u16": 16, "u32": 32,
             "u64": 64, "usize": 64, "bool": 8}
_ABI_NAMES = (
    [f"x{r}" for r in range(19, 29)] + [f"d{r}" for r in range(8, 16)]
    + ["the stack pointer", "the frame pointer (x29)"]
)

U64x8 = ctypes.c_uint64 * 8
U64x3 = ctypes.c_uint64 * 3


def _fmt(value) -> str:
    return repr(value.item() if isinstance(value, np.generic) else value)


def _same(got: np.ndarray, exp: np.ndarray, ulp: int | None) -> np.ndarray:
    """Elementwise 'matches' mask under the kernel's correctness mode."""
    if ulp is None or got.dtype.kind != "f":
        width = got.dtype.itemsize
        raw = lambda a: np.ascontiguousarray(a).view(f"u{width}")  # noqa: E731
        return raw(got) == raw(exp)
    special = ~np.isfinite(exp) | ~np.isfinite(got)
    exact = (got == exp) | (np.isnan(got) & np.isnan(exp))
    with np.errstate(invalid="ignore", over="ignore"):
        close = np.abs(got - exp) <= ulp * np.spacing(np.maximum(np.abs(got), np.abs(exp)))
    return np.where(special, exact, close)


class Runner:
    def __init__(self, k: Kernel, fn: int, harness: ctypes.CDLL, progress_fd: int) -> None:
        self.k = k
        self.fn = fn
        self.call = harness.hk_abi_call
        self.call.argtypes = [ctypes.c_void_p, ctypes.POINTER(U64x8), ctypes.POINTER(U64x8),
                              ctypes.POINTER(U64x3)]
        self.call.restype = None
        self.progress_fd = progress_fd
        self.arenas = ArenaPool()
        self.cache: dict = {}
        self.cache_n = None

    def _fail(self, category: str, message: str, case: Case, **extra) -> dict:
        return {"category": category, "message": message, "case": str(case), **extra}

    def run(self, case: Case) -> dict | None:
        k = self.k
        if case.n != self.cache_n:
            self.cache, self.cache_n = {}, case.n
        inputs = build_inputs(k, case, self.cache)
        expected = k.oracle(*[inputs[a.name] for a in k.ins])

        iargs, fargs, result = U64x8(), U64x8(), U64x3()
        protected = []
        out_addr = out_bytes = 0
        for slot, a in enumerate(k.args):
            if isinstance(a, Size):
                iargs[slot] = case.n
                continue
            item = np_dtype(a.dtype).itemsize
            nbytes = case.n * item
            arena = self.arenas.get(a.name, nbytes)
            addr = arena.place(nbytes, case.arr, item)
            iargs[slot] = addr
            if isinstance(a, In):
                if nbytes:
                    ctypes.memmove(addr, inputs[a.name].ctypes.data, nbytes)
                arena.protect(read_only=True)
                protected.append(arena)
            else:
                out_addr, out_bytes = addr, nbytes
                ctypes.memset(addr, _PATTERN, nbytes)

        os.pwrite(self.progress_fd, str(case).ljust(400).encode(), 0)
        self.call(self.fn, ctypes.byref(iargs), ctypes.byref(fargs), ctypes.byref(result))
        for arena in protected:
            arena.protect(read_only=False)

        if result[2]:
            changed = [name for bit, name in enumerate(_ABI_NAMES) if result[2] >> bit & 1]
            return self._fail("calling convention",
                              f"did not preserve {', '.join(changed)}", case)

        # Return value
        ret = None
        if k.returns in _INT_BITS:
            bits = _INT_BITS[k.returns]
            ret = result[0] & ((1 << bits) - 1)
            if k.returns.startswith("i") and ret >> (bits - 1):
                ret -= 1 << bits
            if k.returns == "bool":
                ret = bool(ret)
        elif k.returns == "f32":
            ret = np.float32(struct.unpack("<f", struct.pack("<I", result[1] & 0xFFFFFFFF))[0])
        elif k.returns == "f64":
            ret = np.float64(struct.unpack("<d", struct.pack("<Q", result[1]))[0])

        out = k.out
        if out is None:
            if k.returns in ("f32", "f64"):
                exp = np.asarray(expected, dtype=ret.dtype)
                ok = bool(_same(np.asarray(ret), exp, k.tolerance_ulp).all())
            else:
                exp = bool(expected) if k.returns == "bool" else int(expected)
                ok = ret == exp
            if not ok:
                return self._fail("wrong result", "wrong return value", case,
                                  expected=_fmt(exp), got=_fmt(ret))
            return None

        dt = np_dtype(out.dtype)
        exp = np.ascontiguousarray(np.asarray(expected), dtype=dt).ravel()
        capacity = case.n
        used = capacity
        if out.used == "return":
            if ret != len(exp):
                return self._fail("wrong result", "wrong number of elements written", case,
                                  expected=str(len(exp)), got=str(ret))
            used = ret
        elif len(exp) != capacity:
            raise ValueError(f"oracle returned {len(exp)} elements, expected {capacity}")

        raw = ctypes.string_at(out_addr, out_bytes) if out_bytes else b""
        got = np.frombuffer(raw, dtype=dt)
        same = _same(got[:used], exp, k.tolerance_ulp)
        if not same.all():
            i = int(np.argmin(same))
            return self._fail("wrong result", f"wrong value in {out.name}[{i}]", case,
                              expected=_fmt(exp[i]), got=_fmt(got[i]))
        tail = raw[used * dt.itemsize:]
        if tail.count(_PATTERN) != len(tail):
            return self._fail("memory safety",
                              f"wrote past the {used} used elements of {out.name}", case)
        return None


def main(argv: list[str]) -> int:
    kernel_dir, dylib, harness_path, progress, fresh_seed, only = argv
    k = load_spec(Path(kernel_dir))
    lib = ctypes.CDLL(dylib)
    fn = ctypes.cast(getattr(lib, k.name), ctypes.c_void_p).value
    harness = ctypes.CDLL(harness_path)
    fd = os.open(progress, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    runner = Runner(k, fn, harness, fd)

    if only:
        cases = [Case.parse(only)]
    else:
        cases = itertools.chain(fixed_cases(k), fresh_cases(k, int(fresh_seed)))
    count = 0
    for case in cases:
        count += 1
        failure = runner.run(case)
        if failure:
            print(json.dumps({"ok": False, "cases": count, **failure}))
            return 0
    print(json.dumps({"ok": True, "cases": count}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
