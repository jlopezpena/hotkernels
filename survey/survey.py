# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.5"]
# ///
"""Phase 2 survey: how far is NumPy behind naive `clang -O3` C, per operation?

    uv run survey/survey.py                 full sweep, writes survey/results/
    uv run survey/survey.py --only argmax   candidates whose key matches a regex
    uv run survey/survey.py --quick         fewer samples and no 16M size

This is a standalone harness. It does not use the phase 1 judge, which does not
exist yet. It borrows the judge's timing method where that is cheap: C is timed
inside C, samples are interleaved, the median is reported, the thread asks for
the performance cores, and a reference loop is timed before and after to detect
a noisy machine.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import platform
import re
import statistics
import subprocess
import sys
import time
import timeit
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

HERE = Path(__file__).parent
BUILD = HERE / "build"
RESULTS = HERE / "results"
CFLAGS = ["-O3", "-mcpu=native", "-std=c11", "-Wall", "-Wextra"]
SIZES = [1, 16, 256, 4096, 65536, 1048576, 16777216]
MIN_SAMPLE_S = 0.005
CTYPES = {"p": ctypes.c_void_p, "i": ctypes.c_int64, "f": ctypes.c_float, "d": ctypes.c_double}


def ptr(a: np.ndarray) -> int:
    return a.ctypes.data


@dataclass
class Cand:
    key: str                 # unique name, used in tables
    group: str               # which phase 2 candidate this belongs to
    stmt: str                # the NumPy statement that is timed
    cfn: str                 # naive C kernel in naive.c
    sig: str                 # C argument types: p pointer, i int64, f float, d double
    setup: Callable          # (n, rng) -> env dict for stmt and the lambdas below
    cargs: Callable          # env -> C arguments (without fresh/reps)
    verify: Callable         # (env, ret) -> bool: does naive C agree with NumPy?
    out: str | None = None   # env name of the output buffer, if the kernel writes one
    alloc: bool = False      # the NumPy statement allocates its result
    work: Callable = field(default=lambda env: env["n"])  # elements processed per call


def rng_f(dtype):
    return lambda n, rng: rng.random(n, dtype=dtype)


CANDS: list[Cand] = []


def add(*a, **k):
    CANDS.append(Cand(*a, **k))


# ------------------------------------------------------------------ argmax / argmin
for name, dt, gen in [
    ("argmax_f32", np.float32, rng_f(np.float32)),
    ("argmax_f64", np.float64, rng_f(np.float64)),
    ("argmax_i64", np.int64, lambda n, rng: rng.integers(-(1 << 40), 1 << 40, n)),
]:
    add(name, "argmax/argmin", "np.argmax(x)", name, "pii",
        lambda n, rng, gen=gen: {"x": gen(n, rng)},
        lambda e: (ptr(e["x"]), e["n"], 1),
        lambda e, r: r == np.argmax(e["x"]))
add("argmin_f64", "argmax/argmin", "np.argmin(x)", "argmin_f64", "pii",
    lambda n, rng: {"x": rng.random(n)},
    lambda e: (ptr(e["x"]), e["n"], 1),
    lambda e, r: r == np.argmin(e["x"]))

# ------------------------------------------------------------------ controls
add("max_f32 (control)", "control", "np.max(x)", "max_f32", "pip",
    lambda n, rng: {"x": rng.random(n, dtype=np.float32), "res": np.empty(1, np.float32)},
    lambda e: (ptr(e["x"]), e["n"], ptr(e["res"])),
    lambda e, r: e["res"][0] == np.max(e["x"]))
add("sum_f32 (control)", "control", "np.sum(x)", "sum_f32", "piip",
    lambda n, rng: {"x": rng.random(n, dtype=np.float32), "res": np.empty(1, np.float32)},
    lambda e: (ptr(e["x"]), e["n"], 1, ptr(e["res"])),
    lambda e, r: np.isclose(e["res"][0], np.sum(e["x"]), rtol=1e-2))
add("sum_f64 (control)", "control", "np.sum(x)", "sum_f64", "piip",
    lambda n, rng: {"x": rng.random(n), "res": np.empty(1)},
    lambda e: (ptr(e["x"]), e["n"], 1, ptr(e["res"])),
    lambda e, r: np.isclose(e["res"][0], np.sum(e["x"]), rtol=1e-6))

# ------------------------------------------------------------------ boolean mask selection
for dt, tag in [(np.float32, "f32"), (np.float64, "f64")]:
    for p in (0.5, 0.05):
        add(f"compress_{tag} p={p}", "a[mask]", "x[mask]", f"compress_{tag}", "pppi",
            lambda n, rng, dt=dt, p=p: {"x": rng.random(n).astype(dt), "mask": rng.random(n) < p,
                                        "out": np.empty(n, dt)},
            lambda e: (ptr(e["x"]), ptr(e["mask"]), ptr(e["out"]), e["n"]),
            lambda e, r: np.array_equal(e["out"][:r], e["x"][e["mask"]]),
            out="out", alloc=True)

# ------------------------------------------------------------------ count_nonzero
add("count_nonzero_bool", "count_nonzero", "np.count_nonzero(x)", "count_nonzero_bool", "pi",
    lambda n, rng: {"x": rng.random(n) < 0.5},
    lambda e: (ptr(e["x"]), e["n"]),
    lambda e, r: r == np.count_nonzero(e["x"]))
add("count_nonzero_i64", "count_nonzero", "np.count_nonzero(x)", "count_nonzero_i64", "pi",
    lambda n, rng: {"x": rng.integers(0, 3, n)},
    lambda e: (ptr(e["x"]), e["n"]),
    lambda e, r: r == np.count_nonzero(e["x"]))
add("count_nonzero_f64", "count_nonzero", "np.count_nonzero(x)", "count_nonzero_f64", "pi",
    lambda n, rng: {"x": rng.random(n) * (rng.random(n) < 0.5)},
    lambda e: (ptr(e["x"]), e["n"]),
    lambda e, r: r == np.count_nonzero(e["x"]))

# ------------------------------------------------------------------ nonzero
for p in (0.5, 0.05):
    add(f"nonzero_bool p={p}", "nonzero", "np.nonzero(mask)[0]", "nonzero_bool", "ppi",
        lambda n, rng, p=p: {"mask": rng.random(n) < p, "out": np.empty(n, np.int64)},
        lambda e: (ptr(e["mask"]), ptr(e["out"]), e["n"]),
        lambda e, r: np.array_equal(e["out"][:r], np.nonzero(e["mask"])[0]),
        out="out", alloc=True)

# ------------------------------------------------------------------ cumsum and friends
for name, gen, dt in [
    ("cumsum_f32", rng_f(np.float32), np.float32),
    ("cumsum_f64", rng_f(np.float64), np.float64),
    ("cumsum_i64", lambda n, rng: rng.integers(-1000, 1000, n), np.int64),
]:
    setup = lambda n, rng, gen=gen, dt=dt: {"x": gen(n, rng), "out": np.empty(n, dt)}
    cargs = lambda e: (ptr(e["x"]), ptr(e["out"]), e["n"])
    verify = lambda e, r: np.array_equal(e["out"], np.cumsum(e["x"]))
    add(name, "cumsum", "np.cumsum(x)", name, "ppi", setup, cargs, verify, out="out", alloc=True)
    add(name + " out=", "cumsum", "np.cumsum(x, out=out)", name, "ppi", setup, cargs, verify, out="out")
add("cummin_f64", "extra", "np.minimum.accumulate(x)", "cummin_f64", "ppi",
    lambda n, rng: {"x": rng.random(n), "out": np.empty(n)},
    lambda e: (ptr(e["x"]), ptr(e["out"]), e["n"]),
    lambda e, r: np.array_equal(e["out"], np.minimum.accumulate(e["x"])),
    out="out", alloc=True)
add("diff_f64 (control)", "control", "np.diff(x)", "diff_f64", "ppi",
    lambda n, rng: {"x": rng.random(n), "out": np.empty(n)},
    lambda e: (ptr(e["x"]), ptr(e["out"]), e["n"]),
    lambda e, r: np.array_equal(e["out"][: e["n"] - 1], np.diff(e["x"])),
    out="out", alloc=True)

# ------------------------------------------------------------------ searchsorted
# Haystack of n sorted values, min(n, 2**20) random needles. Reported per needle.
for name, gen in [
    ("searchsorted_f64", lambda n, rng: rng.random(n)),
    ("searchsorted_i64", lambda n, rng: rng.integers(0, 1 << 40, n)),
]:
    def setup(n, rng, gen=gen):
        m = min(n, 1 << 20)
        return {"a": np.sort(gen(n, rng)), "v": gen(m, rng), "m": m, "out": np.empty(m, np.int64)}
    add(name, "searchsorted", "np.searchsorted(a, v)", name, "pipip", setup,
        lambda e: (ptr(e["a"]), e["n"], ptr(e["v"]), e["m"], ptr(e["out"])),
        lambda e, r: np.array_equal(e["out"], np.searchsorted(e["a"], e["v"])),
        out="out", alloc=True, work=lambda e: e["m"])

# ------------------------------------------------------------------ clip
for name, gen, dt, lo, hi, t in [
    ("clip_f32", rng_f(np.float32), np.float32, 0.25, 0.75, "f"),
    ("clip_f64", rng_f(np.float64), np.float64, 0.25, 0.75, "d"),
    ("clip_i64", lambda n, rng: rng.integers(0, 1000, n), np.int64, 250, 750, "i"),
]:
    setup = lambda n, rng, gen=gen, dt=dt, lo=lo, hi=hi: {
        "x": gen(n, rng), "out": np.empty(n, dt), "lo": lo, "hi": hi}
    cargs = lambda e: (ptr(e["x"]), ptr(e["out"]), e["n"], e["lo"], e["hi"])
    verify = lambda e, r: np.array_equal(e["out"], np.clip(e["x"], e["lo"], e["hi"]))
    add(name, "clip", "np.clip(x, lo, hi)", name, "ppi" + t + t, setup, cargs, verify, out="out", alloc=True)
    add(name + " out=", "clip", "np.clip(x, lo, hi, out=out)", name, "ppi" + t + t, setup, cargs, verify, out="out")

# ------------------------------------------------------------------ any / all (no early exit possible)
add("any_bool (all False)", "any/all", "np.any(x)", "any_bool", "pi",
    lambda n, rng: {"x": np.zeros(n, bool)},
    lambda e: (ptr(e["x"]), e["n"]),
    lambda e, r: bool(r) == bool(np.any(e["x"])))
add("all_bool (all True)", "any/all", "np.all(x)", "all_bool", "pi",
    lambda n, rng: {"x": np.ones(n, bool)},
    lambda e: (ptr(e["x"]), e["n"]),
    lambda e, r: bool(r) == bool(np.all(e["x"])))

# ------------------------------------------------------------------ take (random indices)
take_setup = lambda n, rng: {"a": rng.random(n), "idx": rng.integers(0, n, n), "out": np.empty(n)}
take_cargs = lambda e: (ptr(e["a"]), e["n"], ptr(e["idx"]), e["n"], ptr(e["out"]))
take_verify = lambda e, r: r == 0 and np.array_equal(e["out"], e["a"][e["idx"]])
add("take_f64 a[idx]", "take", "a[idx]", "take_f64", "pipip", take_setup, take_cargs, take_verify,
    out="out", alloc=True)
add("take_f64 np.take", "take", "np.take(a, idx)", "take_f64", "pipip", take_setup, take_cargs, take_verify,
    out="out", alloc=True)
add("take_f64 np.take out=", "take", "np.take(a, idx, out=out)", "take_f64", "pipip", take_setup, take_cargs,
    take_verify, out="out")

# ------------------------------------------------------------------ isin (8 test values)
add("isin_i64 m=8", "isin", "np.isin(x, test)", "isin_i64", "pipip",
    lambda n, rng: {"x": rng.integers(0, 1000, n), "test": rng.integers(0, 1000, 8), "out": np.empty(n, bool)},
    lambda e: (ptr(e["x"]), e["n"], ptr(e["test"]), 8, ptr(e["out"])),
    lambda e, r: np.array_equal(e["out"], np.isin(e["x"], e["test"])),
    out="out", alloc=True)

# ------------------------------------------------------------------ extras not on the phase 2 list
add("where_f64", "extra", "np.where(c, a, b)", "where_f64", "ppppi",
    lambda n, rng: {"c": rng.random(n) < 0.5, "a": rng.random(n), "b": rng.random(n), "out": np.empty(n)},
    lambda e: (ptr(e["c"]), ptr(e["a"]), ptr(e["b"]), ptr(e["out"]), e["n"]),
    lambda e, r: np.array_equal(e["out"], np.where(e["c"], e["a"], e["b"])),
    out="out", alloc=True)
add("bincount_i64 256 bins", "extra", "np.bincount(x, minlength=256)", "bincount_i64", "pipi",
    lambda n, rng: {"x": rng.integers(0, 256, n), "out": np.empty(256, np.int64)},
    lambda e: (ptr(e["x"]), e["n"], ptr(e["out"]), 256),
    lambda e, r: np.array_equal(e["out"], np.bincount(e["x"], minlength=256)),
    out="out", alloc=True)

# ------------------------------------------------------------------ stepped slices (xs[::2], n elements used)
add("argmax_f64 [::2]", "stepped slices", "np.argmax(xs[::2])", "argmax_f64", "pii",
    lambda n, rng: {"xs": rng.random(2 * n)},
    lambda e: (ptr(e["xs"]), e["n"], 2),
    lambda e, r: r == np.argmax(e["xs"][::2]))
add("sum_f64 [::2]", "stepped slices", "np.sum(xs[::2])", "sum_f64", "piip",
    lambda n, rng: {"xs": rng.random(2 * n), "res": np.empty(1)},
    lambda e: (ptr(e["xs"]), e["n"], 2, ptr(e["res"])),
    lambda e, r: np.isclose(e["res"][0], np.sum(e["xs"][::2]), rtol=1e-6))
add("copy_f64 [::2] out=", "stepped slices", "np.copyto(out, xs[::2])", "copy_f64_strided", "ppii",
    lambda n, rng: {"xs": rng.random(2 * n), "out": np.empty(n)},
    lambda e: (ptr(e["xs"]), ptr(e["out"]), e["n"], 2),
    lambda e, r: np.array_equal(e["out"], e["xs"][::2]),
    out="out")
strided_add_setup = lambda n, rng: {"xs": rng.random(2 * n), "ys": rng.random(2 * n), "out": np.empty(n)}
strided_add_cargs = lambda e: (ptr(e["xs"]), ptr(e["ys"]), ptr(e["out"]), e["n"], 2)
strided_add_verify = lambda e, r: np.array_equal(e["out"], e["xs"][::2] + e["ys"][::2])
add("add_f64 [::2]", "stepped slices", "xs[::2] + ys[::2]", "add_f64_strided", "pppii",
    strided_add_setup, strided_add_cargs, strided_add_verify, out="out", alloc=True)
add("add_f64 [::2] out=", "stepped slices", "np.add(xs[::2], ys[::2], out=out)", "add_f64_strided", "pppii",
    strided_add_setup, strided_add_cargs, strided_add_verify, out="out")


# ------------------------------------------------------------------ wider net (to reach about ten candidates)
def last_true(n, rng):
    x = np.zeros(n, bool)
    x[-1] = True
    return {"x": x}
add("argmax_bool (find first)", "wider", "np.argmax(x)", "argmax_u8", "pii", last_true,
    lambda e: (ptr(e["x"]), e["n"], 1),
    lambda e, r: r == np.argmax(e["x"]))
add("sum_bool", "wider", "np.sum(x)", "count_nonzero_bool", "pi",
    lambda n, rng: {"x": rng.random(n) < 0.5},
    lambda e: (ptr(e["x"]), e["n"]),
    lambda e, r: r == np.sum(e["x"]))
add("sum_u8", "wider", "np.sum(x)", "sum_u8", "pi",
    lambda n, rng: {"x": rng.integers(0, 256, n, dtype=np.uint8)},
    lambda e: (ptr(e["x"]), e["n"]),
    lambda e, r: r == int(np.sum(e["x"])))
add("dot_i64", "wider", "np.dot(x, y)", "dot_i64", "ppi",
    lambda n, rng: {"x": rng.integers(-1000, 1000, n), "y": rng.integers(-1000, 1000, n)},
    lambda e: (ptr(e["x"]), ptr(e["y"]), e["n"]),
    lambda e, r: r == int(np.dot(e["x"], e["y"])))
def equal_pair(n, rng):
    x = rng.random(n)
    return {"x": x, "y": x.copy()}
add("array_equal_f64 (equal)", "wider", "np.array_equal(x, y)", "array_equal_f64", "ppi", equal_pair,
    lambda e: (ptr(e["x"]), ptr(e["y"]), e["n"]),
    lambda e, r: bool(r) == bool(np.array_equal(e["x"], e["y"])))
add("packbits", "wider", "np.packbits(x)", "packbits", "ppi",
    lambda n, rng: {"x": rng.random(n) < 0.5, "out": np.empty((n + 7) // 8, np.uint8)},
    lambda e: (ptr(e["x"]), ptr(e["out"]), e["n"]),
    lambda e, r: np.array_equal(e["out"], np.packbits(e["x"])),
    out="out", alloc=True)
for name, gen, dst in [
    ("cast_f64_f32", rng_f(np.float64), np.float32),
    ("cast_u8_f32", lambda n, rng: rng.integers(0, 256, n, dtype=np.uint8), np.float32),
    ("cast_i64_f64", lambda n, rng: rng.integers(-(1 << 40), 1 << 40, n), np.float64),
]:
    add(name, "wider", f"x.astype(np.{np.dtype(dst).name})", name, "ppi",
        lambda n, rng, gen=gen, dst=dst: {"x": gen(n, rng), "out": np.empty(n, dst), "dst": dst},
        lambda e: (ptr(e["x"]), ptr(e["out"]), e["n"]),
        lambda e, r: np.array_equal(e["out"], e["x"].astype(e["dst"])),
        out="out", alloc=True)
def interp_setup(n, rng):
    m = min(n, 1 << 20)
    return {"xp": np.sort(rng.random(n)), "fp": rng.random(n), "v": rng.random(m), "m": m, "out": np.empty(m)}
add("interp_f64", "wider", "np.interp(v, xp, fp)", "interp_f64", "pipppi", interp_setup,
    lambda e: (ptr(e["v"]), e["m"], ptr(e["xp"]), ptr(e["fp"]), e["n"], ptr(e["out"])),
    lambda e, r: np.allclose(e["out"], np.interp(e["v"], e["xp"], e["fp"]), rtol=1e-9, atol=1e-12),
    out="out", alloc=True, work=lambda e: e["m"])
def add_at_setup(n, rng):
    idx, v = rng.integers(0, 4096, n), rng.random(n)
    want = np.zeros(4096)
    np.add.at(want, idx, v)
    return {"acc": np.zeros(4096), "idx": idx, "v": v, "want": want}
add("add_at_f64 4096 bins", "wider", "np.add.at(acc, idx, v)", "add_at_f64", "pppi", add_at_setup,
    lambda e: (ptr(e["acc"]), ptr(e["idx"]), ptr(e["v"]), e["n"]),
    lambda e, r: np.array_equal(e["acc"], e["want"]))
def put_setup(n, rng):
    a, idx, v = rng.random(n), rng.integers(0, n, n), rng.random(n)
    want = a.copy()
    want[idx] = v
    return {"a": a, "idx": idx, "v": v, "want": want}
add("put_f64 a[idx]=v", "wider", "a[idx] = v", "put_f64", "pippi", put_setup,
    lambda e: (ptr(e["a"]), e["n"], ptr(e["idx"]), ptr(e["v"]), e["n"]),
    lambda e, r: r == 0 and np.array_equal(e["a"], e["want"]))
def maskfill_setup(n, rng):
    x, mask = rng.random(n), rng.random(n) < 0.5
    want = x.copy()
    want[mask] = 0.0
    return {"x": x, "mask": mask, "want": want}
add("maskfill_f64 x[mask]=0", "wider", "x[mask] = 0.0", "maskfill_f64", "ppid", maskfill_setup,
    lambda e: (ptr(e["x"]), ptr(e["mask"]), e["n"], 0.0),
    lambda e, r: np.array_equal(e["x"], e["want"]))
add("correlate_f64 k=8", "wider", "np.correlate(x, k, 'valid')", "correlate_f64", "pppii",
    lambda n, rng: {"x": rng.random(n + 7), "k": rng.random(8), "out": np.empty(n)},
    lambda e: (ptr(e["x"]), ptr(e["k"]), ptr(e["out"]), e["n"], 8),
    lambda e, r: np.allclose(e["out"], np.correlate(e["x"], e["k"], "valid"), rtol=1e-12),
    out="out", alloc=True)
add("histogram_f64 64 bins", "wider", "np.histogram(x, bins=64, range=(0.0, 1.0))", "histogram_f64", "pipidd",
    lambda n, rng: {"x": rng.random(n), "out": np.empty(64, np.int64)},
    lambda e: (ptr(e["x"]), e["n"], ptr(e["out"]), 64, 0.0, 1.0),
    lambda e, r: np.array_equal(e["out"], np.histogram(e["x"], bins=64, range=(0.0, 1.0))[0]),
    out="out", alloc=True)
add("copy_f64 [::-1] out=", "stepped slices", "np.copyto(out, x[::-1])", "copy_f64_strided", "ppii",
    lambda n, rng: {"x": rng.random(n), "out": np.empty(n)},
    lambda e: (ptr(e["x"]) + 8 * (e["n"] - 1), ptr(e["out"]), e["n"], -1),
    lambda e, r: np.array_equal(e["out"], e["x"][::-1]),
    out="out")

# ------------------------------------------------------------------ branchless C variants of the mask operations
for p in (0.5, 0.05):
    add(f"compress_f64 p={p} (branchless C)", "a[mask]", "x[mask]", "compress_f64_branchless", "pppi",
        lambda n, rng, p=p: {"x": rng.random(n), "mask": rng.random(n) < p, "out": np.empty(n)},
        lambda e: (ptr(e["x"]), ptr(e["mask"]), ptr(e["out"]), e["n"]),
        lambda e, r: np.array_equal(e["out"][:r], e["x"][e["mask"]]),
        out="out", alloc=True)
    add(f"nonzero_bool p={p} (branchless C)", "nonzero", "np.nonzero(mask)[0]", "nonzero_bool_branchless", "ppi",
        lambda n, rng, p=p: {"mask": rng.random(n) < p, "out": np.empty(n, np.int64)},
        lambda e: (ptr(e["mask"]), ptr(e["out"]), e["n"]),
        lambda e, r: np.array_equal(e["out"][:r], np.nonzero(e["mask"])[0]),
        out="out", alloc=True)

# ====================================================================== machinery

def build() -> ctypes.CDLL:
    BUILD.mkdir(exist_ok=True)
    lib = BUILD / "libnaive.dylib"
    subprocess.run(["clang", *CFLAGS, "-shared", "-o", str(lib), str(HERE / "naive.c")], check=True)
    return ctypes.CDLL(str(lib))


def vectorised() -> dict[str, list[str]]:
    """Which kernel loops did clang's loop vectoriser vectorise? Kernel -> list of 'line N: width W'.

    A kernel with an empty list has no vectorised loop. A kernel with several loops (isin, histogram,
    correlate, packbits) needs the line number read against naive.c to see which loop it was.
    """
    src = (HERE / "naive.c").read_text().splitlines()
    err = subprocess.run(
        ["clang", *CFLAGS, "-c", "-o", "/dev/null", str(HERE / "naive.c"),
         "-Rpass=loop-vectorize", "-fno-caret-diagnostics"],
        capture_output=True, text=True, check=True).stderr
    found: dict[str, list[str]] = {}
    for m in re.finditer(r"naive\.c:(\d+):\d+: remark: vectorized loop \(vectorization width: (\d+)", err):
        line = int(m.group(1))
        macro = re.match(r"[A-Z_]+\((\w+)", src[line - 1])   # a kernel generated by a macro
        if macro:
            name = macro.group(1)
        else:                                                # a loop inside a hand-written kernel
            name = next(f.group(1) for i in range(line - 1, -1, -1)
                        if (f := re.match(r"NOINLINE i64 (\w+)\(", src[i])))
        found.setdefault(name, []).append(f"line {line}: width {m.group(2)}")
    return found


def calibrate(run: Callable[[int], float]) -> int:
    reps = 1
    while True:
        t = run(reps)
        if t >= MIN_SAMPLE_S or reps >= 1 << 30:
            return reps
        reps = max(reps * 2, math.ceil(reps * MIN_SAMPLE_S / max(t, 1e-9) * 1.2))


def measure(lib, c: Cand, n: int, samples: int, budget_s: float) -> dict:
    rng = np.random.default_rng(1234)
    env = c.setup(n, rng)
    env["n"] = n
    env["np"] = np
    args = c.cargs(env)

    kern = getattr(lib, c.cfn)
    kern.argtypes = [CTYPES[t] for t in c.sig]
    kern.restype = ctypes.c_int64
    ok = bool(c.verify(env, kern(*args)))

    fresh = env[c.out].nbytes if (c.alloc and c.out) else 0
    extra = [fresh] if c.out else []
    bench = getattr(lib, "bench_" + c.cfn)
    bench.argtypes = [CTYPES[t] for t in c.sig] + [ctypes.c_int64] * (len(extra) + 1)
    bench.restype = ctypes.c_uint64
    timer = timeit.Timer(c.stmt, globals=env)

    run_c = lambda reps: bench(*args, *extra, reps) / 1e9
    run_np = lambda reps: timer.timeit(reps)
    reps_c, reps_np = calibrate(run_c), calibrate(run_np)

    t_c, t_np = [], []
    start = time.perf_counter()
    for i in range(samples):
        t_np.append(run_np(reps_np) / reps_np * 1e9)
        t_c.append(run_c(reps_c) / reps_c * 1e9)
        if i >= 4 and time.perf_counter() - start > budget_s:
            break
    np_ns, c_ns = statistics.median(t_np), statistics.median(t_c)
    return {
        "key": c.key, "group": c.group, "stmt": c.stmt, "cfn": c.cfn, "n": n, "work": c.work(env),
        "alloc": c.alloc, "naive_matches_numpy": ok, "samples": len(t_c),
        "numpy_ns": np_ns, "c_ns": c_ns, "numpy_min_ns": min(t_np), "c_min_ns": min(t_c),
        "numpy_over_c": np_ns / c_ns,
    }


def reference_ns(lib) -> float:
    x = np.random.default_rng(0).random(4096)
    res = np.empty(1)
    lib.bench_sum_f64.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64]
    lib.bench_sum_f64.restype = ctypes.c_uint64
    return min(lib.bench_sum_f64(ptr(x), 4096, 1, ptr(res), 20000) / 20000 for _ in range(15))


def sysctl(name: str) -> str:
    return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="regex on candidate key")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", help="result file stem (default: CPU name)")
    a = ap.parse_args()

    lib = build()
    lib.survey_init()
    sizes = SIZES[:-1] if a.quick else SIZES
    samples, budget = (5, 1.0) if a.quick else (11, 4.0)
    cands = [c for c in CANDS if not a.only or re.search(a.only, c.key)]

    ref_before = reference_ns(lib)
    rows = []
    for c in cands:
        for n in sizes:
            r = measure(lib, c, n, samples, budget)
            rows.append(r)
            flag = "" if r["naive_matches_numpy"] else "  NAIVE C DISAGREES WITH NUMPY"
            print(f"{c.key:28s} n={n:>9,}  numpy {r['numpy_ns']:>14,.1f} ns  C {r['c_ns']:>14,.1f} ns"
                  f"  numpy/C {r['numpy_over_c']:8.2f}x{flag}", flush=True)
    ref_after = reference_ns(lib)
    drift = abs(ref_after / ref_before - 1)
    print(f"reference loop drift: {drift:.1%}" + ("  WARNING: noisy machine" if drift > 0.03 else ""))

    from numpy.lib import introspect
    simd = {k: v for k, v in introspect.opt_func_info().items()
            if k in {"argmax", "argmin", "maximum", "minimum", "add", "not_equal", "logical_or", "logical_and"}}
    meta = {
        "cpu": sysctl("machdep.cpu.brand_string"), "os": platform.platform(),
        "clang": subprocess.run(["clang", "--version"], capture_output=True, text=True).stdout.splitlines()[0],
        "cflags": CFLAGS, "numpy": np.__version__, "python": platform.python_version(),
        "low_power_mode": "lowpowermode         1" in subprocess.run(
            ["pmset", "-g"], capture_output=True, text=True).stdout,
        "reference_drift": drift,
        "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip(),
        "c_vectorised": (lambda v: {fn: v.get(fn, []) for fn in sorted({c.cfn for c in CANDS})})(vectorised()),
        "numpy_simd_dispatch": simd,
    }
    RESULTS.mkdir(exist_ok=True)
    stem = a.out or re.sub(r"\W+", "-", meta["cpu"]).strip("-").lower()
    path = RESULTS / f"{stem}.json"
    path.write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    print(f"wrote {path}")
    return 0 if all(r["naive_matches_numpy"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
