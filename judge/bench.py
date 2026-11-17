"""Timing: measures contenders and reports speedups with confidence intervals.

There are no pass or fail thresholds here. The judge reports what it measured
and how certain it is; a person decides what counts as a win.

A "contender" is anything that can be timed on the same inputs: the assembly
kernel, the naive C reference, an earlier version of the kernel, or the NumPy
oracle called from Python.
"""
from __future__ import annotations

import ctypes
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from . import cases, cores
from .spec import DEFAULT_SCALARS, Kernel, Out, element_count, np_dtype

# Array sizes timed by default: from data that fits in the fastest cache to
# data far larger than any cache.
SIZES = [16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216]
# Timing samples per contender per size.
SAMPLES = 30
# Each sample repeats the call until it has run for at least this long, so
# timer resolution and call overhead are negligible.
TARGET_SAMPLE_NS = 10_000_000

# A reduced run for smoke tests: two sizes, few samples.
QUICK_SIZES = [1024, 65536]
QUICK_SAMPLES = 10

# Calibration multiplies the repeat count by this factor until a trial run is
# long enough to extrapolate from.
_CALIBRATION_GROWTH = 4
# A trial run this fraction of the target is long enough to extrapolate from.
_CALIBRATION_FRACTION = 5
# Upper bound on the repeat count, as a guard against a zero-time kernel.
_MAX_REPS = 1 << 28

# At small sizes the timing loop cycles through several different inputs.
# Timing one input repeatedly lets the processor memorise its branch pattern.
# The pool holds about this many elements in total, capped at a maximum count,
# so it still fits in the fastest cache.
_POOL_TOTAL_ELEMENTS = 4096
_POOL_MAX_INPUTS = 64

# Resamples drawn to estimate each confidence interval.
BOOTSTRAP_RESAMPLES = 2000
# Percentiles that bound a 95% confidence interval.
_INTERVAL_PERCENTILES = [2.5, 97.5]
# Fixed seed so the same measurements always give the same intervals.
_BOOTSTRAP_SEED = 0

# The reference loop is timed this many times and the fastest is kept.
_CANARY_REPEATS = 5
# If the reference loop's time changes by more than this during a run, the
# machine was noisy and the report says so.
CANARY_TOLERANCE = 0.03

DEFAULT_DISTRIBUTION = "random"
# The ctypes type that carries each scalar type into the C timing loop.
_SCALAR_CTYPES = {"usize": ctypes.c_size_t, "i64": ctypes.c_int64,
                  "f32": ctypes.c_float, "f64": ctypes.c_double}
# `pmset -g` lines that report Low Power Mode, and the value meaning "on".
_LOW_POWER_KEYS = ("lowpowermode", "powermode")
_LOW_POWER_ON = "1"
_PMSET_TIMEOUT_SECONDS = 5

# {distribution: {size: {contender: array of nanoseconds per call}}}
Measurements = dict[str, dict[int, dict[str, np.ndarray]]]


def _pool_size(n: int) -> int:
    """Return how many different inputs to cycle through at element count `n`."""
    return max(1, min(_POOL_MAX_INPUTS, _POOL_TOTAL_ELEMENTS // n))


def _inputs(k: Kernel, dist: str, n: int, seed: int) -> dict[str, np.ndarray]:
    """Build one set of timing inputs, from the spec's distribution or random data."""
    if k.distributions and dist in k.distributions:
        rng = np.random.default_rng([seed, n, cases._crc(dist)])
        made = k.distributions[dist](rng, n)
        return {a.name: np.ascontiguousarray(made[a.name], np_dtype(a.dtype)) for a in k.ins}
    return {a.name: cases.default_input(a, n, seed) for a in k.ins}


class Harness:
    """The compiled timing loop for one kernel, loaded and ready to call."""

    def __init__(self, k: Kernel, path: Path) -> None:
        """Load the harness library and declare the signature of its timing loop."""
        self.k = k
        self.lib = ctypes.CDLL(str(path))
        # The kernel's buffer arguments, in order. The timing loop takes one
        # array of pointers per buffer argument, indexed by pool entry.
        self.pointer_args = k.buffers
        self.lib.hk_bench.argtypes = (
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]
            + [ctypes.POINTER(ctypes.c_void_p)] * len(self.pointer_args)
            + [ctypes.c_uint64]
            + [_SCALAR_CTYPES[a.dtype] for a in k.scalars]
        )
        self.lib.hk_bench.restype = ctypes.c_uint64
        self.lib.hk_canary.restype = ctypes.c_uint64
        self.lib.hk_init()
        # The performance core this thread is pinned to, on Linux. Everything
        # timed after this point, NumPy included, runs on it.
        self.core = cores.claim()

    def canary(self) -> int:
        """Time the fixed reference loop and return the fastest of several runs, in ns."""
        return min(self.lib.hk_canary() for _ in range(_CANARY_REPEATS))


def function_address(dylib: Path, name: str) -> tuple[int, ctypes.CDLL]:
    """Load a library and return the address of one function in it.

    The library object is returned too; the caller must keep it alive for as
    long as the address is used.
    """
    lib = ctypes.CDLL(str(dylib))
    return ctypes.cast(getattr(lib, name), ctypes.c_void_p).value, lib


def _calibrate(run: Callable[[int], int]) -> int:
    """Find a repeat count for which one sample lasts about TARGET_SAMPLE_NS."""
    reps = 1
    while True:
        elapsed = run(reps)
        if elapsed >= TARGET_SAMPLE_NS / _CALIBRATION_FRACTION or reps >= _MAX_REPS:
            break
        reps *= _CALIBRATION_GROWTH
    return max(1, int(reps * TARGET_SAMPLE_NS / max(elapsed, 1)))


@dataclass
class _TimingSetup:
    """The timing functions for one distribution and size, with the memory they use."""

    # One function per contender: takes a repeat count, returns elapsed nanoseconds.
    runners: dict[str, Callable[[int], int]]
    # The input and output arrays the native runners point into. Held here so
    # they stay allocated for as long as the runners are in use.
    buffers: list


def timing_label(dist: str, scalars: str) -> str:
    """Name one timing table: an input distribution timed with one scalar set.

    Args:
        dist: Name of the input distribution.
        scalars: Name of the scalar set.

    Returns:
        The distribution's name alone for the default scalar set, so kernels
        without timed scalar sets are reported as before; otherwise both.
    """
    return dist if scalars == DEFAULT_SCALARS else f"{dist}, scalars={scalars}"


def _timing_setup(k: Kernel, harness: Harness, contenders: dict[str, int | None],
                  dist: str, n: int, scalars: str = DEFAULT_SCALARS) -> _TimingSetup:
    """Build one timing function per contender for a given distribution and size.

    Args:
        k: The kernel's spec.
        harness: The compiled timing loop.
        contenders: Contender name -> native function address, or None for the
            NumPy oracle.
        dist: Name of the input distribution.
        n: Element count.
        scalars: Name of the scalar set whose values every call is given.

    Returns:
        The timing functions and the buffers they point into.
    """
    npool = _pool_size(n)
    pool = [_inputs(k, dist, n, seed) for seed in range(npool)]
    scalar_values = k.scalar_values(scalars)
    outs = [{a.name: np.empty(element_count(a, n, scalars=scalar_values), np_dtype(a.dtype))
             for a in k.outs if isinstance(a, Out)} for _ in range(npool)]
    native_scalars = [scalar_values[a.name] for a in k.scalars]

    # One array of pointers per buffer argument, one pointer per pool entry.
    # An in-place buffer points at its input, which the kernel then modifies:
    # every later repeat sees the result of the first. That is only a fair
    # timing for operations that give the same result when applied twice.
    pointer_arrays = []
    for a in harness.pointer_args:
        source = outs if isinstance(a, Out) else pool
        addresses = [entry[a.name].ctypes.data for entry in source]
        pointer_arrays.append((ctypes.c_void_p * npool)(*addresses))
    oracle_scalars = k.oracle_scalars(scalars)
    numpy_args = [(*(entry[a.name] for a in k.ins), *oracle_scalars) for entry in pool]

    def native(fn: int) -> Callable[[int], int]:
        """Time a compiled function through the C timing loop."""
        def run(reps: int) -> int:
            """Call the function `reps` times and return elapsed nanoseconds."""
            return harness.lib.hk_bench(fn, reps, npool, *pointer_arrays, n, *native_scalars)
        return run

    def numpy_oracle(reps: int) -> int:
        """Time the NumPy oracle from Python, including its call overhead."""
        oracle, p = k.numpy or k.oracle, 0
        start = time.perf_counter_ns()
        for _ in range(reps):
            oracle(*numpy_args[p])
            p += 1
            if p == npool:
                p = 0
        return time.perf_counter_ns() - start

    runners = {name: numpy_oracle if fn is None else native(fn) for name, fn in contenders.items()}
    return _TimingSetup(runners, [pool, outs])


def measure(k: Kernel, harness: Harness, contenders: dict[str, int | None],
            sizes: list[int] = SIZES, samples: int = SAMPLES,
            progress: Callable[[str, int], None] | None = None) -> Measurements:
    """Time every contender at every size, for each input distribution.

    Samples from the contenders are interleaved, and the starting contender
    rotates, so slow drift in the machine affects all of them equally. A
    kernel with timed scalar sets gets every distribution once per set; the
    tables are named by `timing_label`.

    Args:
        k: The kernel's spec.
        harness: The compiled timing loop.
        contenders: Contender name -> native function address, or None for the
            NumPy oracle.
        sizes: Element counts to time. Those below the kernel's `min_n` are skipped.
        samples: Timing samples per contender per size.
        progress: Called with (table name, size) after each size finishes.

    Returns:
        The samples, in nanoseconds per call, by table name, size and contender.
    """
    dists = list(k.distributions) if k.distributions else [DEFAULT_DISTRIBUTION]
    data: Measurements = {}
    for dist in dists:
        for scalars in k.timed_scalar_sets:
            label = timing_label(dist, scalars)
            data[label] = {}
            for n in sizes:
                if n < k.min_n:
                    continue
                setup = _timing_setup(k, harness, contenders, dist, n, scalars)
                data[label][n] = _sample(setup.runners, samples)
                if progress:
                    progress(label, n)
    return data


def _sample(runners: dict[str, Callable[[int], int]], samples: int) -> dict[str, np.ndarray]:
    """Take interleaved timing samples of several contenders on the same inputs.

    Args:
        runners: Contender name -> function that takes a repeat count and
            returns elapsed nanoseconds.
        samples: How many samples to take of each contender.

    Returns:
        Contender name -> array of nanoseconds per call, one entry per sample.
    """
    reps = {name: _calibrate(run) for name, run in runners.items()}
    times: dict[str, list[float]] = {name: [] for name in runners}
    order = list(runners)
    for sample in range(samples):
        # Rotate which contender goes first, so none always follows the same one.
        shift = sample % len(order)
        for name in order[shift:] + order[:shift]:
            times[name].append(runners[name](reps[name]) / reps[name])
    return {name: np.array(t) for name, t in times.items()}


def _ratio(base: np.ndarray, kern: np.ndarray, rng: np.random.Generator) -> tuple[float, np.ndarray]:
    """Return the speedup of `kern` over `base`, with bootstrap resamples of it.

    The speedup is the ratio of median times. Samples are resampled in pairs by
    index, because sample i of each contender was taken at nearly the same
    moment and shares whatever noise the machine had then.
    """
    idx = rng.integers(0, len(base), (BOOTSTRAP_RESAMPLES, len(base)))
    resamples = np.median(base[idx], axis=1) / np.median(kern[idx], axis=1)
    return float(np.median(base) / np.median(kern)), resamples


def _summary(point: float, resamples: np.ndarray) -> dict:
    """Summarise a speedup: point estimate, 95% interval, and significance.

    A speedup is significant when its interval does not include 1.0.
    """
    lo, hi = (float(v) for v in np.percentile(resamples, _INTERVAL_PERCENTILES))
    return {"speedup": point, "lo": lo, "hi": hi, "significant": bool(lo > 1.0 or hi < 1.0)}


def _geomean_summary(points: list[float], resamples: list[np.ndarray]) -> dict:
    """Summarise several speedups as their geometric mean, with its own interval."""
    point = float(np.exp(np.mean(np.log(points))))
    combined = np.exp(np.mean(np.log(np.array(resamples)), axis=0))
    return _summary(point, combined)


def analyse(data: Measurements, kernel: str, baselines: list[str]) -> dict:
    """Turn raw timings into speedups of `kernel` over each baseline.

    Returns a report with, per distribution, one row per size and an overall
    geometric mean, plus an overall figure across all distributions. Each
    speedup is stored under the key `vs_<baseline>`.
    """
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    report: dict = {"distributions": {}, "overall": {}}
    all_points: dict[str, list] = {b: [] for b in baselines}
    all_resamples: dict[str, list] = {b: [] for b in baselines}
    for dist, by_size in data.items():
        rows = []
        points: dict[str, list] = {b: [] for b in baselines}
        resamples: dict[str, list] = {b: [] for b in baselines}
        for n, times in by_size.items():
            row = {"n": n, "median_ns": {name: float(np.median(t)) for name, t in times.items()}}
            for b in baselines:
                point, drawn = _ratio(times[b], times[kernel], rng)
                row[f"vs_{b}"] = _summary(point, drawn)
                points[b].append(point)
                resamples[b].append(drawn)
            rows.append(row)
        overall = {f"vs_{b}": _geomean_summary(points[b], resamples[b]) for b in baselines}
        report["distributions"][dist] = {"sizes": rows, "overall": overall}
        for b in baselines:
            all_points[b] += points[b]
            all_resamples[b] += resamples[b]
    for b in baselines:
        report["overall"][f"vs_{b}"] = _geomean_summary(all_points[b], all_resamples[b])
    return report


def low_power_mode() -> bool:
    """Report whether macOS Low Power Mode is on. False when it cannot be determined."""
    try:
        out = subprocess.run(["pmset", "-g"], capture_output=True, text=True,
                             timeout=_PMSET_TIMEOUT_SECONDS).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    for line in out.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] in _LOW_POWER_KEYS and fields[1] == _LOW_POWER_ON:
            return True
    return False


def noise_warnings(canary_before: int, canary_after: int,
                   core: cores.Claim | None = None) -> list[str]:
    """Return warnings to show if the machine was too noisy to trust the timings.

    Args:
        canary_before: Time of the reference loop before the run, in nanoseconds.
        canary_after: Its time after the run.
        core: The core the timing thread was pinned to, from the harness; None
            where threads are not pinned.

    Returns:
        One warning per sign of trouble; empty when the timings can be trusted.
    """
    warnings = []
    drift = abs(canary_after / canary_before - 1)
    if drift > CANARY_TOLERANCE:
        warnings.append(
            f"NOISY MACHINE: the reference loop changed by {drift:.1%} during the run. "
            "Treat these numbers with caution and consider rerunning."
        )
    if low_power_mode():
        warnings.append("NOISY MACHINE: Low Power Mode is on. Timings are not representative.")
    warnings += [f"NOISY MACHINE: {problem}" for problem in cores.throttle_warnings(core)]
    return warnings
