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
from . import placement as P
from .spec import DEFAULT_SCALARS, Kernel, Out, element_count, np_dtype

# Array sizes timed by default: from data that fits in the fastest cache to
# data far larger than any cache.
SIZES = [16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216]
# Timing samples per contender per size: this many where a sample lasts no
# longer than TARGET_SAMPLE_NS, fewer where calls are slow (see _sample_count).
SAMPLES = 300
# Each sample repeats the call until it has run for at least this long, so
# timer resolution and call overhead are negligible.
#
# Samples are short and many on purpose. Whatever is too large for a core's
# own caches is slowed by the memory traffic of the other cores, and on a
# machine that is also someone's desktop that traffic comes in bursts a few
# milliseconds long, many times a second. A sample of 10 ms nearly always
# contains part of one, so its time is a mixture, and the median of 30 such
# samples moves with how much of a burst each caught. A sample of 1 ms is
# either clear of the bursts or inside one, most are clear, and the median is
# then the undisturbed time. Measured on an Intel Core i5-14600KF with a
# desktop session running, one function timed against itself twenty times
# over: with 30 samples of 10 ms the two medians differed by up to 16% at
# 4,194,304 and 16,777,216 bytes; with 300 samples of 1 ms by 0.1% and 0.3%.
# Up to 1,048,576 bytes they agreed to 1% before and to 0.14% after.
TARGET_SAMPLE_NS = 1_000_000
# Fewest samples taken per contender per size, however slow the calls are.
MIN_SAMPLES = 30

# `compare` takes its measurement in this many rounds, each in a process of its
# own, and splits its samples between them.
#
# Within one process two builds of the same source are not the same: they are
# loaded at different addresses, and that alone can cost one of them a
# mispredicted branch per call at one size. The samples of that process all
# carry the same offset, so their interval, a few tenths of a percent wide
# with short samples, does not contain it. Measured on the Intel Core
# i5-14600KF over 41 comparisons of a kernel with itself: the overall figure
# was off by up to 0.7%, and one size by up to 4%. A new process loads both
# builds somewhere else and draws the offset again, so the spread between
# rounds is the uncertainty that matters, and the interval is built from it.
COMPARE_ROUNDS = 5
# Two-sided 95% points of Student's t distribution, by degrees of freedom:
# what a mean of that many rounds plus one needs to turn its standard error
# into a 95% interval.
_T_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}

# A reduced run for smoke tests: two sizes, a third of the samples.
QUICK_SIZES = [1024, 65536]
QUICK_SAMPLES = 100

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
# In the table that times the close placement: the subject itself, timed with
# its output clear of the inputs as everywhere else.
PLACED_APART = "apart"
# The ctypes type that carries each scalar type into the C timing loop.
_SCALAR_CTYPES = {"usize": ctypes.c_size_t, "i64": ctypes.c_int64,
                  "f32": ctypes.c_float, "f64": ctypes.c_double}
# `pmset -g` lines that report Low Power Mode, and the value meaning "on".
_LOW_POWER_KEYS = ("lowpowermode", "powermode")
_LOW_POWER_ON = "1"
_PMSET_TIMEOUT_SECONDS = 5

# {distribution: {size: {contender: array of nanoseconds per call}}}
Measurements = dict[str, dict[int, dict[str, np.ndarray]]]


def _pool_size(elements: int) -> int:
    """Return how many different inputs to cycle through.

    Args:
        elements: Element count of the largest buffer of one call.

    Returns:
        The number of inputs: as many as keep the pool in the fastest cache,
        at least one and at most _POOL_MAX_INPUTS.
    """
    return max(1, min(_POOL_MAX_INPUTS, _POOL_TOTAL_ELEMENTS // max(elements, 1)))


def _inputs(k: Kernel, dist: str, n: int, seed: int,
            scalars: dict[str, int | float]) -> dict[str, np.ndarray]:
    """Build one set of timing inputs, from the spec's distribution or random data.

    Args:
        k: The kernel's spec.
        dist: Name of the input distribution.
        n: The value of the kernel's Size argument.
        seed: Seed of the random data.
        scalars: The scalar values of the call, by argument name. They give
            the length of any input sized by a Scalar.

    Returns:
        One contiguous array per input argument, by argument name.
    """
    if k.distributions and dist in k.distributions:
        rng = np.random.default_rng([seed, n, cases._crc(dist)])
        made = k.distributions[dist](rng, n)
        return {a.name: np.ascontiguousarray(made[a.name], np_dtype(a.dtype)) for a in k.ins}
    return {a.name: cases.default_input(a, n, seed, scalars) for a in k.ins}


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


def _calibrate(run: Callable[[int], int]) -> tuple[int, float]:
    """Find a repeat count for which one sample lasts about TARGET_SAMPLE_NS.

    Args:
        run: Times the given number of calls and returns the nanoseconds taken.

    Returns:
        The repeat count to use for every sample, and how long one sample is
        then expected to last, in nanoseconds. That is longer than
        TARGET_SAMPLE_NS when a single call already is.
    """
    trial = 1
    while True:
        elapsed = run(trial)
        if elapsed >= TARGET_SAMPLE_NS / _CALIBRATION_FRACTION or trial >= _MAX_REPS:
            break
        trial *= _CALIBRATION_GROWTH
    reps = max(1, int(trial * TARGET_SAMPLE_NS / max(elapsed, 1)))
    return reps, elapsed / trial * reps


def _sample_count(sample_ns: list[float], samples: int) -> int:
    """Decide how many samples to take of each contender at one size.

    The time a size is given is fixed: what `samples` samples of
    TARGET_SAMPLE_NS take for every contender. Where a single call of some
    contender lasts longer than that, fewer samples fit.

    Args:
        sample_ns: How long one sample of each contender lasts, in nanoseconds.
        samples: The most samples to take of each contender.

    Returns:
        The number of samples of each contender that fit the time, at most
        `samples` and at least MIN_SAMPLES, or `samples` if that is smaller.
    """
    budget_ns = samples * TARGET_SAMPLE_NS * len(sample_ns)
    fitting = int(budget_ns / max(sum(sample_ns), 1))
    return max(min(MIN_SAMPLES, samples), min(samples, fitting))


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


def _placed_outputs(k: Kernel, pool: list[dict[str, np.ndarray]], n: int,
                    scalars: dict[str, int | float], placement: str) -> list[dict[str, np.ndarray]]:
    """Make the output buffers of every pool entry, placed against that entry's inputs.

    The allocator is not asked where an output goes, because where it sits
    relative to the inputs changes the time of a call (see judge/placement.py).
    Only Out buffers are placed. An in-place buffer is its own input.

    Args:
        k: The kernel's spec.
        pool: The input arrays of each pool entry, by argument name.
        n: The value of the kernel's Size argument.
        scalars: The scalar values of the call, by argument name.
        placement: P.APART or P.CLOSE.

    Returns:
        For each pool entry, its uninitialised output arrays by argument name.
        Empty dictionaries for a kernel without an Out buffer.
    """
    out_args = [a for a in k.outs if isinstance(a, Out)]
    shapes = [(element_count(a, n, scalars=scalars), np_dtype(a.dtype)) for a in out_args]
    calls = [[entry[a.name] for a in k.ins] for entry in pool]
    return [{a.name: array for a, array in zip(out_args, made)}
            for made in P.place(calls, shapes, placement)]


def _timing_setup(k: Kernel, harness: Harness, contenders: dict[str, int | None],
                  dist: str, n: int, scalars: str = DEFAULT_SCALARS,
                  placement: str = P.APART) -> _TimingSetup:
    """Build one timing function per contender for a given distribution and size.

    Args:
        k: The kernel's spec.
        harness: The compiled timing loop.
        contenders: Contender name -> native function address, or None for the
            NumPy oracle.
        dist: Name of the input distribution.
        n: Element count.
        scalars: Name of the scalar set whose values every call is given.
        placement: Where the output buffers of the native contenders sit
            relative to the inputs: P.APART or P.CLOSE. The NumPy oracle
            allocates its own result, so this does not reach it.

    Returns:
        The timing functions and the buffers they point into.
    """
    scalar_values = k.scalar_values(scalars)
    # A scalar can make a buffer longer than `n`; the pool is sized by the
    # largest, so that all of it still fits the fastest cache.
    npool = _pool_size(k.largest_buffer(n, scalars))
    pool = [_inputs(k, dist, n, seed, scalar_values) for seed in range(npool)]
    outs = _placed_outputs(k, pool, n, scalar_values, placement)
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


def has_placed_output(k: Kernel) -> bool:
    """Say whether a kernel writes a buffer whose place relative to an input can vary.

    Args:
        k: The kernel's spec.

    Returns:
        True for a kernel with at least one Out buffer and at least one input.
    """
    return bool(k.ins) and any(isinstance(a, Out) for a in k.outs)


def measure_close(k: Kernel, harness: Harness, subject: str, contenders: dict[str, int],
                  sizes: list[int] = SIZES, samples: int = SAMPLES,
                  progress: Callable[[str, int], None] | None = None) -> Measurements:
    """Time native contenders with each output buffer just behind its input.

    This is the placement every other timing avoids (see judge/placement.py).
    It is timed because a caller's two arrays of a power-of-two size often sit
    exactly there. One table is taken: the first distribution, with the
    default scalars. It starts at the first size that is timed on a single
    input: the outputs of a pool of small inputs cannot each be close to their
    own input and stay close to one another, and spreading them out would be
    timed instead.

    Args:
        k: The kernel's spec. It must have an Out buffer and an input.
        harness: The compiled timing loop.
        subject: Name of the contender the table is about.
        contenders: Contender name -> native function address, the subject
            among them. The NumPy oracle cannot take part: it allocates its
            own result.
        sizes: Element counts to time. Those below the kernel's `min_n`, and
            those timed on a pool of several inputs, are skipped.
        samples: Timing samples per contender per size.
        progress: Called with (table name, size) after each size finishes.

    Returns:
        The samples in the shape `measure` returns, with one more contender,
        PLACED_APART: the subject timed on the same inputs with its output
        clear of them, interleaved with the rest.
    """
    dist = next(iter(k.distributions)) if k.distributions else DEFAULT_DISTRIBUTION
    label = timing_label(dist, DEFAULT_SCALARS)
    data: Measurements = {label: {}}
    for n in sizes:
        if n < k.min_n or _pool_size(k.largest_buffer(n)) > 1:
            continue
        close = _timing_setup(k, harness, contenders, dist, n, placement=P.CLOSE)
        apart = _timing_setup(k, harness, {subject: contenders[subject]}, dist, n)
        runners = {**close.runners, PLACED_APART: apart.runners[subject]}
        data[label][n] = _sample(runners, samples)
        if progress:
            progress(label, n)
    return data


def _sample(runners: dict[str, Callable[[int], int]], samples: int) -> dict[str, np.ndarray]:
    """Take interleaved timing samples of several contenders on the same inputs.

    Args:
        runners: Contender name -> function that takes a repeat count and
            returns elapsed nanoseconds.
        samples: The most samples to take of each contender. Fewer are taken
            where calls are slow (see `_sample_count`); every contender gets
            the same number.

    Returns:
        Contender name -> array of nanoseconds per call, one entry per sample.
    """
    calibrated = {name: _calibrate(run) for name, run in runners.items()}
    reps = {name: count for name, (count, _) in calibrated.items()}
    samples = _sample_count([lasting for _, lasting in calibrated.values()], samples)
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


def _bounded(point: float, lo: float, hi: float) -> dict:
    """Describe a speedup by its point estimate and the ends of its 95% interval.

    Args:
        point: The speedup.
        lo: Lower end of the interval.
        hi: Upper end of the interval.

    Returns:
        The three figures and whether the speedup is significant, which it is
        when the interval does not include 1.0.
    """
    return {"speedup": point, "lo": lo, "hi": hi, "significant": bool(lo > 1.0 or hi < 1.0)}


def _summary(point: float, resamples: np.ndarray) -> dict:
    """Summarise a speedup: point estimate, 95% interval, and significance.

    A speedup is significant when its interval does not include 1.0.
    """
    lo, hi = (float(v) for v in np.percentile(resamples, _INTERVAL_PERCENTILES))
    return _bounded(point, lo, hi)


def _across_rounds(logs: np.ndarray) -> dict:
    """Summarise a speedup measured once per round.

    Args:
        logs: The natural logarithm of the speedup in each round; at least two.

    Returns:
        The geometric mean of the rounds as the speedup, with a 95% interval
        from the spread between rounds (Student's t on the logarithms).
    """
    mean = float(np.mean(logs))
    half = _T_95[len(logs) - 1] * float(np.std(logs, ddof=1)) / np.sqrt(len(logs))
    return _bounded(float(np.exp(mean)), float(np.exp(mean - half)), float(np.exp(mean + half)))


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
            row = {"n": n, "samples": len(times[kernel]),
                   "median_ns": {name: float(np.median(t)) for name, t in times.items()}}
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


def analyse_rounds(rounds: list[Measurements], kernel: str, baselines: list[str]) -> dict:
    """Turn the timings of several rounds into speedups of `kernel` over each baseline.

    Each round was measured in a process of its own. Within a round a speedup
    is the ratio of median times, as in `analyse`. Across rounds it is the
    geometric mean of those ratios, and its interval comes from how much the
    rounds differ, not from the samples within one: see COMPARE_ROUNDS.

    Args:
        rounds: The measurements of each round, all of the same tables, sizes
            and contenders. At least two.
        kernel: Name of the contender the report is about.
        baselines: Names of the contenders it is compared with.

    Returns:
        A report in the shape `analyse` returns. A row's `samples` is the
        total over the rounds, and its `median_ns` the median over the rounds
        of each round's median.
    """
    report: dict = {"distributions": {}, "overall": {}}
    everything: dict[str, list[np.ndarray]] = {b: [] for b in baselines}
    for dist, by_size in rounds[0].items():
        rows = []
        logs: dict[str, list[np.ndarray]] = {b: [] for b in baselines}
        for n in by_size:
            taken = [measured[dist][n] for measured in rounds]
            row = {"n": n, "samples": sum(len(times[kernel]) for times in taken),
                   "median_ns": {name: float(np.median([np.median(times[name]) for times in taken]))
                                 for name in by_size[n]}}
            for b in baselines:
                per_round = np.log([np.median(times[b]) / np.median(times[kernel]) for times in taken])
                row[f"vs_{b}"] = _across_rounds(per_round)
                logs[b].append(per_round)
            rows.append(row)
        # One value per round: the mean over the sizes of that round's logarithms.
        overall = {f"vs_{b}": _across_rounds(np.mean(logs[b], axis=0)) for b in baselines}
        report["distributions"][dist] = {"sizes": rows, "overall": overall}
        for b in baselines:
            everything[b] += logs[b]
    for b in baselines:
        report["overall"][f"vs_{b}"] = _across_rounds(np.mean(everything[b], axis=0))
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
