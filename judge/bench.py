"""Timing: measures contenders and reports speedups with confidence intervals.

There are no pass or fail thresholds here. The judge reports; a person decides.
"""
from __future__ import annotations

import ctypes
import subprocess
import time
from pathlib import Path

import numpy as np

from . import cases
from .spec import In, Kernel, Out, np_dtype

SIZES = [16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216]
SAMPLES = 30
TARGET_NS = 10_000_000  # one sample takes at least this long
BOOTSTRAP = 2000
CANARY_TOLERANCE = 0.03


def _pool_size(n: int) -> int:
    """How many different inputs to cycle through, so branch patterns cannot be memorised."""
    return max(1, min(64, 4096 // n))


def _inputs(k: Kernel, dist: str, n: int, seed: int) -> dict[str, np.ndarray]:
    if k.distributions and dist in k.distributions:
        rng = np.random.default_rng([seed, n, cases._crc(dist)])
        made = k.distributions[dist](rng, n)
        return {a.name: np.ascontiguousarray(made[a.name], np_dtype(a.dtype)) for a in k.ins}
    return {a.name: np.ascontiguousarray(cases.base(a.dtype, n, seed, a.name)) for a in k.ins}


class Harness:
    def __init__(self, k: Kernel, path: Path) -> None:
        self.k = k
        self.lib = ctypes.CDLL(str(path))
        self.pointer_args = [a for a in k.args if isinstance(a, (In, Out))]
        self.lib.hk_bench.argtypes = (
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]
            + [ctypes.POINTER(ctypes.c_void_p)] * len(self.pointer_args)
            + [ctypes.c_uint64]
        )
        self.lib.hk_bench.restype = ctypes.c_uint64
        self.lib.hk_canary.restype = ctypes.c_uint64
        self.lib.hk_init()

    def canary(self) -> int:
        return min(self.lib.hk_canary() for _ in range(5))


def function_address(dylib: Path, name: str) -> tuple[int, ctypes.CDLL]:
    lib = ctypes.CDLL(str(dylib))
    return ctypes.cast(getattr(lib, name), ctypes.c_void_p).value, lib


def _calibrate(run) -> int:
    reps = 1
    while True:
        elapsed = run(reps)
        if elapsed >= TARGET_NS / 5 or reps >= 1 << 28:
            break
        reps *= 4
    return max(1, int(reps * TARGET_NS / max(elapsed, 1)))


def measure(k: Kernel, harness: Harness, contenders: dict[str, int | None],
            sizes: list[int] = SIZES, samples: int = SAMPLES, progress=None) -> dict:
    """Time each contender. A contender is a function address, or None for the NumPy oracle.

    Returns {distribution: {n: {contender: array of ns per call}}}.
    """
    dists = list(k.distributions) if k.distributions else ["random"]
    data: dict = {}
    for dist in dists:
        data[dist] = {}
        for n in sizes:
            if n < k.min_n:
                continue
            npool = _pool_size(n)
            pool = [_inputs(k, dist, n, seed) for seed in range(npool)]
            outs = [
                {a.name: np.empty(n, np_dtype(a.dtype)) for a in k.outs} for _ in range(npool)
            ]
            pointer_arrays = []
            for a in harness.pointer_args:
                source = pool if isinstance(a, In) else outs
                pointer_arrays.append(
                    (ctypes.c_void_p * npool)(*[entry[a.name].ctypes.data for entry in source])
                )
            numpy_args = [tuple(entry[a.name] for a in k.ins) for entry in pool]

            def run_native(fn):
                def run(reps):
                    return harness.lib.hk_bench(fn, reps, npool, *pointer_arrays, n)
                return run

            def run_numpy(reps):
                oracle, p = k.oracle, 0
                t0 = time.perf_counter_ns()
                for _ in range(reps):
                    oracle(*numpy_args[p])
                    p += 1
                    if p == npool:
                        p = 0
                return time.perf_counter_ns() - t0

            runners = {name: run_numpy if fn is None else run_native(fn)
                       for name, fn in contenders.items()}
            reps = {name: _calibrate(run) for name, run in runners.items()}
            times = {name: [] for name in runners}
            order = list(runners)
            for s in range(samples):
                # Alternate contenders, and rotate who goes first, so drift hits all equally.
                for name in order[s % len(order):] + order[: s % len(order)]:
                    times[name].append(runners[name](reps[name]) / reps[name])
            data[dist][n] = {name: np.array(t) for name, t in times.items()}
            if progress:
                progress(dist, n)
    return data


def _ratio(base: np.ndarray, kern: np.ndarray, rng: np.random.Generator) -> tuple[float, np.ndarray]:
    """Speedup of kern over base, with bootstrap replicates. Samples are paired by index."""
    idx = rng.integers(0, len(base), (BOOTSTRAP, len(base)))
    replicates = np.median(base[idx], axis=1) / np.median(kern[idx], axis=1)
    return float(np.median(base) / np.median(kern)), replicates


def _summary(point: float, replicates: np.ndarray) -> dict:
    lo, hi = (float(v) for v in np.percentile(replicates, [2.5, 97.5]))
    return {"speedup": point, "lo": lo, "hi": hi, "significant": bool(lo > 1.0 or hi < 1.0)}


def analyse(data: dict, kernel: str, baselines: list[str]) -> dict:
    """Per-size and overall speedups of `kernel` over each baseline."""
    rng = np.random.default_rng(0)
    report = {"distributions": {}, "overall": {}}
    all_points = {b: [] for b in baselines}
    all_reps = {b: [] for b in baselines}
    for dist, by_n in data.items():
        rows = []
        points = {b: [] for b in baselines}
        reps = {b: [] for b in baselines}
        for n, times in by_n.items():
            row = {"n": n, "median_ns": {name: float(np.median(t)) for name, t in times.items()}}
            for b in baselines:
                point, replicates = _ratio(times[b], times[kernel], rng)
                row[f"vs_{b}"] = _summary(point, replicates)
                points[b].append(point)
                reps[b].append(replicates)
            rows.append(row)
        overall = {}
        for b in baselines:
            overall[f"vs_{b}"] = _geomean_summary(points[b], reps[b])
            all_points[b] += points[b]
            all_reps[b] += reps[b]
        report["distributions"][dist] = {"sizes": rows, "overall": overall}
    for b in baselines:
        report["overall"][f"vs_{b}"] = _geomean_summary(all_points[b], all_reps[b])
    return report


def _geomean_summary(points: list[float], replicates: list[np.ndarray]) -> dict:
    point = float(np.exp(np.mean(np.log(points))))
    reps = np.exp(np.mean(np.log(np.array(replicates)), axis=0))
    return _summary(point, reps)


def low_power_mode() -> bool:
    try:
        out = subprocess.run(["pmset", "-g"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in ("lowpowermode", "powermode") and parts[1] == "1":
            return True
    return False


def noise_warnings(canary_before: int, canary_after: int) -> list[str]:
    warnings = []
    drift = abs(canary_after / canary_before - 1)
    if drift > CANARY_TOLERANCE:
        warnings.append(
            f"NOISY MACHINE: the reference loop changed by {drift:.1%} during the run. "
            "Treat these numbers with caution and consider rerunning."
        )
    if low_power_mode():
        warnings.append("NOISY MACHINE: Low Power Mode is on. Timings are not representative.")
    return warnings


def fmt_ns(ns: float) -> str:
    if ns < 1e3:
        return f"{ns:.1f} ns"
    if ns < 1e6:
        return f"{ns / 1e3:.1f} us"
    return f"{ns / 1e6:.2f} ms"


def fmt_speedup(s: dict) -> str:
    x = s["speedup"]
    value = f"{x:.0f}x" if x >= 100 else f"{x:.1f}x" if x >= 10 else f"{x:.2f}x"
    lo, hi = s["lo"], s["hi"]
    interval = f"{lo:.0f} – {hi:.0f}" if x >= 100 else f"{lo:.2f} – {hi:.2f}"
    mark = "" if s["significant"] else " n.s."
    return f"{value:>7}  [{interval}]{mark}"


def format_table(report: dict, kernel: str, baselines: list[str], labels: dict[str, str]) -> str:
    lines = []
    width = 30
    for dist, section in report["distributions"].items():
        if len(report["distributions"]) > 1 or dist != "random":
            lines.append(f"distribution: {dist}")
        header = f"{'n':>12}  {labels[kernel]:>10}"
        for b in baselines:
            header += f"  {labels[b]:>10}  {'speedup  [95% interval]':<{width}}"
        lines.append(header)
        for row in section["sizes"]:
            line = f"{row['n']:>12,}  {fmt_ns(row['median_ns'][kernel]):>10}"
            for b in baselines:
                line += f"  {fmt_ns(row['median_ns'][b]):>10}  {fmt_speedup(row[f'vs_{b}']):<{width}}"
            lines.append(line)
        line = f"{'overall':>12}  {'':>10}"
        for b in baselines:
            line += f"  {'':>10}  {fmt_speedup(section['overall'][f'vs_{b}']):<{width}}"
        lines.append(line)
        lines.append("")
    if len(report["distributions"]) > 1:
        line = f"{'all':>12}  {'':>10}"
        for b in baselines:
            line += f"  {'':>10}  {fmt_speedup(report['overall'][f'vs_{b}']):<{width}}"
        lines += [line, ""]
    lines.append("n.s. = not statistically significant (the 95% interval includes 1.0)")
    return "\n".join(lines)
