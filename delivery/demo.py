"""Runs an unmodified NumPy program with and without hotkernels.

The program is `examples/workload.py`, which knows nothing of hotkernels.
It is loaded from its file, run on plain NumPy and run again after
`hotkernels.install()`. The two runs must give the same bytes, and the
second should be faster.
"""
from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import hotkernels

from .check import describe

ROOT = Path(__file__).resolve().parent.parent
WORKLOAD = ROOT / "examples" / "workload.py"
# Alternating runs of each side. The best of each is reported.
ROUNDS = 7
# Nanoseconds in a millisecond.
NS_PER_MS = 1e6


@dataclass(frozen=True)
class Demo:
    """The outcome of the demonstration."""

    # Best time of one run of the program on plain NumPy, in milliseconds.
    numpy_ms: float
    # Best time with hotkernels installed, in milliseconds.
    hotkernels_ms: float
    # Whether both runs produced the same results, bit for bit.
    identical: bool
    # hotkernels' report after the installed runs.
    report: hotkernels.Report

    @property
    def speedup(self) -> float:
        """Return how many times faster the program ran with hotkernels."""
        return self.numpy_ms / self.hotkernels_ms


def load_workload() -> ModuleType:
    """Load the example program from its file, as Python would run it."""
    spec = importlib.util.spec_from_file_location("workload", WORKLOAD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _timed(workload: ModuleType, data: tuple) -> tuple[float, tuple]:
    """Run the program once.

    Args:
        workload: The loaded example program.
        data: Its inputs.

    Returns:
        (milliseconds taken, description of its results).
    """
    start = time.perf_counter_ns()
    results = workload.analyse(*data)
    elapsed = (time.perf_counter_ns() - start) / NS_PER_MS
    return elapsed, describe(tuple(results[name] for name in sorted(results)))


def run() -> Demo:
    """Run the program ROUNDS times each way, alternating, and compare.

    Returns:
        The best time of each side, whether the results agree, and the
        report of the installed runs. hotkernels is left uninstalled.
    """
    workload = load_workload()
    data = workload.make_data()
    plain, installed, identical = [], [], True
    hotkernels.uninstall()
    _timed(workload, data)
    for _ in range(ROUNDS):
        hotkernels.uninstall()
        elapsed, expected = _timed(workload, data)
        plain.append(elapsed)
        hotkernels.install()
        elapsed, got = _timed(workload, data)
        installed.append(elapsed)
        identical = identical and got == expected
    # One more run on fresh counters, so the report shows one run's calls.
    hotkernels.uninstall()
    hotkernels.install()
    _timed(workload, data)
    report = hotkernels.report()
    hotkernels.uninstall()
    return Demo(min(plain), min(installed), identical, report)
