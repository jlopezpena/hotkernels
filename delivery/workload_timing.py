"""Runs the workloads with and without hotkernels: results, times and counts.

For each workload and size: one run each way to check that the results are
the same, then alternating timed runs, then one installed run on fresh
counters to read hotkernels' report. A workload whose results differ is not
timed.
"""
from __future__ import annotations

import importlib.metadata
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Iterable

import hotkernels

from .check import describe
from .timing import describe_machine
from .workloads import Workload

# Time each side of a measurement aims for, in nanoseconds: one second.
TARGET_SIDE_NS = 1_000_000_000
# Limits on the number of alternating runs of each side. The best of each
# side is reported, so even a slow workload gets a few.
MIN_ROUNDS = 3
MAX_ROUNDS = 200
# Libraries whose versions are recorded with a run, as installed under these names.
LIBRARIES = ("numpy", "pandas", "scikit-image", "scikit-learn", "scipy")


@dataclass(frozen=True)
class Count:
    """How often the workload reached one hooked call in a single installed run."""

    # The NumPy call, as report() names it.
    numpy_call: str
    # The element type the kernel serves.
    elements: str
    # Calls the kernel took.
    calls: int
    # Calls that reached the hook and went to NumPy.
    left_to_numpy: int


@dataclass(frozen=True)
class Outcome:
    """The result of one workload at one size."""

    # The workload's name.
    workload: str
    # Its title.
    title: str
    # Whether this project wrote it.
    ours: bool
    # The size's name: S, M or L.
    size: str
    # Whether the results with hotkernels were the same as without, bit for bit.
    identical: bool
    # Best time of one run on plain NumPy, in nanoseconds. 0 if not timed.
    numpy_ns: float = 0.0
    # Best time of one run with hotkernels installed, in nanoseconds. 0 if not timed.
    hotkernels_ns: float = 0.0
    # Alternating runs of each side that the times are the best of.
    rounds: int = 0
    # Hooked calls the workload reached, with a count on either side.
    counts: tuple[Count, ...] = field(default=())

    @property
    def speedup(self) -> float:
        """Return how many times faster the workload ran with hotkernels, or 0 if not timed."""
        return self.numpy_ns / self.hotkernels_ns if self.hotkernels_ns else 0.0


def _run_once(workload: Workload, data: tuple) -> tuple[int, tuple]:
    """Run a workload once and describe its results.

    Args:
        workload: The workload.
        data: Its inputs.

    Returns:
        (nanoseconds taken, description of the results).
    """
    start = time.perf_counter_ns()
    results = workload.run(*data)
    elapsed = time.perf_counter_ns() - start
    return elapsed, describe(results)


def rounds_for(first_ns: int) -> int:
    """Choose how many alternating runs each side gets.

    Args:
        first_ns: Time of one run on plain NumPy, in nanoseconds.

    Returns:
        Enough runs for about TARGET_SIDE_NS per side, within MIN_ROUNDS and MAX_ROUNDS.
    """
    return max(MIN_ROUNDS, min(MAX_ROUNDS, math.ceil(TARGET_SIDE_NS / max(first_ns, 1))))


def _counts(report: hotkernels.Report) -> tuple[Count, ...]:
    """Keep the report's rows for hooked calls the workload reached.

    Args:
        report: hotkernels' report after one installed run on fresh counters.

    Returns:
        One count per kernel with a call on either side.
    """
    return tuple(Count(k.numpy_call, k.elements, k.calls, k.left_to_numpy or 0)
                 for k in report.kernels if k.calls or k.left_to_numpy)


def measure(workload: Workload, size: str) -> Outcome:
    """Check, time and count one workload at one size.

    Args:
        workload: The workload.
        size: The name of one of its sizes.

    Returns:
        The outcome. hotkernels is left uninstalled.
    """
    data = workload.make_data(**workload.sizes[size])
    hotkernels.uninstall()
    first_ns, expected = _run_once(workload, data)
    hotkernels.install()
    _, got = _run_once(workload, data)
    hotkernels.uninstall()
    if got != expected:
        return Outcome(workload.name, workload.title, workload.ours, size, identical=False)

    rounds = rounds_for(first_ns)
    plain, installed = [], []
    for _ in range(rounds):
        plain.append(_run_once(workload, data)[0])
        hotkernels.install()
        installed.append(_run_once(workload, data)[0])
        hotkernels.uninstall()

    # Installing anew resets the counters, so the report shows one run's calls.
    hotkernels.install()
    workload.run(*data)
    report = hotkernels.report()
    hotkernels.uninstall()
    return Outcome(workload.name, workload.title, workload.ours, size, True,
                   min(plain), min(installed), rounds, _counts(report))


def run(cases: Iterable[tuple[Workload, str]]) -> list[Outcome]:
    """Measure each workload at each size given.

    Args:
        cases: (workload, size name) pairs, in the order to run them.

    Returns:
        One outcome per pair.
    """
    return [measure(workload, size) for workload, size in cases]


def library_versions() -> dict[str, str]:
    """Return the installed version of each library in LIBRARIES, by name."""
    versions = {}
    for name in LIBRARIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def describe_run() -> dict:
    """Record what a run was measured on: the machine and the library versions."""
    return {**describe_machine(), "libraries": library_versions()}


def as_rows(outcomes: list[Outcome]) -> list[dict]:
    """Convert outcomes to plain dictionaries for a JSON file, with the speedup."""
    return [{**asdict(o), "speedup": o.speedup} for o in outcomes]
