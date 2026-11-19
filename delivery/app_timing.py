"""Times whole applications with and without hotkernels.

Each application of `delivery.apps` is run on plain NumPy, after
`hotkernels.install()` and after `hotkernels.install(inexact=True)`, in
alternation. The results of the first two must be the same bytes. One more
installed run on fresh counters gives the calls each kernel took and the
calls left to NumPy. See docs/applications-design.md.
"""
from __future__ import annotations

import importlib.metadata
import statistics
import time
import warnings
from dataclasses import asdict, dataclass
from types import ModuleType

import hotkernels

from . import apps
from .check import describe

# Alternating runs of each side. The best of each is reported.
ROUNDS = 7
# Nanoseconds in a millisecond.
NS_PER_MS = 1e6
# Distributions whose versions a recorded run states, where installed.
LIBRARIES = ("trimesh", "scikit-learn", "empyrical-reloaded", "pandas", "scipy")


@dataclass(frozen=True)
class Count:
    """How often one hooked NumPy call was made in one run of an application."""

    # The NumPy call, as a user writes it.
    call: str
    # Element type the kernel serves, or an empty string for a count of calls left to NumPy.
    elements: str
    # Number of calls.
    calls: int


@dataclass(frozen=True)
class AppTiming:
    """One application's times, results and counts."""

    # The application's module name.
    name: str
    # One line saying what it does.
    title: str
    # Where its code comes from.
    source: str
    # One line saying how large its data is.
    size: str
    # Best time of one run on plain NumPy, in milliseconds.
    numpy_ms: float
    # Best time after install(), in milliseconds.
    hotkernels_ms: float
    # Best time after install(inexact=True), in milliseconds.
    inexact_ms: float
    # Median time on plain NumPy, in milliseconds: with the best, a measure of the noise.
    numpy_median_ms: float
    # Median time after install(), in milliseconds.
    hotkernels_median_ms: float
    # Whether every installed run gave the same bytes as the plain run before it.
    identical: bool
    # Whether every install(inexact=True) run gave the same bytes as the plain run.
    inexact_identical: bool
    # Calls taken by kernels in one run after install().
    taken: tuple[Count, ...]
    # Calls that reached a hook and went to NumPy in one run after install().
    left: tuple[Count, ...]

    @property
    def speedup(self) -> float:
        """Return how many times faster the application ran after install()."""
        return self.numpy_ms / self.hotkernels_ms

    @property
    def inexact_speedup(self) -> float:
        """Return how many times faster it ran after install(inexact=True)."""
        return self.numpy_ms / self.inexact_ms

    @property
    def kernel_calls(self) -> int:
        """Return the calls kernels took in one run."""
        return sum(count.calls for count in self.taken)

    @property
    def left_calls(self) -> int:
        """Return the calls left to NumPy in one run."""
        return sum(count.calls for count in self.left)


def timed(app: ModuleType, data: tuple) -> tuple[float, tuple]:
    """Run an application once.

    Args:
        app: The application's module.
        data: Its inputs.

    Returns:
        (milliseconds taken, description of its results).
    """
    start = time.perf_counter_ns()
    results = app.run(*data)
    elapsed = (time.perf_counter_ns() - start) / NS_PER_MS
    return elapsed, describe(results)


def counts(report: hotkernels.Report) -> tuple[tuple[Count, ...], tuple[Count, ...]]:
    """Read a report's counters.

    Args:
        report: hotkernels' report after one run.

    Returns:
        (calls taken, per kernel; calls left to NumPy, per NumPy call),
        leaving out the zeros.
    """
    taken = tuple(Count(k.numpy_call, k.elements, k.calls) for k in report.kernels if k.calls)
    left = tuple(Count(k.numpy_call, "", k.left_to_numpy) for k in report.kernels if k.left_to_numpy)
    return taken, left


def measure(name: str) -> AppTiming:
    """Time one application every way and compare its results.

    Args:
        name: One of `apps.NAMES`. Its libraries must be installed.

    Returns:
        The times, whether the results agree, and the counts. hotkernels is
        left uninstalled.
    """
    app = apps.load(name)
    data = app.make_data()
    plain, installed, inexact = [], [], []
    identical = inexact_identical = True
    with warnings.catch_warnings():
        # The libraries warn about their own arithmetic on the generated
        # data, the same way on each side.
        warnings.simplefilter("ignore")
        hotkernels.uninstall()
        timed(app, data)
        for _ in range(ROUNDS):
            hotkernels.uninstall()
            elapsed, expected = timed(app, data)
            plain.append(elapsed)
            hotkernels.install()
            elapsed, got = timed(app, data)
            installed.append(elapsed)
            identical = identical and got == expected
            hotkernels.uninstall()
            hotkernels.install(inexact=True)
            elapsed, got = timed(app, data)
            inexact.append(elapsed)
            inexact_identical = inexact_identical and got == expected
        # One more run on fresh counters, so the counts are one run's.
        hotkernels.uninstall()
        hotkernels.install()
        timed(app, data)
        taken, left = counts(hotkernels.report())
        hotkernels.uninstall()
    return AppTiming(
        name=name, title=app.TITLE, source=app.SOURCE, size=app.SIZE,
        numpy_ms=min(plain), hotkernels_ms=min(installed), inexact_ms=min(inexact),
        numpy_median_ms=statistics.median(plain), hotkernels_median_ms=statistics.median(installed),
        identical=identical, inexact_identical=inexact_identical, taken=taken, left=left,
    )


def library_versions() -> dict[str, str]:
    """Return the version of each of LIBRARIES that is installed."""
    versions = {}
    for library in LIBRARIES:
        try:
            versions[library] = importlib.metadata.version(library)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def as_row(timing: AppTiming) -> dict:
    """Turn a timing into what the JSON record holds."""
    return asdict(timing)


def from_row(row: dict) -> AppTiming:
    """Rebuild a timing from its JSON record."""
    fields = dict(row)
    fields["taken"] = tuple(Count(**count) for count in row["taken"])
    fields["left"] = tuple(Count(**count) for count in row["left"])
    return AppTiming(**fields)
