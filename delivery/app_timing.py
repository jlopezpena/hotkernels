"""Times whole applications with and without hotkernels.

Each application of `delivery.apps`, at each of its sizes, is run on plain NumPy, after
`hotkernels.install()` and after `hotkernels.install(inexact=True)`, in
alternation. The results of the first two must be the same bytes. One more
installed run on fresh counters gives the calls each kernel took and the
calls left to NumPy. See docs/applications-design.md.
"""
from __future__ import annotations

import importlib.metadata
import json
import math
import statistics
import subprocess
import time
import warnings
from dataclasses import asdict, dataclass
from types import ModuleType

import hotkernels

from . import apps
from .check import describe

# Alternating rounds of each side. The best of each is reported.
ROUNDS = 7
# Shortest round, in milliseconds. An application whose run is shorter is run
# as many times in a round as it takes to last this long, and the time of one
# run is the round's divided by that number, so a short run is not lost in the
# resolution of the clock and the noise of the machine.
SHORTEST_ROUND_MS = 100.0
# Nanoseconds in a millisecond.
NS_PER_MS = 1e6
# The size each application ran at when it had one, which records written
# then do not state: the first record, delivery/results/apple-m1-pro-apps.json.
SINGLE_SIZE = {"mlp_train": "large", "min_char_rnn": "small", "trimesh_mesh": "large", "sklearn_metrics": "large",
               "empyrical_stats": "small"}
# Runs of `gh run list` looked at for unfinished ones: more than CI starts for one push.
RECENT_RUNS = 20
# The status `gh run list` gives a run that has finished, whatever its outcome.
FINISHED = "completed"
# Seconds to wait for `gh` to answer.
GH_TIMEOUT_SECONDS = 30
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
    # The size it ran at, one of `apps.SIZE_NAMES`.
    size: str
    # One line saying what it does.
    title: str
    # Where its code comes from.
    source: str
    # One line saying how large its data is at that size.
    data: str
    # Runs of the application in each timed round.
    repeats: int
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


def timed(app: ModuleType, data: tuple, repeats: int = 1) -> tuple[float, tuple]:
    """Run an application one or more times.

    Args:
        app: The application's module.
        data: Its inputs.
        repeats: How many times to run it. Each run starts from the same
            inputs, and the results of the last one are described.

    Returns:
        (milliseconds taken by one run, the mean over the repeats;
        description of the results).
    """
    start = time.perf_counter_ns()
    for _ in range(repeats):
        results = app.run(*data)
    elapsed = (time.perf_counter_ns() - start) / NS_PER_MS / repeats
    return elapsed, describe(results)


def repeats_for(milliseconds: float) -> int:
    """Say how many runs make a round of at least SHORTEST_ROUND_MS.

    Args:
        milliseconds: The time of one run, in milliseconds.

    Returns:
        The number of runs, at least one.
    """
    return max(1, math.ceil(SHORTEST_ROUND_MS / max(milliseconds, 1e-3)))


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


def measure(name: str, size: str) -> AppTiming:
    """Time one application at one size every way and compare its results.

    Args:
        name: One of `apps.NAMES`. Its libraries must be installed.
        size: One of `apps.SIZE_NAMES`.

    Returns:
        The times, whether the results agree, and the counts. hotkernels is
        left uninstalled.
    """
    app = apps.load(name)
    data = app.make_data(size)
    plain, installed, inexact = [], [], []
    identical = inexact_identical = True
    with warnings.catch_warnings():
        # The libraries warn about their own arithmetic on the generated
        # data, the same way on each side.
        warnings.simplefilter("ignore")
        hotkernels.uninstall()
        # The first run warms the caches and imports; its time sets the repeats.
        repeats = repeats_for(timed(app, data)[0])
        for _ in range(ROUNDS):
            hotkernels.uninstall()
            elapsed, expected = timed(app, data, repeats)
            plain.append(elapsed)
            hotkernels.install()
            elapsed, got = timed(app, data, repeats)
            installed.append(elapsed)
            identical = identical and got == expected
            hotkernels.uninstall()
            hotkernels.install(inexact=True)
            elapsed, got = timed(app, data, repeats)
            inexact.append(elapsed)
            inexact_identical = inexact_identical and got == expected
        # One more run on fresh counters, so the counts are one run's.
        hotkernels.uninstall()
        hotkernels.install()
        timed(app, data)
        taken, left = counts(hotkernels.report())
        hotkernels.uninstall()
    return AppTiming(
        name=name, size=size, title=app.TITLE, source=app.SOURCE, data=app.SIZES[size], repeats=repeats,
        numpy_ms=min(plain), hotkernels_ms=min(installed), inexact_ms=min(inexact),
        numpy_median_ms=statistics.median(plain), hotkernels_median_ms=statistics.median(installed),
        identical=identical, inexact_identical=inexact_identical, taken=taken, left=left,
    )


def unfinished_runs() -> int | None:
    """Count the repository's GitHub Actions runs that have not finished.

    This machine hosts the CI runners, so an unfinished run is load on it.

    Returns:
        The number among the last RECENT_RUNS that are queued, waiting or in
        progress, or None if `gh` is missing or could not answer.
    """
    try:
        result = subprocess.run(["gh", "run", "list", "--limit", str(RECENT_RUNS), "--json", "status"],
                                capture_output=True, text=True, timeout=GH_TIMEOUT_SECONDS)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return sum(1 for run in json.loads(result.stdout) if run["status"] != FINISHED)


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
    """Rebuild a timing from its JSON record.

    Args:
        row: One application's timing as `as_row` wrote it, or as a record
            written before applications had two sizes: there "size" held the
            line now in "data", the size is the one in SINGLE_SIZE, and every
            round was one run.

    Returns:
        The timing.
    """
    fields = dict(row)
    if "data" not in fields:
        fields["data"] = fields["size"]
        fields["size"] = SINGLE_SIZE[fields["name"]]
        fields["repeats"] = 1
    fields["taken"] = tuple(Count(**count) for count in row["taken"])
    fields["left"] = tuple(Count(**count) for count in row["left"])
    return AppTiming(**fields)
