"""The delivery tool: tests and times hotkernels from Python.

Run with `uv run delivery <command>`.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

import typer

import hotkernels

from . import check as C
from . import numpy_suite
from . import timing as T
from .cases import INEXACT_OPERATIONS, OPERATIONS, Case
from .render import render

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)

# Exit codes.
EXIT_OK = 0
EXIT_FAILED = 1

# Failures printed per operation unless --all-failures is given.
SHOWN_FAILURES = 5
# Recorded timings, committed to the repository: one JSON file and one report per machine.
RESULTS = Path(__file__).resolve().parent / "results"
# Load average above which a timing run is marked as taken on a busy machine.
BUSY_LOAD = 1.0
# A kernel-taking call counts as slower than NumPy below this speedup. The
# margin is the noise of a best-of-several timing.
SLOWER_THAN = 0.97


@app.callback()
def main() -> None:
    """Configure logging for every command: plain messages on standard error."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # The commands install and uninstall hotkernels many times over; its own
    # message for each would bury their output.
    logging.getLogger(hotkernels.__name__).setLevel(logging.WARNING)


def _keep(only: Optional[str], case_text: Optional[str]):
    """Build the filter that decides which cases run.

    Args:
        only: Keep only cases of the operation with this name, or all if None.
        case_text: Keep only cases whose label contains this text, or all if None.

    Returns:
        A function from a case to whether it runs.
    """
    def keep(case: Case) -> bool:
        """Say whether one case passes both filters."""
        return (only is None or case.operation == only) and (case_text is None or case_text in case.label)
    return keep


@app.command()
def check(
    only: Optional[str] = typer.Option(None, help="Run one operation, for example `clip`."),
    case: Optional[str] = typer.Option(None, help="Run only cases whose label contains this text."),
    all_failures: bool = typer.Option(False, help="Print every failure, not the first few per operation."),
) -> None:
    """Run every hooked NumPy call with and without hotkernels and compare them.

    Args:
        only: Operation to restrict the run to.
        case: Text a case's label must contain.
        all_failures: Whether to print every failure.
    """
    operations = {name: make for name, make in OPERATIONS.items() if only in (None, name)}
    inexact = {name: make for name, make in INEXACT_OPERATIONS.items() if only in (None, name)}
    summaries = [] if only or case else [C.installation_checks()]
    for summary in C.run(operations, inexact, _keep(only, case)):
        log.info("%s: %d cases, %d failed", summary.operation, summary.cases, len(summary.failures))
        summaries.append(summary)
    failures = sum(len(s.failures) for s in summaries)
    shown = max((len(s.failures) for s in summaries), default=0) if all_failures else SHOWN_FAILURES
    log.info("\n%s", render("check.txt.j2", summaries=summaries, shown=shown,
                            total_cases=sum(s.cases for s in summaries),
                            total_taken=sum(s.taken for s in summaries), total_failures=failures).rstrip("\n"))
    raise typer.Exit(EXIT_OK if failures == 0 else EXIT_FAILED)


def _slug(text: str) -> str:
    """Turn a processor name into a file name: lower case, words joined by hyphens."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _time_context(meta: dict, measurements: list) -> dict:
    """Build the variables both timing templates take.

    Args:
        meta: What the run was measured on.
        measurements: Every measurement of the run.

    Returns:
        The measurements split into calls a kernel took and calls left to
        NumPy, the kernel-taking calls that came out slower than NumPy, and
        the metadata.
    """
    taken = [m for m in measurements if m.kernel]
    return {
        "meta": meta,
        "taken": taken,
        "left": [m for m in measurements if not m.kernel],
        "slower": [m for m in taken if m.speedup < SLOWER_THAN],
        "busy_load": BUSY_LOAD,
        "rounds": T.ROUNDS,
    }


@app.command()
def time(
    only: Optional[str] = typer.Option(None, help="Time one operation, for example `clip`."),
    record: bool = typer.Option(False, help="Write the results to delivery/results/."),
) -> None:
    """Time NumPy statements from Python, with and without hotkernels.

    Only meaningful on native hardware and an idle machine.

    Args:
        only: Operation to restrict the run to.
        record: Whether to write a JSON file and a Markdown report.
    """
    meta = T.describe_machine()
    entries = [entry for entry in (*T.TAKEN, *T.LEFT) if only in (None, entry.operation)]
    measurements = T.run(entries)
    context = _time_context(meta, measurements)
    log.info("%s", render("time.txt.j2", **context).rstrip("\n"))
    if record:
        RESULTS.mkdir(exist_ok=True)
        stem = _slug(meta["cpu"]) or "unknown-machine"
        (RESULTS / f"{stem}.json").write_text(json.dumps({"meta": meta, "rows": T.as_rows(measurements)}, indent=2) + "\n")
        (RESULTS / f"{stem}.md").write_text(render("time.md.j2", **context))
        log.info("recorded in %s", RESULTS / f"{stem}.md")


@app.command(name="numpy-tests")
def numpy_tests(
    select: Optional[str] = typer.Option(None, help="A pytest -k expression, to run part of the suite."),
) -> None:
    """Run NumPy's own tests with and without hotkernels and compare every outcome.

    Takes several minutes. Only the exact kernels are installed.

    Args:
        select: A pytest `-k` expression restricting the run.
    """
    comparison = numpy_suite.compare(select)
    log.info("%s", render("numpy_tests.txt.j2", comparison=comparison).rstrip("\n"))
    raise typer.Exit(EXIT_OK if not comparison.differences else EXIT_FAILED)


if __name__ == "__main__":
    app()
