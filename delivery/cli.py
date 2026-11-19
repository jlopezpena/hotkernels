"""The delivery tool: tests and times hotkernels from Python.

Run with `uv run delivery <command>`.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

import hotkernels

from . import check as C
from . import demo as D
from . import history as H
from . import numpy_suite
from . import timing as T
from . import workload_timing as W
from . import workloads as WL
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
BUSY_LOAD = H.BUSY_LOAD
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


@app.command()
def demo() -> None:
    """Run examples/workload.py, an unmodified NumPy program, with and without hotkernels."""
    outcome = D.run()
    log.info("%s", render("demo.txt.j2", demo=outcome).rstrip("\n"))
    raise typer.Exit(EXIT_OK if outcome.identical and outcome.speedup > 1 else EXIT_FAILED)


def _workload_cases(only: Optional[str], sizes: tuple[str, ...]) -> tuple[list, list, list]:
    """Choose which workloads run, at which sizes.

    Args:
        only: Name of the one workload to run, or None for all.
        sizes: Names of the sizes to run.

    Returns:
        (workloads chosen, (workload, size) pairs to run, (name, missing
        libraries) for each chosen workload that cannot run here).
    """
    chosen = [w for w in WL.load() if only in (None, w.name)]
    runnable = [w for w in chosen if not w.missing()]
    skipped = [(w.name, w.missing()) for w in chosen if w.missing()]
    cases = [(w, name) for w in runnable for name in sizes if name in w.sizes]
    return runnable, cases, skipped


def _sizes(text: Optional[str]) -> tuple[str, ...]:
    """Read the --size option.

    Args:
        text: Size names separated by commas, such as "M,L", or None.

    Returns:
        The names, in the order of SIZE_NAMES; every size but the largest if None.

    Raises:
        typer.BadParameter: If a name is not a size.
    """
    if text is None:
        return WL.SIZE_NAMES[:-1]
    names = {name.strip() for name in text.split(",")}
    if not names <= set(WL.SIZE_NAMES):
        raise typer.BadParameter(f"sizes are {', '.join(WL.SIZE_NAMES)}, separated by commas", param_hint="--size")
    return tuple(name for name in WL.SIZE_NAMES if name in names)


def _write_history(directory: Path, meta: dict, rows: list[dict], sizes: tuple[str, ...], source: str,
                   started: datetime) -> Path:
    """Write one run's history file.

    Args:
        directory: The directory given to --history.
        meta: The run's description, taken when it started.
        rows: The run's rows, as `workload_timing.as_rows` makes them.
        sizes: The size names the run covered.
        source: What started the run, one of history.SOURCES.
        started: When the run started, in UTC.

    Returns:
        The file written.
    """
    full = H.run_meta(meta, list(sizes), source, started, hotkernels.__version__)
    full["load_average_end"] = T.load_average()
    path = H.run_path(directory, _slug(meta["cpu"]) or "unknown-machine", started, full["commit"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": full, "rows": rows}, indent=2) + "\n")
    return path


@app.command()
def workloads(
    only: Optional[str] = typer.Option(None, help="Run one workload, for example `azimint_naive`."),
    size: Optional[str] = typer.Option(None, help="Sizes to run, separated by commas: S, M, L. Default: S,M."),
    record: bool = typer.Option(False, help="Write the results to delivery/results/."),
    history: Optional[Path] = typer.Option(
        None, help="Also write the run to DIR/history/<machine>/, for the data branch. Refused from a dirty tree."),
    source: str = typer.Option(H.MANUAL, help="What started the run, as the history records it: "
                               + ", ".join(H.SOURCES) + "."),
) -> None:
    """Run code not written for hotkernels with and without it: results, times and counts.

    Needs the libraries of the `workloads` dependency group: `uv run --group
    workloads delivery workloads`. Times are only meaningful on native
    hardware and an idle machine.

    Args:
        only: Workload to restrict the run to.
        size: Sizes to run, separated by commas.
        record: Whether to write a JSON file and a Markdown report.
        history: Directory to write a history file under, or None.
        source: What started the run: nightly, manual or backfill.
    """
    sizes = _sizes(size)
    if source not in H.SOURCES:
        raise typer.BadParameter(f"source must be one of {', '.join(H.SOURCES)}", param_hint="--source")
    # A history file names the commit it timed; with changes in tracked files it would not say what ran.
    if history is not None and H.is_dirty():
        raise typer.BadParameter("tracked files differ from the commit; commit or stash them first",
                                 param_hint="--history")
    started = H.now_utc()
    meta = W.describe_run()
    chosen, cases, skipped = _workload_cases(only, sizes)
    outcomes = W.run(cases)
    failed = [f"{o.workload} ({o.size})" for o in outcomes if not o.identical]
    context = {"meta": meta, "outcomes": outcomes, "skipped": skipped, "failed": failed,
               "workloads": chosen, "busy_load": BUSY_LOAD}
    log.info("%s", render("workloads.txt.j2", **context).rstrip("\n"))
    if record:
        RESULTS.mkdir(exist_ok=True)
        stem = f"{_slug(meta['cpu']) or 'unknown-machine'}-workloads"
        (RESULTS / f"{stem}.json").write_text(json.dumps({"meta": meta, "rows": W.as_rows(outcomes)}, indent=2) + "\n")
        (RESULTS / f"{stem}.md").write_text(render("workloads.md.j2", **context))
        log.info("recorded in %s", RESULTS / f"{stem}.md")
    if history is not None:
        log.info("history in %s", _write_history(history, meta, W.as_rows(outcomes), sizes, source, started))
    raise typer.Exit(EXIT_OK if not failed else EXIT_FAILED)


def _history_lines(series: H.Series) -> list[dict]:
    """Turn a series into one line per run for the history report.

    Args:
        series: The series.

    Returns:
        Per run: when it ran, the commit, the versions, the times where the
        series is one workload's, the speedup, and the marks that apply.
    """
    lines = []
    for group in series.groups:
        for index, (run, speedup) in enumerate(group.points):
            row = run.row(series.workload, series.size) if series.workload else None
            marks = [mark for mark, applies in (("break", group.break_before and index == 0), ("busy", run.busy),
                                                (run.meta["source"], run.meta["source"] != H.NIGHTLY)) if applies]
            lines.append({"recorded_at": run.recorded_at.strftime("%Y-%m-%d %H:%M"), "commit": run.commit[:H.SHORT_COMMIT],
                          "numpy": run.meta["numpy"], "numpy_ns": row["numpy_ns"] if row else None,
                          "hotkernels_ns": row["hotkernels_ns"] if row else None, "speedup": speedup, "marks": marks})
    return lines


@app.command(name="history")
def history_report(
    machine: Optional[str] = typer.Option(None, help="Machine, as its history directory is named. Default: this one."),
    workload: Optional[str] = typer.Option(None, help="One workload. Default: the headline and every workload."),
    size: str = typer.Option(H.HEADLINE_SIZE, help="The size: S, M or L."),
    ref: str = typer.Option(H.DEFAULT_REF, help="The branch the runs are read from."),
) -> None:
    """Print the recorded workload runs of one machine, in commit order.

    Reads the data branch through git; run `git fetch origin perf-history` first.

    Args:
        machine: The machine's slug, or None for the one this runs on.
        workload: The one workload to show, or None for the headline and all of them.
        size: The size to show.
        ref: The branch the runs are read from.
    """
    try:
        runs = H.load_runs(ref)
    except H.HistoryError as error:
        log.error("%s", error)
        raise typer.Exit(EXIT_FAILED)
    machine = machine or _slug(T.machine())
    names = [workload] if workload else list(dict.fromkeys(
        row["workload"] for run in runs if run.machine == machine for row in run.rows))
    series = [] if workload else [H.headline_series(runs, machine, size)]
    series += [H.workload_series(runs, machine, name, size) for name in names]
    blocks = [{"workload": s.workload, "lines": _history_lines(s)} for s in series]
    log.info("%s", render("history.txt.j2", machine=machine, size=size, ref=ref, blocks=blocks,
                          machines=H.machines(runs), busy_load=BUSY_LOAD).rstrip("\n"))


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
