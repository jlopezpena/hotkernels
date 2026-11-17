"""The delivery tool: tests and times hotkernels from Python.

Run with `uv run delivery <command>`.
"""
from __future__ import annotations

import logging
from typing import Optional

import typer

from . import check as C
from .cases import INEXACT_OPERATIONS, OPERATIONS, Case
from .render import render

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)

# Exit codes.
EXIT_OK = 0
EXIT_FAILED = 1

# Failures printed per operation unless --all-failures is given.
SHOWN_FAILURES = 5


@app.callback()
def main() -> None:
    """Configure logging for every command: plain messages on standard error."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


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
    summaries = []
    for summary in C.run(operations, inexact, _keep(only, case)):
        log.info("%s: %d cases, %d failed", summary.operation, summary.cases, len(summary.failures))
        summaries.append(summary)
    failures = sum(len(s.failures) for s in summaries)
    shown = max((len(s.failures) for s in summaries), default=0) if all_failures else SHOWN_FAILURES
    log.info("\n%s", render("check.txt.j2", summaries=summaries, shown=shown,
                            total_cases=sum(s.cases for s in summaries),
                            total_taken=sum(s.taken for s in summaries), total_failures=failures).rstrip("\n"))
    raise typer.Exit(EXIT_OK if failures == 0 else EXIT_FAILED)


if __name__ == "__main__":
    app()
