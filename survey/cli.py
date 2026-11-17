"""The survey: measures NumPy against naive C, one operation at a time.

Run with `uv run survey <command>`. Results are for ranking candidates; the
judge is what decides whether a kernel ships.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import typer

from .candidates import candidates
from .measure import (BUDGET_S, QUICK_BUDGET_S, QUICK_SAMPLES, QUICK_SIZES, SAMPLES, SIZES, build,
                      describe_machine, measure, reference_ns)
from .render import NOISY_DRIFT, render_progress, render_report

log = logging.getLogger(__name__)

# Recorded results, committed to the repository.
RESULTS = Path(__file__).resolve().parent / "results"

EXIT_OK = 0
# Some naive C kernel disagreed with NumPy, so its timings mean nothing.
EXIT_DISAGREED = 1

# Anything that is not a lowercase letter or digit becomes a dash in a file name.
_SLUG_SEPARATORS = re.compile(r"[^a-z0-9]+")
_JSON_INDENT = 1

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)


@app.callback()
def configure() -> None:
    """Set up logging. Progress and reports go to standard output as plain lines."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)


def _write_report(result_file: Path) -> Path:
    """Render the Markdown report that belongs to a result file.

    Args:
        result_file: A JSON file written by `survey run`.

    Returns:
        The path of the report, which sits beside the result file.
    """
    report = result_file.with_suffix(".md")
    report.write_text(render_report(json.loads(result_file.read_text())))
    return report


@app.command()
def run(
    only: Optional[str] = typer.Option(None, "--only", help="Measure only candidates whose name matches this regex."),
    quick: bool = typer.Option(False, "--quick", help="Fewer samples and no 16M size, as a smoke test."),
    out: Optional[str] = typer.Option(None, "--out", help="Result file name without extension. Default: the CPU name."),
) -> None:
    """Measure the candidates and write a result file and its report.

    Args:
        only: Regex selecting candidates by name; all of them when omitted.
        quick: Use the quick sizes and sample counts.
        out: Stem of the result file under `survey/results/`.
    """
    lib = build()
    sizes = QUICK_SIZES if quick else SIZES
    samples, budget_s = (QUICK_SAMPLES, QUICK_BUDGET_S) if quick else (SAMPLES, BUDGET_S)
    chosen = [c for c in candidates() if not only or re.search(only, c.key)]

    before = reference_ns(lib)
    rows = []
    for cand in chosen:
        for n in sizes:
            rows.append(measure(lib, cand, n, samples, budget_s))
            log.info(render_progress(rows[-1]))
    drift = abs(reference_ns(lib) / before - 1)
    if drift > NOISY_DRIFT:
        log.warning("WARNING: the machine was noisy; the reference loop drifted by %.1f%%", drift * 100)

    meta = describe_machine([c.cfn for c in chosen], drift)
    for warning in meta.get("warnings", []):
        log.warning("WARNING: %s", warning)
    stem = out or _SLUG_SEPARATORS.sub("-", meta["cpu"].lower()).strip("-")
    RESULTS.mkdir(exist_ok=True)
    result_file = RESULTS / f"{stem}.json"
    result_file.write_text(json.dumps({"meta": meta, "rows": rows}, indent=_JSON_INDENT))
    log.info("wrote %s and %s", result_file, _write_report(result_file))
    raise typer.Exit(EXIT_OK if all(row["naive_matches_numpy"] for row in rows) else EXIT_DISAGREED)


@app.command()
def report(
    result_file: Path = typer.Argument(..., exists=True, dir_okay=False, help="A JSON file from `survey run`."),
) -> None:
    """Regenerate the Markdown report of an existing result file.

    Args:
        result_file: The JSON file to render; the report is written beside it.
    """
    log.info("wrote %s", _write_report(result_file))
