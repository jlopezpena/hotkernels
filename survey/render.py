"""Renders survey results from Jinja templates.

Python prepares the numbers; `survey/templates/` decides the layout.
"""
from __future__ import annotations

import math
from collections import defaultdict
from operator import itemgetter
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

TEMPLATES = Path(__file__).resolve().parent / "templates"
REPORT_TEMPLATE = "report.md.j2"
PROGRESS_TEMPLATE = "progress.txt.j2"

# Sizes that get a column in the report.
SHOWN_SIZES = [16, 4096, 65536, 1048576, 16777216]
# The ranking uses sizes from here up, where per-call overhead no longer matters.
LARGE_N = 65536
# The size at which time per element is reported.
PER_ELEMENT_N = 1048576
# The size that shows NumPy's fixed cost per call.
FIXED_COST_N = 1

# A run whose reference loop drifts by more than this is marked as noisy.
NOISY_DRIFT = 0.03

# Ratios are shown with fewer decimals as they grow: 1.63x, 11.0x, 622x.
_TWO_DECIMALS_BELOW = 10
_ONE_DECIMAL_BELOW = 100


def format_ratio(x: float) -> str:
    """Format a ratio of two times.

    Args:
        x: The ratio.

    Returns:
        The ratio with a precision that suits its size, for example `1.63x`.
    """
    if x >= _ONE_DECIMAL_BELOW:
        return f"{x:.0f}x"
    if x >= _TWO_DECIMALS_BELOW:
        return f"{x:.1f}x"
    return f"{x:.2f}x"


def _environment() -> Environment:
    """Build the Jinja environment and register the formatting filters.

    Returns:
        An environment that loads from `survey/templates/` and fails on any
        undefined name.
    """
    env = Environment(
        loader=FileSystemLoader(TEMPLATES),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        undefined=StrictUndefined,
    )
    env.filters["ratio"] = format_ratio
    return env


def _geometric_mean(values: list[float]) -> float:
    """Compute a geometric mean.

    Args:
        values: Positive numbers; at least one.

    Returns:
        Their geometric mean.
    """
    return math.exp(sum(math.log(v) for v in values) / len(values))


def _candidate_summary(key: str, by_size: dict[int, dict], vectorised: dict[str, list[str]]) -> dict:
    """Summarise one candidate across sizes for the report table.

    Args:
        key: The candidate's name.
        by_size: Its result rows, keyed by element count.
        vectorised: Kernel name -> vectorised loops, from the run's metadata.

    Returns:
        The values of one table row. Sizes that were not measured are None.
    """
    first = next(iter(by_size.values()))
    per_element = by_size.get(PER_ELEMENT_N)
    fixed = by_size.get(FIXED_COST_N)
    return {
        "key": key,
        "stmt": first["stmt"],
        "alloc": first["alloc"],
        "widths": [loop.split("width ")[1] for loop in vectorised[first["cfn"]]],
        "gap": _geometric_mean([r["numpy_over_c"] for n, r in by_size.items() if n >= LARGE_N]),
        "ratios": [by_size[n]["numpy_over_c"] if n in by_size else None for n in SHOWN_SIZES],
        "c_ns_per_element": per_element["c_ns"] / per_element["work"] if per_element else None,
        "numpy_ns_per_element": per_element["numpy_ns"] / per_element["work"] if per_element else None,
        "numpy_fixed_ns": fixed["numpy_ns"] if fixed else None,
    }


def render_report(data: dict) -> str:
    """Render a result file as a Markdown report.

    Args:
        data: A result file as written by `survey run`: `meta` and `rows`.

    Returns:
        The report, with candidates ranked by how far NumPy is behind naive C
        at large sizes.
    """
    by_key: dict[str, dict[int, dict]] = defaultdict(dict)
    for row in data["rows"]:
        by_key[row["key"]][row["n"]] = row
    summaries = [_candidate_summary(key, sizes, data["meta"]["c_vectorised"]) for key, sizes in by_key.items()]
    summaries.sort(key=itemgetter("gap"), reverse=True)
    disagreeing = sorted({row["key"] for row in data["rows"] if not row["naive_matches_numpy"]})
    return _environment().get_template(REPORT_TEMPLATE).render(
        meta=data["meta"], candidates=summaries, sizes=SHOWN_SIZES, disagreeing=disagreeing,
        large_n=LARGE_N, per_element_n=PER_ELEMENT_N, fixed_cost_n=FIXED_COST_N,
        noisy_drift=NOISY_DRIFT,
    )


def render_progress(row: dict) -> str:
    """Render the one-line progress report for a finished measurement.

    Args:
        row: A result row from `measure`.

    Returns:
        The line, without a trailing newline.
    """
    return _environment().get_template(PROGRESS_TEMPLATE).render(row=row).rstrip("\n")
