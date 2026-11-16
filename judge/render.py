"""Renders all generated text from Jinja templates.

Reports, generated C source and Markdown all live in `judge/templates/`. Python
code passes numbers and structures; the templates decide the layout.
"""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

TEMPLATES = Path(__file__).resolve().parent / "templates"

NS_PER_US = 1_000
NS_PER_MS = 1_000_000

# Speedups are shown with fewer decimals as they grow: 1.63x, 11.0x, 622x.
_TWO_DECIMALS_BELOW = 10
_ONE_DECIMAL_BELOW = 100


def format_ns(ns: float) -> str:
    """Format a duration in nanoseconds using the largest unit that keeps it readable."""
    if ns < NS_PER_US:
        return f"{ns:.1f} ns"
    if ns < NS_PER_MS:
        return f"{ns / NS_PER_US:.1f} us"
    return f"{ns / NS_PER_MS:.2f} ms"


def format_speedup(stats: dict) -> str:
    """Format the point estimate of a speedup, for example `1.63x`."""
    x = stats["speedup"]
    if x >= _ONE_DECIMAL_BELOW:
        return f"{x:.0f}x"
    if x >= _TWO_DECIMALS_BELOW:
        return f"{x:.1f}x"
    return f"{x:.2f}x"


def format_interval(stats: dict) -> str:
    """Format the 95% confidence interval of a speedup, for example `[1.60 – 1.66]`."""
    if stats["speedup"] >= _ONE_DECIMAL_BELOW:
        return f"[{stats['lo']:.0f} – {stats['hi']:.0f}]"
    return f"[{stats['lo']:.2f} – {stats['hi']:.2f}]"


def format_speedup_cell(stats: dict) -> str:
    """Format a full table cell: speedup, interval, and `n.s.` when not significant."""
    mark = "" if stats["significant"] else " n.s."
    return f"{format_speedup(stats):>7}  {format_interval(stats)}{mark}"


def _environment() -> Environment:
    """Build the Jinja environment and register the formatting filters."""
    env = Environment(
        loader=FileSystemLoader(TEMPLATES),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        undefined=StrictUndefined,
    )
    env.filters.update(
        thousands=lambda n: f"{n:,}",
        ns=format_ns,
        pct=lambda fraction: f"{fraction:.1%}",
        speedup_cell=format_speedup_cell,
        rjust=lambda text, width: str(text).rjust(width),
        ljust=lambda text, width: str(text).ljust(width),
    )
    return env


_ENV = _environment()


def render(template: str, **context) -> str:
    """Render the named template from `judge/templates/` with the given variables."""
    return _ENV.get_template(template).render(**context)
