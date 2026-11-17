"""Renders the tool's reports through Jinja templates in `delivery/templates/`."""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

TEMPLATES = Path(__file__).resolve().parent / "templates"

# Nanoseconds in a microsecond and in a millisecond.
NS_PER_US = 1e3
NS_PER_MS = 1e6


def format_time(ns: float) -> str:
    """Format a duration in the unit that keeps it readable.

    Args:
        ns: Duration in nanoseconds.

    Returns:
        The duration with three significant digits and its unit.
    """
    if ns < NS_PER_US:
        return f"{ns:.3g} ns"
    if ns < NS_PER_MS:
        return f"{ns / NS_PER_US:.3g} µs"
    return f"{ns / NS_PER_MS:.3g} ms"


def _environment() -> Environment:
    """Build the Jinja environment and register the formatting filters."""
    env = Environment(loader=FileSystemLoader(TEMPLATES), undefined=StrictUndefined,
                      trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)
    env.filters.update(
        thousands=lambda n: f"{n:,}",
        time=format_time,
        times=lambda ratio: f"{ratio:.2f}x",
        signed_ns=lambda ns: f"{ns:+.1f} ns",
        signed_percent=lambda fraction: f"{fraction:+.1%}",
        rjust=lambda text, width: str(text).rjust(width),
        ljust=lambda text, width: str(text).ljust(width),
    )
    return env


_ENV = _environment()


def render(template: str, **context) -> str:
    """Render the named template with the given variables."""
    return _ENV.get_template(template).render(**context)
