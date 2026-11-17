"""The documentation site's command line.

Run with `uv run docsite <command>`.
"""
from __future__ import annotations

import logging
from pathlib import Path

import typer

from .build import DEFAULT_OUT, build
from .serve import serve

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Build and preview the hotkernels documentation site.")

# Port `serve` listens on unless told otherwise.
DEFAULT_PORT = 8000


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Also log every request.")) -> None:
    """Configure logging for every command.

    Args:
        verbose: Log at debug level, which includes the server's request log.
    """
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(message)s")


@app.command("build")
def build_command(
    out: Path = typer.Option(DEFAULT_OUT, "--out", help="Directory to build into. It is emptied first."),
) -> None:
    """Build the site into a directory of static files.

    Args:
        out: Directory to build into.
    """
    build(out)


@app.command("serve")
def serve_command(
    out: Path = typer.Option(DEFAULT_OUT, "--out", help="Directory to build into and serve from."),
    port: int = typer.Option(DEFAULT_PORT, "--port", help="Port to listen on."),
) -> None:
    """Build the site and serve it locally, rebuilding when a source changes.

    Args:
        out: Directory to build into and serve from.
        port: TCP port to listen on.
    """
    serve(out, port)
