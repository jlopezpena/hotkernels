"""`python -m hotkernels`: run a script or a module with hotkernels installed.

    python -m hotkernels [--inexact] [--report] script.py [args ...]
    python -m hotkernels [--inexact] [--report] -m module [args ...]

The script runs as `python script.py` would run it, as `__main__`, with its
own arguments in `sys.argv` and its directory first on `sys.path`, so it needs
no change at all. Its exit status is the command's.

The options are read from `sys.argv` by hand rather than with `typer`, the
repository's rule for command lines: this code ships in the package, and a
command-line library would become a dependency of every installation. After
the options, everything is passed to the script untouched.
"""
from __future__ import annotations

import logging
import os
import runpy
import sys
from dataclasses import dataclass, field

import hotkernels

# The command's options.
INEXACT = "--inexact"
REPORT = "--report"
MODULE = "-m"
HELP = ("-h", "--help")
# Ends the options, for a script whose name starts with a dash.
END_OF_OPTIONS = "--"

# Exit status for a command line that cannot be understood, as Python's own.
USAGE_ERROR = 2

USAGE = """\
usage: python -m hotkernels [--inexact] [--report] script.py [args ...]
       python -m hotkernels [--inexact] [--report] -m module [args ...]

Runs the script or module as python would, with hotkernels.install() in
effect, and exits with its status.

  --inexact   also install the kernels that are not bit-identical to NumPy
  --report    print hotkernels.report() to standard error when it ends"""

# The command's logger. Not `__name__`: run with -m, that is "__main__", the
# same logger as a user's script that calls logging.getLogger(__name__).
CLI_LOGGER = "hotkernels.cli"

log = logging.getLogger(CLI_LOGGER)


class UsageError(Exception):
    """The command line cannot be understood; the message says why."""


@dataclass(frozen=True)
class Invocation:
    """What to run, and how, as read from the command line."""

    # Whether to install the kernels that are not bit-identical to NumPy.
    inexact: bool
    # Whether to print the report when the script ends.
    report: bool
    # The script's path, or the module's name when `is_module` is set.
    target: str
    # Whether `target` is a module to run as `python -m` would.
    is_module: bool
    # The arguments that follow the target, passed to it untouched.
    args: list[str] = field(default_factory=list)


def parse(argv: list[str]) -> Invocation | None:
    """Read the command line.

    Args:
        argv: The arguments after `python -m hotkernels`.

    Returns:
        What to run, or None when help was asked for.
    """
    inexact = report = False
    position = 0
    while position < len(argv):
        arg = argv[position]
        if arg in HELP:
            return None
        if arg == INEXACT:
            inexact = True
        elif arg == REPORT:
            report = True
        elif arg == MODULE:
            if position + 1 == len(argv):
                raise UsageError("-m needs a module name")
            return Invocation(inexact, report, argv[position + 1], True, argv[position + 2:])
        elif arg.startswith(MODULE) and not arg.startswith("--"):
            # Python's own -mmodule spelling, with no space.
            return Invocation(inexact, report, arg[len(MODULE):], True, argv[position + 1:])
        elif arg == END_OF_OPTIONS:
            position += 1
            break
        elif arg.startswith("-"):
            raise UsageError(f"unknown option {arg}")
        else:
            break
        position += 1
    if position == len(argv):
        raise UsageError("no script or module to run")
    return Invocation(inexact, report, argv[position], False, argv[position + 1:])


def run(invocation: Invocation) -> None:
    """Install hotkernels and run the script or module.

    A `SystemExit` from the script passes through, so its exit status is the
    command's.

    Args:
        invocation: What to run.
    """
    hotkernels.install(inexact=invocation.inexact)
    sys.argv = [invocation.target, *invocation.args]
    try:
        if invocation.is_module:
            # run_module replaces sys.argv[0] with the module's file, as `python -m` does.
            runpy.run_module(invocation.target, run_name="__main__", alter_sys=True)
        else:
            # `python script.py` puts the script's directory first on the path;
            # under `python -m` that place holds the working directory instead.
            sys.path[0] = os.path.dirname(os.path.abspath(invocation.target))
            runpy.run_path(invocation.target, run_name="__main__")
    finally:
        if invocation.report:
            log.info("%s", hotkernels.report())


def _configure_logging() -> None:
    """Send the command's messages to standard error, leaving the root logger alone.

    The root logger belongs to the user's program, which runs in this process.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


def main(argv: list[str] | None = None) -> None:
    """Run the command.

    Args:
        argv: The arguments after `python -m hotkernels`; `sys.argv[1:]` when None.
    """
    _configure_logging()
    try:
        invocation = parse(sys.argv[1:] if argv is None else argv)
    except UsageError as error:
        log.error("python -m hotkernels: %s\n\n%s", error, USAGE)
        raise SystemExit(USAGE_ERROR)
    if invocation is None:
        log.info("%s", USAGE)
        return
    run(invocation)


if __name__ == "__main__":
    main()
