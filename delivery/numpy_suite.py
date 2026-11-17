"""Runs NumPy's own test suite on plain NumPy and again with hotkernels
installed, and compares the outcome of every test.

NumPy's tests know nothing of hotkernels, which makes them the broadest
evidence available that installing it breaks nothing: a test that passes on
plain NumPy and fails with hotkernels is a behaviour some user relies on.
Only the exact kernels are installed; a kernel that is not bit-identical is
expected to fail tests that compare bits.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .pytest_plugin import INSTALL_VALUE, INSTALL_VARIABLE, OUTCOMES_VARIABLE

# The pytest plugin that installs hotkernels and records outcomes.
PLUGIN = "delivery.pytest_plugin"
# NumPy's own selection for an ordinary test run: everything not marked slow.
MARKER = "not slow"
# Part of NumPy whose tests are left out: they compile Fortran, need tools
# this project does not, and touch nothing hotkernels changes.
SKIPPED_PACKAGE = "f2py"
# Outcome recorded for a test that one of the two runs did not report.
MISSING = "not run"


@dataclass(frozen=True)
class Difference:
    """One test whose outcome differs between the two runs."""

    # pytest's id of the test.
    test: str
    # Outcome on plain NumPy.
    plain: str
    # Outcome with hotkernels installed.
    installed: str


@dataclass(frozen=True)
class Comparison:
    """The result of the two runs."""

    # Number of tests either run reported.
    tests: int
    # Number that passed on plain NumPy.
    passed_plain: int
    # Number that passed with hotkernels installed.
    passed_installed: int
    # Every test whose outcome differs.
    differences: tuple[Difference, ...]


def _run_suite(install: bool, outcomes: Path, select: str | None) -> None:
    """Run NumPy's tests once in a subprocess and record every outcome.

    Args:
        install: Whether hotkernels is installed for the run.
        outcomes: File the outcomes are written to.
        select: A pytest `-k` expression to run part of the suite, or None.
    """
    environment = dict(os.environ, **{OUTCOMES_VARIABLE: str(outcomes)})
    environment[INSTALL_VARIABLE] = INSTALL_VALUE if install else ""
    command = [sys.executable, "-m", "pytest", "--pyargs", "numpy", "-p", PLUGIN, "-m", MARKER,
               "-q", "--no-header", "-p", "no:cacheprovider", "--continue-on-collection-errors",
               f"--ignore={Path(np.__file__).parent / SKIPPED_PACKAGE}"]
    if select:
        command += ["-k", select]
    # pytest's exit status says tests failed, which is an outcome to compare
    # and not an error of this tool; its output is not needed.
    subprocess.run(command, env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def _passed(outcomes: dict[str, str]) -> int:
    """Count the tests whose call phase passed."""
    return sum(1 for outcome in outcomes.values() if outcome == "call:passed")


def compare(select: str | None = None) -> Comparison:
    """Run the suite twice and compare.

    Args:
        select: A pytest `-k` expression to run part of the suite, or None
            for all of it.

    Returns:
        The comparison. A test missing from one run counts as a difference.
    """
    with tempfile.TemporaryDirectory(prefix="delivery-numpy-") as directory:
        plain_file, installed_file = Path(directory) / "plain.json", Path(directory) / "installed.json"
        _run_suite(False, plain_file, select)
        _run_suite(True, installed_file, select)
        plain = json.loads(plain_file.read_text())
        installed = json.loads(installed_file.read_text())
    tests = sorted(set(plain) | set(installed))
    differences = tuple(
        Difference(test, plain.get(test, MISSING), installed.get(test, MISSING))
        for test in tests if plain.get(test, MISSING) != installed.get(test, MISSING)
    )
    return Comparison(len(tests), _passed(plain), _passed(installed), differences)
