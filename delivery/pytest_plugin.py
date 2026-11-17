"""A pytest plugin for running another project's tests under hotkernels.

Loaded with `-p delivery.pytest_plugin`. It installs hotkernels before any
test runs when the environment asks for it, and writes the outcome of every
test to a JSON file, so that a run with hotkernels can be compared with a run
without.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import hotkernels

# Environment variable that, set to INSTALL_VALUE, makes the plugin call
# hotkernels.install() before the tests.
INSTALL_VARIABLE = "DELIVERY_INSTALL_HOTKERNELS"
INSTALL_VALUE = "1"
# Environment variable naming the JSON file the outcomes are written to.
OUTCOMES_VARIABLE = "DELIVERY_OUTCOMES_FILE"
# The phase of a test whose outcome is the test's own, as pytest names it.
CALL_PHASE = "call"

# Test id -> outcome, filled as the tests run.
_outcomes: dict[str, str] = {}


def pytest_configure(config) -> None:
    """Install hotkernels for the whole test session, if asked to.

    Args:
        config: The pytest configuration. Unused.
    """
    if os.environ.get(INSTALL_VARIABLE) == INSTALL_VALUE:
        hotkernels.install()


def pytest_runtest_logreport(report) -> None:
    """Record the outcome of one phase of one test.

    A test's outcome is that of its call phase. A failure in setup or
    teardown is recorded too, so that an error there is not lost.

    Args:
        report: pytest's report for the phase.
    """
    if report.when == CALL_PHASE or report.outcome != "passed":
        _outcomes[report.nodeid] = f"{report.when}:{report.outcome}"


def pytest_sessionfinish(session, exitstatus) -> None:
    """Write every recorded outcome to the file the environment names.

    Args:
        session: The pytest session. Unused.
        exitstatus: pytest's exit status. Unused.
    """
    path = os.environ.get(OUTCOMES_VARIABLE)
    if path:
        Path(path).write_text(json.dumps(_outcomes, indent=0, sort_keys=True))
