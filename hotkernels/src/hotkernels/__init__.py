"""hotkernels: faster kernels for NumPy operations, installed with one line.

    import hotkernels
    hotkernels.install()

`install()` points a handful of function pointers inside NumPy at kernels that
were checked against NumPy and timed against it. Existing NumPy code then
runs faster with no other change. A call a kernel does not handle goes to
NumPy exactly as it was written. `uninstall()` puts every pointer back, and
`report()` shows what is installed and how often each kernel ran.

Importing this package changes nothing. Only `install()` does, and it covers
the whole process: every thread and every library that calls NumPy.
"""
from __future__ import annotations

import contextlib
import importlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from . import report as _report
from .environment import BINDING_MODULE, unsupported_reason
from .hooks import HOOKS, Hook, Probe
from .report import Report

__all__ = ["install", "uninstall", "enabled", "report", "Report"]

log = logging.getLogger(__name__)


@dataclass
class _State:
    """What install() has done in this process."""

    # Whether install() is in effect.
    installed: bool = False
    # Whether kernels that are not bit-identical to NumPy were asked for.
    inexact: bool = False
    # Why install() did nothing, when it did nothing.
    unsupported: str | None = None
    # One line per hook that could not be installed or removed cleanly.
    problems: list[str] = field(default_factory=list)


# The one installation state of the process, and the lock that guards it.
_state = _State()
_lock = threading.RLock()


def _load_binding():
    """Return the extension module that switches pointers and counts calls."""
    return importlib.import_module(BINDING_MODULE)


def _probe_passes(binding, probe: Probe) -> bool:
    """Run one self-test call and say whether its kernel took it and was right.

    Args:
        binding: The extension module.
        probe: The call to make and the kernel that should take it.

    Returns:
        True when the answer was right and the kernel's call counter moved.
    """
    before = binding.counters()[0][probe.kernel]
    try:
        right = probe.call()
    except Exception:  # A probe that raises has failed, whatever it raised.
        log.debug("probe for %s raised", probe.kernel, exc_info=True)
        return False
    return right and binding.counters()[0][probe.kernel] > before


def _apply(binding, hook: Hook, inexact: bool) -> str | None:
    """Apply one hook and keep it only if every probe passes.

    Args:
        binding: The extension module.
        hook: The hook to apply.
        inexact: Whether kernels that are not bit-identical to NumPy were
            asked for. Their probes are skipped otherwise.

    Returns:
        None when the hook is in place, otherwise a line saying why it is not.
        NumPy is as it was in that case.
    """
    try:
        binding.apply(hook.name)
    except Exception as error:  # Any failure here means NumPy is not shaped as expected.
        return f"{hook.name}: could not be applied ({error})"
    for probe in hook.probes:
        if probe.inexact and not inexact:
            continue
        if not _probe_passes(binding, probe):
            binding.remove(hook.name)
            return f"{hook.name}: removed again, its self-test for {probe.kernel} failed"
    return None


def _remove_all(binding) -> list[str]:
    """Remove every hook that is in place, in reverse order.

    Args:
        binding: The extension module.

    Returns:
        One line per hook whose pointer had been changed by something else
        since install() and was therefore left alone.
    """
    problems = []
    for hook in reversed(HOOKS):
        if binding.applied(hook.name) and not binding.remove(hook.name):
            problems.append(f"{hook.name}: left in place, something else changed it after install()")
    return problems


def install(inexact: bool = False) -> None:
    """Make NumPy use the kernels, for the whole process.

    Does nothing on a platform, Python build or NumPy that is not supported,
    and never raises because of one; `report()` then says why. Calling it
    again with the same argument does nothing.

    Args:
        inexact: Also install the kernels whose results are not bit-identical
            to NumPy's. Today those are the float32 and float64 `cumsum`
            kernels, which regroup the additions: their output stays within a
            declared tolerance but, unlike NumPy's, is not monotonic for
            non-negative input.
    """
    global _state
    with _lock:
        if _state.installed and _state.inexact == inexact:
            return
        reason = unsupported_reason()
        if reason:
            _state = _State(unsupported=reason)
            log.info("hotkernels is not installed: %s", reason)
            return
        binding = _load_binding()
        _remove_all(binding)
        binding.set_inexact(inexact)
        problems = [problem for problem in (_apply(binding, hook, inexact) for hook in HOOKS) if problem]
        binding.reset_counters()
        _state = _State(installed=True, inexact=inexact, problems=problems)
        for problem in problems:
            log.warning("hotkernels: %s", problem)
        log.info("hotkernels installed: %d of %d hooks", len(HOOKS) - len(problems), len(HOOKS))


def uninstall() -> None:
    """Put every pointer back, so NumPy is as it was before `install()`.

    Does nothing when hotkernels is not installed.
    """
    global _state
    with _lock:
        if not _state.installed:
            return
        binding = _load_binding()
        problems = _remove_all(binding)
        binding.set_inexact(False)
        _state = _State(problems=problems)
        for problem in problems:
            log.warning("hotkernels: %s", problem)
        log.info("hotkernels uninstalled")


@contextlib.contextmanager
def enabled(inexact: bool = False) -> Iterator[None]:
    """Install hotkernels for the duration of a `with` block.

    The kernels are installed for the whole process while the block runs, not
    only for the thread that entered it. If hotkernels was already installed,
    the block leaves it installed.

    Args:
        inexact: As for `install()`.

    Returns:
        A context manager that yields nothing.
    """
    with _lock:
        was_installed = _state.installed
        install(inexact)
    try:
        yield
    finally:
        if not was_installed:
            uninstall()


def report() -> Report:
    """Say what is installed and how often each kernel has run.

    Returns:
        A `Report`. Printing it gives a table with one row per kernel: the
        NumPy call it serves, whether it is assembly or C, whether it is
        bit-identical to NumPy, how many calls it took, and how many calls
        reached its hook and were left to NumPy.
    """
    with _lock:
        taken: dict[str, int] = {}
        passed: dict[str, int] = {}
        applied: frozenset[str] = frozenset()
        if _state.installed:
            binding = _load_binding()
            taken, passed = binding.counters()
            applied = frozenset(hook.name for hook in HOOKS if binding.applied(hook.name))
        return _report.build(_state.installed, _state.unsupported, _state.inexact, np.__version__,
                             applied, taken, passed, tuple(_state.problems))
