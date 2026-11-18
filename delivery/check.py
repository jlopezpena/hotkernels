"""Runs the cases: each call once on plain NumPy and once with hotkernels
installed, and compares everything a caller could observe.

Two runs of a case agree when the result has the same type, element type,
shape, flags and bytes (or, for a kernel that declares a tolerance, values
within it), the same exception was raised with the same message, the same
warnings were issued, and the arguments are in the same state afterwards.
On top of that each case says whether a kernel should have taken the call,
and the binding's counters are checked against it: a condition in the
binding that silently stopped matching would pass every comparison while
delivering nothing.
"""
from __future__ import annotations

import hashlib
import importlib
import warnings
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

import numpy as np

import hotkernels
from hotkernels.environment import BINDING_MODULE
from hotkernels.hooks import KERNELS

from .cases import ANY, Case

# Bytes of a digest that stands in for an array's contents.
DIGEST_BYTES = 16
# Longest description of a value shown in a failure.
SHOWN_CHARACTERS = 300

# What a case ended in.
RETURNED = "returned"
RAISED = "raised"


@dataclass(frozen=True)
class Outcome:
    """Everything observable about one run of a case."""

    # (RETURNED, description of the result) or (RAISED, exception type, message).
    ending: tuple
    # (category name, message) of every warning issued, in order.
    warnings: tuple
    # Description of the arguments after the call.
    arguments: tuple
    # The result itself, kept only for cases compared within a tolerance.
    raw: object = None


@dataclass(frozen=True)
class Failure:
    """One case whose two runs disagree, or whose kernel expectation failed."""

    # The case's label.
    label: str
    # What disagreed.
    problem: str
    # What plain NumPy gave, shortened.
    expected: str
    # What the run with hotkernels gave, shortened.
    got: str


@dataclass(frozen=True)
class Summary:
    """The outcome of one operation's cases."""

    # Operation name, with a suffix for the run under install(inexact=True).
    operation: str
    # Number of cases run.
    cases: int
    # Number of cases a kernel took.
    taken: int
    # The cases that failed.
    failures: tuple[Failure, ...]


def _digest(data: bytes) -> str:
    """Return a short hash that stands in for a buffer's contents."""
    return hashlib.blake2b(data, digest_size=DIGEST_BYTES).hexdigest()


def describe(value: object) -> tuple:
    """Reduce a value to a comparable description.

    An array becomes its type, element type, shape, flags and a hash of its
    bytes, so +0.0 differs from -0.0 and NaNs are compared by their bits. A
    masked array adds its mask. Containers are described element by element.

    Args:
        value: A result, or an argument after a call.

    Returns:
        A nested tuple that is equal for two values exactly when a caller
        could not tell them apart.
    """
    if isinstance(value, np.ma.MaskedArray):
        return ("masked", type(value).__name__, describe(np.asarray(value.data)), describe(np.ma.getmaskarray(value)))
    if isinstance(value, np.ndarray):
        contents = repr(value.tolist()) if value.dtype.hasobject else _digest(np.ascontiguousarray(value).tobytes())
        flags = (value.flags.c_contiguous, value.flags.f_contiguous, value.flags.writeable)
        return ("array", type(value).__name__, value.dtype.str, value.shape, flags, contents)
    if isinstance(value, np.generic):
        return ("scalar", type(value).__name__, value.dtype.str, _digest(value.tobytes()))
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(describe(item) for item in value))
    return ("object", type(value).__name__, repr(value))


def capture(case: Case) -> Outcome:
    """Run one case and record everything observable about it.

    Args:
        case: The case. Its arguments are built afresh.

    Returns:
        The outcome, with the raw result kept when the case has a tolerance.
    """
    arguments = case.make()
    raw = None
    with warnings.catch_warnings(record=True) as issued:
        warnings.simplefilter("always")
        try:
            raw = case.call(*arguments)
            ending = (RETURNED, describe(raw))
        except Exception as error:  # Any exception is a result to compare.
            ending = (RAISED, type(error).__name__, str(error))
    recorded = tuple((w.category.__name__, str(w.message)) for w in issued)
    return Outcome(ending, recorded, describe(arguments), raw if case.tolerance else None)


def _within_tolerance(case: Case, expected: Outcome, got: Outcome) -> bool:
    """Say whether a result is within the tolerance its kernel's spec declares.

    The rule is the judge's: NaN and infinity must be in the same positions,
    and a finite element may differ by the spec's number of units in the last
    place, taken at the magnitude the spec's scale function gives.

    Args:
        case: A case with a tolerance, whose one argument is the input array.
        expected: The run on plain NumPy.
        got: The run with hotkernels.

    Returns:
        True when both runs returned arrays of one type and shape that agree
        within the tolerance.
    """
    a, b = expected.raw, got.raw
    if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
        return False
    if type(a) is not type(b) or a.dtype != b.dtype or a.shape != b.shape:
        return False
    (x,) = case.make()
    scale = np.asarray(case.tolerance.tolerance_scale(np.asarray(x).ravel()), dtype=np.float64)
    special = ~np.isfinite(a) | ~np.isfinite(b)
    identical = (a == b) | (np.isnan(a) & np.isnan(b))
    with np.errstate(invalid="ignore", over="ignore"):
        magnitude = np.minimum(np.abs(scale), np.finfo(a.dtype).max).astype(a.dtype)
        close = np.abs(b - a) <= case.tolerance.tolerance_ulp * np.spacing(magnitude)
    return bool(np.where(special, identical, close).all())


def _shorten(value: object) -> str:
    """Format a value for a failure message, cut to a readable length."""
    text = repr(value)
    return text if len(text) <= SHOWN_CHARACTERS else text[:SHOWN_CHARACTERS] + "..."


def _disagreement(case: Case, expected: Outcome, got: Outcome) -> Failure | None:
    """Compare the two runs of a case.

    Args:
        case: The case.
        expected: The run on plain NumPy.
        got: The run with hotkernels installed.

    Returns:
        A Failure naming the first thing that differs, or None.
    """
    if expected.ending != got.ending:
        both_returned = expected.ending[0] == got.ending[0] == RETURNED
        if not (case.tolerance and both_returned and _within_tolerance(case, expected, got)):
            return Failure(case.label, "result differs", _shorten(expected.ending), _shorten(got.ending))
    if expected.warnings != got.warnings:
        return Failure(case.label, "warnings differ", _shorten(expected.warnings), _shorten(got.warnings))
    if expected.arguments != got.arguments:
        return Failure(case.label, "arguments differ after the call", _shorten(expected.arguments), _shorten(got.arguments))
    return None


def _expectation_failure(case: Case, own_kernels: tuple[str, ...], before: dict, after: dict) -> Failure | None:
    """Check the binding's counters against what a case expects.

    Args:
        case: The case.
        own_kernels: The kernels of the case's operation.
        before: Calls taken per kernel before the case ran.
        after: The same afterwards.

    Returns:
        A Failure when the expected kernel did not run, or when one of the
        operation's kernels ran for a call that must be left to NumPy.
    """
    if case.expect == ANY:
        return None
    ran = [kernel for kernel in own_kernels if after[kernel] > before[kernel]]
    if case.expect is None and ran:
        return Failure(case.label, "a kernel took a call that belongs to NumPy", "no kernel", ", ".join(ran))
    if case.expect is not None and after[case.expect] <= before[case.expect]:
        return Failure(case.label, "the kernel did not take the call", case.expect, ", ".join(ran) or "no kernel")
    return None


def operation_kernels(operation: str) -> tuple[str, ...]:
    """Return the names of the kernels that serve an operation."""
    return tuple(kernel.name for kernel in KERNELS if kernel.operation == operation)


def run_operation(operation: str, cases: Iterable[Case], inexact: bool, keep: Callable[[Case], bool]) -> Summary:
    """Run one operation's cases on plain NumPy, then with hotkernels installed.

    Args:
        operation: Operation name.
        cases: The cases.
        inexact: Whether to install the kernels that are not bit-identical.
        keep: Says which cases to run.

    Returns:
        The summary. hotkernels is left uninstalled.
    """
    selected = [case for case in cases if keep(case)]
    hotkernels.uninstall()
    expected = [capture(case) for case in selected]
    hotkernels.install(inexact=inexact)
    binding = importlib.import_module(BINDING_MODULE)
    own_kernels = operation_kernels(operation)
    failures, taken = [], 0
    try:
        for case, want in zip(selected, expected):
            before = binding.counters()[0]
            got = capture(case)
            after = binding.counters()[0]
            taken += any(after[kernel] > before[kernel] for kernel in own_kernels)
            failure = _disagreement(case, want, got) or _expectation_failure(case, own_kernels, before, after)
            if failure:
                failures.append(failure)
    finally:
        hotkernels.uninstall()
    name = f"{operation} (inexact)" if inexact else operation
    return Summary(name, len(selected), taken, tuple(failures))


# Name of the summary line for the checks of install() and uninstall() themselves.
INSTALLATION = "install and uninstall"
# NumPy objects whose identity must survive install() and uninstall(): functions
# of the numpy module, methods of ndarray, and methods of the ufunc type.
_HOOKED_OBJECTS = ("argmax", "argmin", "count_nonzero", "sum", "histogram", "bincount", "isin", "interp", "clip",
                   "compress", "extract", "putmask", "copyto", "where")
_HOOKED_METHODS = ("astype", "sum", "nonzero", "cumsum", "searchsorted", "argmax", "argmin", "clip")
_HOOKED_UFUNC_METHODS = ("accumulate",)
# The hooks behind the clip ufunc's loop, which also take every call the
# function and method hooks of clip take.
_CLIP_LOOP_HOOKS = ("loop.clip.f32", "loop.clip.f64")
# Element count of the arrays the check of clip's direct path clips.
_DIRECT_CLIP_N = 16


def _numpy_objects() -> list[object]:
    """Return the NumPy objects hotkernels changes pointers in."""
    return ([getattr(np, name) for name in _HOOKED_OBJECTS] + [getattr(np.ndarray, name) for name in _HOOKED_METHODS]
            + [getattr(np.ufunc, name) for name in _HOOKED_UFUNC_METHODS])


def _kernel_calls(binding) -> int:
    """Return how many calls all kernels have taken."""
    return sum(binding.counters()[0].values())


def _exercise() -> None:
    """Make one call to each kind of hook that a kernel takes when installed."""
    x = np.arange(1000, dtype=np.float64)
    mask = x > 500
    x.argmax(), np.clip(x, 0.0, 1.0), np.count_nonzero(x), x.astype(np.uint8).sum(), x[mask]
    x[mask] = 0.0


def _clip_direct_path(binding) -> bool:
    """Say whether `np.clip` and `x.clip` reach the clip kernels without the clip loop.

    The loop hook takes every call the function and method hooks take, so
    the kernel counters cannot tell which ran. With the loop hooks removed
    for the moment, only the direct path can move them.

    Args:
        binding: The extension module, with hotkernels installed.

    Returns:
        True when each of the four calls, `np.clip` and `x.clip` on float32
        and on float64, moved its kernel's counter and gave NumPy's answer.
    """
    for hook in _CLIP_LOOP_HOOKS:
        binding.remove(hook)
    try:
        served = []
        for dtype, kernel in ((np.float32, "clip_f32"), (np.float64, "clip_f64")):
            x = np.linspace(-1.0, 1.0, _DIRECT_CLIP_N, dtype=dtype)
            expected = np.minimum(np.maximum(x, dtype(-0.5)), dtype(0.5))
            for clip in (np.clip, np.ndarray.clip):
                before = binding.counters()[0][kernel]
                result = clip(x, -0.5, 0.5)
                right = result.dtype == dtype and np.array_equal(result, expected)
                served.append(right and binding.counters()[0][kernel] == before + 1)
        return all(served)
    finally:
        for hook in _CLIP_LOOP_HOOKS:
            binding.apply(hook)


def installation_checks() -> Summary:
    """Check install(), uninstall(), enabled() and report() themselves.

    Returns:
        A summary with one failure per statement that does not hold: NumPy's
        objects keep their identity, uninstall() leaves no hook behind, a
        second install() changes nothing, enabled() restores the state it
        found, and float cumsum is only taken under install(inexact=True).
    """
    binding = importlib.import_module(BINDING_MODULE)
    checks: list[tuple[str, bool]] = []
    hotkernels.uninstall()
    before = _numpy_objects()

    hotkernels.install()
    checks.append(("NumPy's objects are the same objects after install()", all(a is b for a, b in zip(before, _numpy_objects()))))
    _exercise()
    calls = _kernel_calls(binding)
    checks.append(("kernels take calls after install()", calls >= 6))
    hotkernels.install()
    checks.append(("a second install() does not reset the counters", _kernel_calls(binding) == calls))
    checks.append(("report() counts the calls", sum(k.calls for k in hotkernels.report().kernels) == calls))
    checks.append(("report() renders when installed", "installed" in str(hotkernels.report())))
    checks.append(("np.clip and x.clip reach the kernels without the clip loop", _clip_direct_path(binding)))
    np.cumsum(np.ones(1000))
    checks.append(("float cumsum is left to NumPy by install()", binding.counters()[0]["cumsum_f64"] == 0))
    with hotkernels.enabled():
        pass
    checks.append(("enabled() leaves an existing installation in place", hotkernels.report().installed))

    hotkernels.install(inexact=True)
    np.cumsum(np.ones(1000))
    checks.append(("float cumsum is taken under install(inexact=True)", binding.counters()[0]["cumsum_f64"] == 1))

    hotkernels.uninstall()
    binding.reset_counters()
    _exercise()
    checks.append(("no hook is applied after uninstall()", not any(binding.applied(name) for name in binding.hooks())))
    checks.append(("no kernel takes a call after uninstall()", _kernel_calls(binding) == 0))
    checks.append(("NumPy's objects are the same objects after uninstall()", all(a is b for a, b in zip(before, _numpy_objects()))))
    checks.append(("report() renders when not installed", "not installed" in str(hotkernels.report())))
    with hotkernels.enabled():
        checks.append(("enabled() installs for its block", hotkernels.report().installed))
    checks.append(("enabled() uninstalls what it installed", not hotkernels.report().installed))

    failures = tuple(Failure(statement, "does not hold", "true", "false") for statement, holds in checks if not holds)
    return Summary(INSTALLATION, len(checks), 0, failures)


def run(operations: dict, inexact_operations: dict, keep: Callable[[Case], bool]) -> Iterator[Summary]:
    """Run every operation, the exact kernels first.

    Args:
        operations: Operation name -> function yielding its cases, run under
            a plain install().
        inexact_operations: The same for cases run under install(inexact=True).
        keep: Says which cases to run.

    Yields:
        One summary per operation that had at least one case to run.
    """
    for inexact, table in ((False, operations), (True, inexact_operations)):
        for operation, make_cases in table.items():
            summary = run_operation(operation, make_cases(), inexact, keep)
            if summary.cases:
                yield summary
