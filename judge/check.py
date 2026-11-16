"""Lint and correctness: the automatic pass or fail.

`check` lints the assembly, builds everything, and runs every correctness case
against both the kernel and the naive C reference. The naive reference is
checked too, because the speed comparison is meaningless if it is wrong.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .build import ARCH, BUILD, BuildError, build_asm, build_harness, build_naive
from .lint import lint_source
from .render import render
from .spec import Kernel, load_spec
from .worker import MEMORY_SAFETY

# Failure categories decided here. The worker defines the rest.
LINT = "lint"
BUILD_FAILED = "build"
CRASH = "crash"
REFERENCE = "reference"      # the naive C reference itself is wrong
JUDGE_ERROR = "judge error"  # the judge or the spec failed, not the kernel

KERNEL = "kernel"
NAIVE = "naive"

# A check normally takes seconds. A kernel that has not finished by now is hung.
_TIMEOUT_SECONDS = 300
# Signals that mean the kernel touched memory it must not.
_MEMORY_SIGNALS = {signal.SIGSEGV, signal.SIGBUS}
# How many lines of tool output or traceback to keep in a failure.
_MAX_BUILD_DETAIL_LINES = 12
_MAX_TRACEBACK_LINES = 8
# Size of the random seed for fresh cases, in bytes.
_SEED_BYTES = 4


@dataclass
class Failure:
    """Why a kernel was rejected."""

    # One of the failure categories, such as "wrong result" or "memory safety".
    category: str
    # One-line explanation.
    message: str
    # Extra lines to show under the message, such as expected and actual values.
    details: list[str] = field(default_factory=list)
    # Description of the case that failed, in the form `--case` accepts. None
    # when the failure happened before any case ran.
    case: str | None = None
    # Which code failed: KERNEL for the assembly, NAIVE for the C reference.
    who: str = KERNEL


@dataclass
class Built:
    """The compiled artefacts for one kernel."""

    # The kernel's spec.
    k: Kernel
    # Directory holding spec.py and naive.c.
    kernel_dir: Path
    # Assembly source the kernel was built from.
    asm: Path
    # Shared library containing the assembly kernel.
    kernel: Path
    # Shared library containing the naive C reference.
    naive: Path
    # Shared library with the timing loop and the calling-convention wrapper.
    harness: Path


@dataclass
class CheckResult:
    """The outcome of a check."""

    # Kernel name.
    name: str
    # True when lint, build and every correctness case passed.
    ok: bool
    # The first failure, or None when ok.
    failure: Failure | None = None
    # How many cases ran against the kernel.
    cases: int = 0
    # Seed of the fresh random cases, so a run can be reproduced.
    fresh_seed: int = 0
    # The compiled artefacts, when the build got that far.
    built: Built | None = None


def build_all(kernel_dir: Path, asm: Path | None = None, tag: str = KERNEL,
              lint: bool = True) -> tuple[Built | None, Failure | None, Kernel]:
    """Lint the assembly and build the kernel, the naive reference and the harness.

    `asm` overrides the default source path, and `tag` names the build outputs;
    the selftest uses both to build broken variants. Returns (artefacts,
    failure, spec), where exactly one of artefacts and failure is None.
    """
    k = load_spec(kernel_dir)
    asm = asm or kernel_dir / ARCH / "kernel.s"
    if not asm.exists():
        return None, Failure(BUILD_FAILED, f"no {ARCH} kernel at {asm}"), k
    if lint:
        issues = lint_source(asm.read_text(), ARCH)
        if issues:
            return None, Failure(LINT, issues[0], issues[1:]), k
    try:
        kernel, issues = build_asm(k, asm, tag)
        if issues:
            return None, Failure(LINT, issues[0], issues[1:]), k
        naive = build_naive(k, kernel_dir / "naive.c")
        harness = build_harness(k)
    except BuildError as e:
        lines = str(e).splitlines() or ["build failed"]
        return None, Failure(BUILD_FAILED, lines[0], lines[1:_MAX_BUILD_DETAIL_LINES]), k
    return Built(k, kernel_dir, asm, kernel, naive, harness), None, k


@dataclass
class _Worker:
    """A running correctness subprocess and the files it reports through."""

    # The subprocess.
    proc: subprocess.Popen
    # File the worker updates with the case about to run.
    progress: Path
    # File the worker writes its outcome to.
    result: Path
    # Which code it is testing: KERNEL or NAIVE.
    who: str


def _start_worker(built: Built, dylib: Path, who: str, fresh_seed: int, case: str | None) -> _Worker:
    """Start a subprocess that runs the correctness cases against one library."""
    stem = BUILD / built.k.name / f"{who}-{os.getpid()}"
    progress, result = stem.with_suffix(".progress"), stem.with_suffix(".result")
    progress.write_text("")
    result.unlink(missing_ok=True)
    command = [sys.executable, "-m", "judge.worker", str(built.kernel_dir), str(dylib),
               str(built.harness), str(progress), str(result), str(fresh_seed)]
    if case:
        command += ["--case", case]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return _Worker(proc, progress, result, who)


def _collect(worker: _Worker) -> tuple[Failure | None, int]:
    """Wait for a worker and turn its outcome into (failure, cases run).

    Handles the three ways a worker can end: it hangs, it is killed by a
    signal, or it exits and leaves a result file.
    """
    try:
        _, stderr = worker.proc.communicate(timeout=_TIMEOUT_SECONDS)
        timed_out = False
    except subprocess.TimeoutExpired:
        worker.proc.kill()
        _, stderr = worker.proc.communicate()
        timed_out = True
    last_case = worker.progress.read_text().strip() or None
    outcome = json.loads(worker.result.read_text()) if worker.result.exists() else None
    worker.progress.unlink(missing_ok=True)
    worker.result.unlink(missing_ok=True)

    if timed_out:
        return Failure(CRASH, f"did not finish within {_TIMEOUT_SECONDS} seconds",
                       case=last_case, who=worker.who), 0
    returncode = worker.proc.returncode
    if returncode < 0:  # killed by a signal
        name = signal.Signals(-returncode).name
        if -returncode in _MEMORY_SIGNALS:
            message = f"crashed with {name}: accessed memory outside a buffer, or wrote to an input"
            return Failure(MEMORY_SAFETY, message, case=last_case, who=worker.who), 0
        return Failure(CRASH, f"crashed with {name}", case=last_case, who=worker.who), 0
    if returncode != 0 or outcome is None:
        # The worker itself raised: a bug in the judge or in the kernel's spec.
        lines = stderr.strip().splitlines()
        message = lines[-1] if lines else "worker failed"
        return Failure(JUDGE_ERROR, message, lines[-_MAX_TRACEBACK_LINES:-1],
                       case=last_case, who=worker.who), 0

    if outcome["ok"]:
        return None, outcome["cases"]
    details = []
    if "expected" in outcome:
        details = [f"expected: {outcome['expected']}", f"got:      {outcome['got']}"]
    failure = Failure(outcome["category"], outcome["message"], details, outcome["case"], worker.who)
    return failure, outcome["cases"]


def check(kernel_dir: Path, asm: Path | None = None, case: str | None = None,
          fresh_seed: int | None = None, tag: str = KERNEL) -> CheckResult:
    """Lint, build and run every correctness case.

    The kernel and the naive reference run in parallel subprocesses. A wrong
    naive reference is reported first, since nothing else can be trusted then.
    Pass `case` to run a single case, as printed by an earlier failure.
    """
    built, failure, k = build_all(kernel_dir, asm, tag)
    if failure:
        return CheckResult(k.name, False, failure)
    if fresh_seed is None:
        fresh_seed = int.from_bytes(os.urandom(_SEED_BYTES), "little")

    workers = [
        _start_worker(built, built.naive, NAIVE, fresh_seed, case),
        _start_worker(built, built.kernel, KERNEL, fresh_seed, case),
    ]
    outcomes = {worker.who: _collect(worker) for worker in workers}

    naive_failure, _ = outcomes[NAIVE]
    if naive_failure:
        if naive_failure.category != JUDGE_ERROR:
            naive_failure = Failure(REFERENCE, f"naive.c is wrong: {naive_failure.message}",
                                    naive_failure.details, naive_failure.case, NAIVE)
        return CheckResult(k.name, False, naive_failure, fresh_seed=fresh_seed, built=built)
    kernel_failure, cases = outcomes[KERNEL]
    return CheckResult(k.name, kernel_failure is None, kernel_failure, cases, fresh_seed, built)


def format_result(result: CheckResult, command_name: str | None = None) -> str:
    """Render a check outcome as the PASS or FAIL text shown to the user.

    `command_name` is the kernel name to use in the suggested rerun command.
    """
    return render("check.txt.j2", result=result, arch=ARCH, command_name=command_name).rstrip()
