"""Lint and correctness: the automatic pass or fail."""
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
from .spec import Kernel, load_spec

_TIMEOUT = 300
_MEMORY_SIGNALS = {signal.SIGSEGV, signal.SIGBUS}


@dataclass
class Failure:
    category: str  # lint | build | wrong result | memory safety | calling convention | crash | reference | judge error
    message: str
    details: list[str] = field(default_factory=list)
    case: str | None = None
    who: str = "kernel"


@dataclass
class Built:
    k: Kernel
    kernel_dir: Path
    asm: Path
    kernel: Path
    naive: Path
    harness: Path


@dataclass
class CheckResult:
    name: str
    ok: bool
    failure: Failure | None = None
    cases: int = 0
    fresh_seed: int = 0
    built: Built | None = None


def build_all(kernel_dir: Path, asm: Path | None = None, tag: str = "kernel",
              lint: bool = True) -> tuple[Built | None, Failure | None, Kernel]:
    k = load_spec(kernel_dir)
    asm = asm or kernel_dir / ARCH / "kernel.s"
    if not asm.exists():
        return None, Failure("build", f"no {ARCH} kernel at {asm}"), k
    if lint:
        issues = lint_source(asm.read_text(), ARCH)
        if issues:
            return None, Failure("lint", issues[0], issues[1:]), k
    try:
        kernel, issues = build_asm(k, asm, tag)
        if issues:
            return None, Failure("lint", issues[0], issues[1:]), k
        naive = build_naive(k, kernel_dir / "naive.c")
        harness = build_harness(k)
    except BuildError as e:
        lines = str(e).splitlines() or ["build failed"]
        return None, Failure("build", lines[0], lines[1:12]), k
    return Built(k, kernel_dir, asm, kernel, naive, harness), None, k


def _start_worker(built: Built, dylib: Path, who: str, fresh_seed: int, case: str | None):
    progress = BUILD / built.k.name / f"progress-{who}-{os.getpid()}.txt"
    progress.write_text("")
    proc = subprocess.Popen(
        [sys.executable, "-m", "judge.worker", str(built.kernel_dir), str(dylib),
         str(built.harness), str(progress), str(fresh_seed), case or ""],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    return proc, progress


def _collect(proc: subprocess.Popen, progress: Path, who: str) -> tuple[Failure | None, int]:
    try:
        stdout, stderr = proc.communicate(timeout=_TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        last = progress.read_text().strip()
        progress.unlink(missing_ok=True)
        return Failure("crash", f"did not finish within {_TIMEOUT} seconds", case=last, who=who), 0
    last = progress.read_text().strip()
    progress.unlink(missing_ok=True)

    if proc.returncode < 0:
        sig = signal.Signals(-proc.returncode)
        if sig in _MEMORY_SIGNALS:
            return Failure("memory safety",
                           f"crashed with {sig.name}: accessed memory outside a buffer, or wrote to an input",
                           case=last, who=who), 0
        return Failure("crash", f"crashed with {sig.name}", case=last, who=who), 0
    if proc.returncode != 0:
        lines = stderr.strip().splitlines()
        return Failure("judge error", lines[-1] if lines else "worker failed", lines[-8:-1],
                       case=last or None, who=who), 0

    result = json.loads(stdout.strip().splitlines()[-1])
    if result["ok"]:
        return None, result["cases"]
    details = []
    if "expected" in result:
        details = [f"expected: {result['expected']}", f"got:      {result['got']}"]
    return Failure(result["category"], result["message"], details, result["case"], who), result["cases"]


def check(kernel_dir: Path, asm: Path | None = None, case: str | None = None,
          fresh_seed: int | None = None, tag: str = "kernel") -> CheckResult:
    """Lint, build and run every correctness case against the kernel and the naive reference."""
    built, failure, k = build_all(kernel_dir, asm, tag)
    if failure:
        return CheckResult(k.name, False, failure)
    if fresh_seed is None:
        fresh_seed = int.from_bytes(os.urandom(4), "little")

    workers = {
        "naive": _start_worker(built, built.naive, "naive", fresh_seed, case),
        "kernel": _start_worker(built, built.kernel, "kernel", fresh_seed, case),
    }
    outcomes = {who: _collect(proc, progress, who) for who, (proc, progress) in workers.items()}

    naive_failure, _ = outcomes["naive"]
    if naive_failure and naive_failure.category == "judge error":
        return CheckResult(k.name, False, naive_failure, fresh_seed=fresh_seed, built=built)
    if naive_failure:
        naive_failure = Failure("reference", f"naive.c is wrong: {naive_failure.message}",
                                naive_failure.details, naive_failure.case, "naive")
        return CheckResult(k.name, False, naive_failure, fresh_seed=fresh_seed, built=built)
    kernel_failure, cases = outcomes["kernel"]
    if kernel_failure:
        return CheckResult(k.name, False, kernel_failure, cases, fresh_seed, built)
    return CheckResult(k.name, True, None, cases, fresh_seed, built)


def format_result(result: CheckResult, command_name: str | None = None) -> str:
    head = f"{result.name} ({ARCH})"
    if result.ok:
        return f"PASS {head}: {result.cases:,} cases, fresh seed {result.fresh_seed}"
    f = result.failure
    lines = [f"FAIL {head}: {f.category}", f"  {f.message}"]
    if f.case:
        lines.append(f"  case:     {f.case}")
    lines += [f"  {d}" for d in f.details]
    if f.case and command_name:
        lines.append(f"  rerun:    uv run judge check {command_name} --case '{f.case}'")
    return "\n".join(lines)
