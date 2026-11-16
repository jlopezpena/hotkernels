"""The judge: checks hotkernels assembly kernels against NumPy and times them.

Run with `uv run judge <command>`.
"""
from __future__ import annotations

import json
import logging
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import typer

from . import bench as B
from . import selftest as S
from .build import ARCH, BUILD, ROOT, BuildError, build_asm
from .check import KERNEL, NAIVE, Built, CheckResult, build_all, check, format_result
from .render import render

log = logging.getLogger(__name__)

KERNELS = ROOT / "kernels"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
# Recorded results, committed to the repository.
RESULTS = ROOT / "results"
# The latest report of every run. Not committed.
RUNS = ROOT / ".judge" / "runs"
LEADERBOARD = ROOT / "LEADERBOARD.md"

# Exit codes. Speed never fails a run; only correctness and misuse do.
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2

# Contender names used in timing reports.
NUMPY = "numpy"
NEW, OLD = "new", "old"
_BENCH_LABELS = {KERNEL: "kernel", NAIVE: "naive C", NUMPY: "NumPy*"}
_COMPARE_LABELS = {NEW: "new", OLD: "old"}

NUMPY_NOTE = (
    "NumPy is timed from Python and includes its per-call overhead; the kernel is timed "
    "without a binding. At small sizes the NumPy column flatters the kernel. Naive C is the "
    "honest baseline."
)

# Branch the reference files must match during a kernel run.
_REFERENCE_BRANCH = "main"
# Anything that is not a lowercase letter or digit becomes a dash in a file name.
_SLUG_SEPARATORS = re.compile(r"[^a-z0-9]+")
_JSON_INDENT = 2

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)

KernelArg = typer.Argument(..., help="Kernel name: a directory under kernels/.")
UnlockedOpt = typer.Option(
    False, "--unlocked",
    help="Allow modified reference files. For people creating a kernel or changing the "
         "judge; never for agent runs.",
)
QuickOpt = typer.Option(False, "--quick", help="Two sizes and few samples, as a smoke test.")


@app.callback()
def configure(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show progress while timing."),
) -> None:
    """Set up logging. Reports go to standard output as plain lines."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
        force=True,
    )


def _git(*args: str) -> subprocess.CompletedProcess:
    """Run a git command in the repository and return the completed process."""
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def resolve(name: str) -> tuple[Path, bool]:
    """Find a kernel by name. Returns (directory, whether it is a judge fixture)."""
    if (KERNELS / name / "spec.py").exists():
        return KERNELS / name, False
    if (FIXTURES / name / "spec.py").exists():
        return FIXTURES / name, True
    log.error("unknown kernel %r: no kernels/%s/spec.py", name, name)
    raise typer.Exit(EXIT_REFUSED)


def reference_changes(name: str) -> list[str]:
    """List the locked files that differ from the reference branch.

    The locked files are the judge itself, its dependencies, and the kernel's
    spec and naive reference: everything a kernel is measured against.
    """
    if _git("rev-parse", "--verify", "--quiet", _REFERENCE_BRANCH).returncode != 0:
        return []
    locked = ["judge", "pyproject.toml", "uv.lock",
              f"kernels/{name}/spec.py", f"kernels/{name}/naive.c"]
    changed = _git("diff", "--name-only", _REFERENCE_BRANCH, "--", *locked).stdout.split()
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", *locked).stdout.split()
    return sorted(set(changed + untracked))


def guard(name: str, is_fixture: bool, unlocked: bool) -> bool:
    """Refuse to run when the reference files were modified.

    Returns True when the reference files are verified unchanged, and False
    when they differ but `unlocked` allowed the run anyway. Fixtures are part
    of the judge and are not guarded.
    """
    if is_fixture:
        return True
    changed = reference_changes(name)
    if not changed:
        return True
    log.info(render("guard.txt.j2", unlocked=unlocked, changed=changed, name=name, arch=ARCH))
    if unlocked:
        return False
    raise typer.Exit(EXIT_REFUSED)


def machine() -> str:
    """Return a human-readable name for this machine's processor."""
    if platform.system() == "Darwin":
        brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                               capture_output=True, text=True).stdout.strip()
        if brand:
            return brand
    return platform.processor() or platform.machine()


def environment(locked: bool) -> dict:
    """Describe where and how a report was produced, for the JSON record."""
    clang = subprocess.run(["clang", "--version"], capture_output=True, text=True).stdout
    return {
        "arch": ARCH,
        "machine": machine(),
        "os": f"{platform.system()} {platform.release()}",
        "clang": clang.splitlines()[0] if clang else "unknown",
        "numpy": np.__version__,
        "python": platform.python_version(),
        "git_head": _git("rev-parse", "--short", "HEAD").stdout.strip(),
        "git_dirty": bool(_git("status", "--porcelain").stdout.strip()),
        "reference_locked": locked,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _log_progress(dist: str, n: int) -> None:
    """Log that one size has been timed. Shown only with --verbose."""
    log.debug("  timed %s n=%s", dist, f"{n:,}")


def _sweep(quick: bool) -> tuple[list[int], int]:
    """Return the (sizes, samples) to time: the full sweep, or the quick one."""
    return (B.QUICK_SIZES, B.QUICK_SAMPLES) if quick else (B.SIZES, B.SAMPLES)


def run_bench(built: Built, quick: bool, locked: bool) -> dict:
    """Time the kernel against naive C and NumPy and return the full report."""
    k = built.k
    harness = B.Harness(k, built.harness)
    kernel_fn, _kernel_lib = B.function_address(built.kernel, k.name)
    naive_fn, _naive_lib = B.function_address(built.naive, k.name)
    sizes, samples = _sweep(quick)

    canary_before = harness.canary()
    data = B.measure(k, harness, {KERNEL: kernel_fn, NAIVE: naive_fn, NUMPY: None},
                     sizes, samples, _log_progress)
    canary_after = harness.canary()

    report = B.analyse(data, KERNEL, [NAIVE, NUMPY])
    report.update(kernel=k.name, samples=samples,
                  warnings=B.noise_warnings(canary_before, canary_after),
                  environment=environment(locked))
    return report


def log_bench(report: dict) -> None:
    """Log the timing table for a kernel against naive C and NumPy."""
    env = report["environment"]
    title = (f"{report['kernel']} ({env['arch']}, {env['machine']}), NumPy {env['numpy']}, "
             f"{report['samples']} samples per point")
    log.info(render("timing.txt.j2", title=title, warnings=report["warnings"], report=report,
                    kernel=KERNEL, baselines=[NAIVE, NUMPY], labels=_BENCH_LABELS,
                    notes=[f"* {NUMPY_NOTE}"], verdict=None).rstrip())


def save_run(report: dict, record: bool) -> None:
    """Write the report as JSON under .judge/runs, and under results/ when recording."""
    env = report["environment"]
    text = json.dumps(report, indent=_JSON_INDENT) + "\n"
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"{report['kernel']}-{env['arch']}.json"
    path.write_text(text)
    log.info("\nfull report: %s", path.relative_to(ROOT))
    if record:
        slug = _SLUG_SEPARATORS.sub("-", env["machine"].lower()).strip("-")
        target = RESULTS / report["kernel"] / f"{env['arch']}-{slug}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        log.info("recorded:    %s", target.relative_to(ROOT))


def _checked(kernel: str, kernel_dir: Path, case: Optional[str] = None) -> CheckResult:
    """Run the correctness check, log its outcome, and exit if it failed."""
    result = check(kernel_dir, case=case)
    log.info(format_result(result, kernel))
    if not result.ok:
        raise typer.Exit(EXIT_FAILED)
    return result


@app.command("check")
def check_command(
    kernel: str = KernelArg,
    case: Optional[str] = typer.Option(None, help="Run a single case, as printed by a failure."),
    unlocked: bool = UnlockedOpt,
) -> None:
    """Lint and correctness only. Fast enough to run after every edit."""
    kernel_dir, is_fixture = resolve(kernel)
    guard(kernel, is_fixture, unlocked)
    _checked(kernel, kernel_dir, case)


@app.command()
def bench(kernel: str = KernelArg, quick: bool = QuickOpt, unlocked: bool = UnlockedOpt) -> None:
    """Timing report only, without the correctness check."""
    kernel_dir, is_fixture = resolve(kernel)
    locked = guard(kernel, is_fixture, unlocked)
    built, failure, k = build_all(kernel_dir)
    if failure:
        log.info(format_result(CheckResult(k.name, False, failure), kernel))
        raise typer.Exit(EXIT_FAILED)
    report = run_bench(built, quick, locked)
    log_bench(report)
    save_run(report, record=False)


@app.command()
def run(
    kernel: str = KernelArg,
    quick: bool = QuickOpt,
    record: bool = typer.Option(False, "--record", help="Also save the report under results/."),
    unlocked: bool = UnlockedOpt,
) -> None:
    """Correctness, then the timing report against naive C and NumPy."""
    kernel_dir, is_fixture = resolve(kernel)
    locked = guard(kernel, is_fixture, unlocked)
    result = _checked(kernel, kernel_dir)
    log.info("")
    report = run_bench(result.built, quick, locked)
    report["correctness"] = {"cases": result.cases, "fresh_seed": result.fresh_seed}
    log_bench(report)
    save_run(report, record)


def _build_old_version(built: Built, revision: str) -> Path:
    """Build the kernel as it was at an earlier git revision.

    Exits with a message when the kernel did not exist then, or no longer
    builds or passes lint.
    """
    k = built.k
    relative = built.asm.relative_to(ROOT).as_posix()
    shown = _git("show", f"{revision}:{relative}")
    if shown.returncode != 0:
        log.error("\ncannot compare: %s does not exist at %s", relative, revision)
        raise typer.Exit(EXIT_REFUSED)
    source = BUILD / k.name / "old.s"
    source.write_text(shown.stdout)
    try:
        dylib, issues = build_asm(k, source, OLD)
    except BuildError as e:
        log.error("\ncannot compare: the version at %s does not build: %s", revision, e)
        raise typer.Exit(EXIT_REFUSED)
    if issues:
        log.error("\ncannot compare: the version at %s fails lint: %s", revision, issues[0])
        raise typer.Exit(EXIT_REFUSED)
    return dylib


def _compare_verdict(overall: dict) -> str:
    """Describe the overall difference as one of three outcomes, with its interval."""
    if not overall["significant"]:
        outcome = "no significant difference"
    elif overall["speedup"] > 1:
        outcome = "significantly faster"
    else:
        outcome = "significantly slower"
    return (f"{outcome} ({overall['speedup']:.3f}x, "
            f"95% interval {overall['lo']:.3f} – {overall['hi']:.3f})")


@app.command()
def compare(
    kernel: str = KernelArg,
    against: str = typer.Option(..., help="Git revision to compare with, for example HEAD."),
    quick: bool = QuickOpt,
    unlocked: bool = UnlockedOpt,
) -> None:
    """Is the working copy of a kernel faster than an earlier version of it?

    Both versions are timed interleaved in one process, which detects much
    smaller differences than comparing against numbers recorded earlier.
    """
    kernel_dir, is_fixture = resolve(kernel)
    guard(kernel, is_fixture, unlocked)
    built = _checked(kernel, kernel_dir).built
    k = built.k
    old_dylib = _build_old_version(built, against)

    harness = B.Harness(k, built.harness)
    new_fn, _new_lib = B.function_address(built.kernel, k.name)
    old_fn, _old_lib = B.function_address(old_dylib, k.name)
    sizes, samples = _sweep(quick)
    canary_before = harness.canary()
    data = B.measure(k, harness, {NEW: new_fn, OLD: old_fn}, sizes, samples, _log_progress)
    canary_after = harness.canary()

    report = B.analyse(data, NEW, [OLD])
    log.info("\n" + render(
        "timing.txt.j2",
        title=f"{k.name} ({ARCH}): working copy vs {against}",
        warnings=B.noise_warnings(canary_before, canary_after),
        report=report, kernel=NEW, baselines=[OLD], labels=_COMPARE_LABELS, notes=[],
        verdict=_compare_verdict(report["overall"][f"vs_{OLD}"]),
    ).rstrip())


@app.command()
def selftest(
    no_speed: bool = typer.Option(False, "--no-speed", help="Skip the timing fixture, for CI."),
    stability: bool = typer.Option(
        False, "--stability", help="Repeat the timing sweep and report how much it varies."),
) -> None:
    """Prove the judge catches kernels that are wrong in known ways."""
    passed = S.stability() if stability else S.selftest(speed=not no_speed)
    if not passed:
        raise typer.Exit(EXIT_FAILED)


@app.command()
def leaderboard() -> None:
    """Regenerate LEADERBOARD.md from the recorded results."""
    results = [json.loads(path.read_text()) for path in sorted(RESULTS.glob("*/*.json"))]
    LEADERBOARD.write_text(render("leaderboard.md.j2", results=results, numpy_note=NUMPY_NOTE))
    log.info("wrote %s (%d results)", LEADERBOARD.name, len(results))


if __name__ == "__main__":
    app()
