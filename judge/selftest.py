"""Proves the judge catches kernels that are wrong in known ways.

Each fixture kernel has a correct version and several deliberately broken
variants. The selftest passes only if the correct versions pass and every
broken variant is caught for the reason it was built to trigger.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from . import bench as B
from .build import ARCH, ARM64, X86_64
from .check import KERNEL, LINT, NAIVE, CheckResult, build_all, check
from .render import render
from .worker import CALLING_CONVENTION, MEMORY_SAFETY, WRONG_RESULT

log = logging.getLogger(__name__)

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Fixture kernel -> broken variant -> the failure category that must catch it.
EXPECT = {
    "argmax_f32": {
        "skip_tail": WRONG_RESULT,
        "ignore_nan": WRONG_RESULT,
        "last_tie": WRONG_RESULT,
        "aligned_only": WRONG_RESULT,
        "overread": MEMORY_SAFETY,
        "clobber_register": CALLING_CONVENTION,
    },
    "compress_f32": {
        "write_past_used": MEMORY_SAFETY,
        "write_input": MEMORY_SAFETY,
    },
    # The same operation with a scratch output buffer. Its correct kernel and
    # its naive C both leave a stray element past the reported output, which is
    # exactly what compress_f32/write_past_used is rejected for.
    "compress_scratch_f32": {
        "write_past_capacity": MEMORY_SAFETY,
    },
    # A kernel with a scalar argument that also sizes its output. Both broken
    # variants are right for the default scalar value and wrong for another,
    # so they are caught only if the other scalar sets are really run.
    "tally_u8": {
        "fixed_modulus": WRONG_RESULT,
        "clears_default_capacity": MEMORY_SAFETY,
    },
}
# Broken variants that exist for one architecture only, because the rule they
# break does: architecture -> fixture kernel -> variant -> failure category.
_ARCH_EXPECT = {
    # macOS reserves x18.
    ARM64: {"argmax_f32": {"uses_x18": LINT}},
    # x86-64 kernels are held to an instruction-set level, and must clear the
    # upper halves of the vector registers before returning.
    X86_64: {"argmax_f32": {"beyond_level": LINT, "no_vzeroupper": CALLING_CONVENTION}},
}
# A variant that is correct but slow: it must pass correctness and then be
# reported as significantly slower than naive C.
SLOW_KERNEL, SLOW_VARIANT = "argmax_f32", "slow"
# The slow variant is timed on a reduced sweep; the gap is large enough.
_SLOW_SIZES = [1024, 65536]
_SLOW_SAMPLES = 15

# The stability check repeats the full timing sweep this many times ...
STABILITY_RUNS = 10
# ... and passes when no size varies by more than this between runs.
STABILITY_LIMIT = 0.03
STABILITY_KERNEL = "argmax_f32"


@dataclass
class Outcome:
    """The result of one selftest step."""

    # What was tested, such as "argmax_f32/skip_tail".
    label: str
    # True when the judge behaved as required.
    ok: bool
    # One-line explanation of what happened.
    detail: str


def _variant_path(name: str, variant: str) -> Path:
    """Return the assembly source of a broken variant of a fixture kernel."""
    return FIXTURES / name / ARCH / "broken" / f"{variant}.s"


def _describe_failure(result: CheckResult) -> str:
    """Summarise a check failure in one line."""
    f = result.failure
    return f"{f.category}: {f.message}"


def _correct_outcome(name: str) -> Outcome:
    """Check that the correct version of a fixture kernel passes."""
    result = check(FIXTURES / name)
    if result.ok:
        return Outcome(f"{name} (correct)", True, f"passed {result.cases:,} cases")
    return Outcome(f"{name} (correct)", False,
                   f"unexpectedly failed: {_describe_failure(result)} [{result.failure.case}]")


def _broken_outcome(name: str, variant: str, expected: str) -> Outcome:
    """Check that a broken variant is rejected, and for the expected reason."""
    label = f"{name}/{variant}"
    result = check(FIXTURES / name, asm=_variant_path(name, variant), tag=f"broken-{variant}")
    if result.ok:
        return Outcome(label, False, f"was NOT caught (expected: {expected})")
    caught_correctly = result.failure.category == expected and result.failure.who == KERNEL
    if caught_correctly:
        return Outcome(label, True, f"caught as {_describe_failure(result)}")
    return Outcome(label, False, f"caught as {_describe_failure(result)}, expected {expected}")


def _slow_outcomes(speed: bool) -> Iterator[Outcome]:
    """Check the slow variant: it must pass correctness, then be reported as slower."""
    label = f"{SLOW_KERNEL}/{SLOW_VARIANT}"
    result = check(FIXTURES / SLOW_KERNEL, asm=_variant_path(SLOW_KERNEL, SLOW_VARIANT),
                   tag=f"broken-{SLOW_VARIANT}")
    if not result.ok:
        yield Outcome(f"{label} (correctness)", False,
                      f"unexpectedly failed: {_describe_failure(result)}")
        return
    yield Outcome(f"{label} (correctness)", True, "passes correctness, as intended")
    if not speed:
        return

    built = result.built
    harness = B.Harness(built.k, built.harness)
    kernel_fn, _kernel_lib = B.function_address(built.kernel, built.k.name)
    naive_fn, _naive_lib = B.function_address(built.naive, built.k.name)
    data = B.measure(built.k, harness, {KERNEL: kernel_fn, NAIVE: naive_fn},
                     sizes=_SLOW_SIZES, samples=_SLOW_SAMPLES)
    overall = B.analyse(data, KERNEL, [NAIVE])["overall"][f"vs_{NAIVE}"]
    reported_slower = overall["significant"] and overall["speedup"] < 1
    verdict = "significantly slower" if reported_slower else "NOT significantly slower"
    yield Outcome(f"{label} (speed)", reported_slower,
                  f"reported as {verdict}: {overall['speedup']:.2f}x "
                  f"[{overall['lo']:.2f} – {overall['hi']:.2f}]")


def _expected() -> dict[str, dict[str, str]]:
    """Return every broken variant this architecture has.

    Returns:
        Fixture kernel -> variant -> the failure category that must catch it:
        the variants every architecture shares, then this one's own.
    """
    own = _ARCH_EXPECT.get(ARCH, {})
    return {name: {**variants, **own.get(name, {})} for name, variants in EXPECT.items()}


def _outcomes(speed: bool) -> Iterator[Outcome]:
    """Yield the outcome of every selftest step, in order."""
    for name, variants in _expected().items():
        yield _correct_outcome(name)
        for variant, expected in variants.items():
            yield _broken_outcome(name, variant, expected)
    yield from _slow_outcomes(speed)


def selftest(speed: bool = True) -> bool:
    """Run every selftest step and log the results. Returns True when all passed.

    With `speed` off, the slow variant is checked for correctness only; shared
    CI runners are too noisy to time anything.
    """
    log.info("judge selftest (%s)%s\n", ARCH, "" if speed else ", timing skipped")
    outcomes = list(_outcomes(speed))
    failed = sum(not outcome.ok for outcome in outcomes)
    log.info(render("selftest.txt.j2", outcomes=outcomes, failed=failed).rstrip())
    return failed == 0


def stability(runs: int = STABILITY_RUNS) -> bool:
    """Repeat the timing sweep on a correct fixture and log how much it varies.

    Returns True when, at every size, the slowest run's median is within
    STABILITY_LIMIT of the fastest run's.
    """
    built, failure, k = build_all(FIXTURES / STABILITY_KERNEL)
    if failure:
        log.error("build failed: %s", failure.message)
        return False
    harness = B.Harness(k, built.harness)
    kernel_fn, _kernel_lib = B.function_address(built.kernel, k.name)

    medians: dict[int, list[float]] = {n: [] for n in B.SIZES}
    for run in range(1, runs + 1):
        data = B.measure(k, harness, {KERNEL: kernel_fn})
        for n, times in data[B.DEFAULT_DISTRIBUTION].items():
            medians[n].append(float(np.median(times[KERNEL])))
        log.info("  run %d/%d done", run, runs)

    rows = [{"n": n, "fastest": min(values), "slowest": max(values),
             "spread": max(values) / min(values) - 1} for n, values in medians.items()]
    worst = max(row["spread"] for row in rows)
    ok = worst <= STABILITY_LIMIT
    log.info("\n" + render("stability.txt.j2", runs=runs, name=k.name, arch=ARCH, rows=rows,
                           worst=worst, ok=ok, limit=STABILITY_LIMIT).rstrip())
    return ok
