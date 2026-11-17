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
from . import placement as P
from .build import ARCH, ARM64, X86_64
from .cases import TIE_DISTINCT_VALUES, build_inputs, fixed_cases, repeated_class, sizes
from .check import KERNEL, LINT, NAIVE, CheckResult, build_all, check
from .render import render
from .spec import Out, drawn_class, load_spec
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
    # A kernel with an input whose length is a scalar. One variant reads a
    # byte past that input, which only a buffer placed by the scalar's value
    # catches. The other is right for the default length alone.
    "rank_u8": {
        "needle_overread": MEMORY_SAFETY,
        "fixed_count": WRONG_RESULT,
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

# The fixture whose timing buffers are inspected: it has two inputs and an
# output, and the sizes cover a pool of many small inputs and single large ones.
PLACEMENT_KERNEL = "compress_f32"
_PLACEMENT_SIZES = [16, 1024, 65536, 1048576]

# The stability check repeats the full timing sweep this many times ...
STABILITY_RUNS = 10
# ... and passes when no size varies by more than this between runs.
STABILITY_LIMIT = 0.03
STABILITY_KERNEL = "argmax_f32"


# The fixture with an argument drawn from another, whose cases are inspected.
DRAWN_KERNEL = "rank_u8"


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


def _misplaced(placement: str) -> list[str]:
    """Find timing outputs that are not where the judge says it puts them.

    Args:
        placement: P.APART or P.CLOSE.

    Returns:
        One description per output buffer of the placement fixture that is
        out of place: for P.APART, not clear of an input of its own call; for
        P.CLOSE, not P.CLOSE_BYTES after the input it is placed against.
        Empty when every buffer is in place.
    """
    built, _failure, k = build_all(FIXTURES / PLACEMENT_KERNEL)
    harness = B.Harness(k, built.harness)
    naive, _naive_lib = B.function_address(built.naive, k.name)
    found = []
    for n in _PLACEMENT_SIZES:
        pool, outs = B._timing_setup(k, harness, {NAIVE: naive}, B.DEFAULT_DISTRIBUTION, n,
                                     placement=placement).buffers
        for entry, made in zip(pool, outs):
            inputs = [entry[a.name] for a in k.ins]
            starts = [array.ctypes.data for array in inputs]
            for a in (a for a in k.outs if isinstance(a, Out)):
                address = made[a.name].ctypes.data
                if placement == P.APART:
                    in_place = P.clear_of(address, starts)
                else:
                    anchor = P.anchor_of(made[a.name].size, inputs)
                    in_place = P.gaps(address, [anchor]) == [P.CLOSE_BYTES]
                if not in_place:
                    found.append(f"{a.name} at n={n:,}: {P.gaps(address, starts)} bytes after its inputs")
    return found


def _placement_outcomes() -> Iterator[Outcome]:
    """Check that timing outputs sit where the judge says, in both placements."""
    wanted = {P.APART: "every output is clear of its inputs",
              P.CLOSE: f"every output is {P.CLOSE_BYTES} bytes behind its input"}
    for placement, description in wanted.items():
        found = _misplaced(placement)
        yield Outcome(f"timing buffers ({placement})", not found, found[0] if found else description)


def _missing_drawn_cases(name: str) -> list[str]:
    """Find the sizes and scalar sets at which a fixture lacks the drawn cases it is promised.

    At every size the fixture is checked at, in every scalar set, there must
    be a case in which each argument with `draws_from` takes its drawn class
    while the source takes its class of repeated values; the drawn array must
    hold only elements of the source as the kernel receives it, and the
    source must repeat values wherever it has more elements than that class
    has distinct values.

    Args:
        name: A fixture kernel with at least one argument that draws from another.

    Returns:
        One description per size, scalar set and argument that falls short.
        Empty when nothing does.
    """
    k = load_spec(FIXTURES / name)
    cases = list(fixed_cases(k))
    found = []
    for a in (a for a in k.ins if a.draws_from):
        source = next(s for s in k.ins if s.name == a.draws_from)
        repeated = repeated_class(source.dtype)
        wanted = {(a.name, drawn_class(a)), (source.name, repeated)}
        for n in sizes(k.min_n, k.check_max_n):
            for scalars in k.scalar_set_names:
                match = [c for c in cases if c.n == n and c.scalars == scalars
                         and wanted <= {(arg, cls) for arg, cls, _p in c.picks}]
                where = f"{a.name} at n={n:,}, scalars={scalars}"
                if not match:
                    found.append(f"{where}: no case drawn from {source.name}={repeated}")
                    continue
                inputs = build_inputs(k, match[0])
                drawn, values = inputs[a.name], inputs[source.name]
                if values.size and not np.isin(drawn, values).all():
                    found.append(f"{where}: values that are not elements of {source.name}")
                if values.size > TIE_DISTINCT_VALUES and np.unique(values).size == values.size:
                    found.append(f"{where}: {source.name} has no repeated value")
    return found


def _drawn_outcomes() -> Iterator[Outcome]:
    """Check that values drawn from another argument are generated where promised."""
    found = _missing_drawn_cases(DRAWN_KERNEL)
    yield Outcome(f"{DRAWN_KERNEL} (drawn values)", not found,
                  found[0] if found else "drawn from repeated values at every size and scalar set")


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
    yield from _drawn_outcomes()
    yield from _placement_outcomes()
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
