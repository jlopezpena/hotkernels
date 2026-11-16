"""Proves the judge catches kernels that are wrong in known ways."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import bench as B
from .build import ARCH
from .check import build_all, check

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# fixture kernel -> broken variant -> how the judge must catch it
EXPECT = {
    "argmax_f32": {
        "skip_tail": "wrong result",
        "ignore_nan": "wrong result",
        "last_tie": "wrong result",
        "aligned_only": "wrong result",
        "overread": "memory safety",
        "clobber_register": "calling convention",
        "uses_x18": "lint",
    },
    "compress_f32": {
        "write_past_used": "memory safety",
        "write_input": "memory safety",
    },
}
SLOW = ("argmax_f32", "slow")  # correct, but must be reported as significantly slower
STABILITY_LIMIT = 0.03


def selftest(speed: bool = True) -> int:
    failures = 0

    def report(ok: bool, label: str, detail: str) -> None:
        nonlocal failures
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {label:<34} {detail}")

    print(f"judge selftest ({ARCH})\n")
    for name, variants in EXPECT.items():
        kernel_dir = FIXTURES / name
        result = check(kernel_dir)
        detail = f"passed {result.cases:,} cases" if result.ok else (
            f"unexpectedly failed: {result.failure.category}: {result.failure.message}"
            f" [{result.failure.case}]")
        report(result.ok, f"{name} (correct)", detail)

        for variant, expected in variants.items():
            asm = kernel_dir / ARCH / "broken" / f"{variant}.s"
            result = check(kernel_dir, asm=asm, tag=f"broken-{variant}")
            if result.ok:
                report(False, f"{name}/{variant}", f"was NOT caught (expected: {expected})")
                continue
            f = result.failure
            ok = f.category == expected and f.who == "kernel"
            report(ok, f"{name}/{variant}",
                   f"caught as {f.category}: {f.message}" if ok
                   else f"caught as {f.category} ({f.message}), expected {expected}")

    name, variant = SLOW
    kernel_dir = FIXTURES / name
    asm = kernel_dir / ARCH / "broken" / f"{variant}.s"
    result = check(kernel_dir, asm=asm, tag=f"broken-{variant}")
    report(result.ok, f"{name}/{variant} (correctness)",
           "passes correctness, as intended" if result.ok
           else f"unexpectedly failed: {result.failure.category}: {result.failure.message}")
    if speed and result.ok:
        built = result.built
        harness = B.Harness(built.k, built.harness)
        kernel_fn, _keep1 = B.function_address(built.kernel, built.k.name)
        naive_fn, _keep2 = B.function_address(built.naive, built.k.name)
        data = B.measure(built.k, harness, {"kernel": kernel_fn, "naive": naive_fn},
                         sizes=[1024, 65536], samples=15)
        overall = B.analyse(data, "kernel", ["naive"])["overall"]["vs_naive"]
        ok = overall["significant"] and overall["speedup"] < 1
        report(ok, f"{name}/{variant} (speed)",
               f"reported as {'significantly slower' if ok else 'NOT significantly slower'}: "
               f"{overall['speedup']:.2f}x [{overall['lo']:.2f} – {overall['hi']:.2f}]")
    elif not speed:
        print(f"  skip  {name}/{variant} (speed){'':<13} timing skipped")

    print(f"\n{'selftest passed' if not failures else f'selftest FAILED: {failures} problem(s)'}")
    return 1 if failures else 0


def stability(runs: int = 10) -> int:
    """Times the correct argmax fixture repeatedly and reports the spread per size."""
    kernel_dir = FIXTURES / "argmax_f32"
    built, failure, k = build_all(kernel_dir)
    if failure:
        print(f"build failed: {failure.message}")
        return 1
    harness = B.Harness(k, built.harness)
    kernel_fn, _keep = B.function_address(built.kernel, k.name)
    medians: dict[int, list[float]] = {n: [] for n in B.SIZES}
    for run in range(runs):
        data = B.measure(k, harness, {"kernel": kernel_fn})
        for n, times in data["random"].items():
            medians[n].append(float(np.median(times["kernel"])))
        print(f"  run {run + 1}/{runs} done", flush=True)

    print(f"\ntiming stability over {runs} runs ({k.name}, {ARCH})\n")
    print(f"{'n':>12}  {'fastest':>10}  {'slowest':>10}  {'spread':>7}")
    worst = 0.0
    for n, values in medians.items():
        spread = max(values) / min(values) - 1
        worst = max(worst, spread)
        print(f"{n:>12,}  {B.fmt_ns(min(values)):>10}  {B.fmt_ns(max(values)):>10}  {spread:>6.1%}")
    ok = worst <= STABILITY_LIMIT
    print(f"\nworst spread {worst:.1%} "
          f"({'within' if ok else 'OUTSIDE'} the {STABILITY_LIMIT:.0%} target)")
    return 0 if ok else 1
