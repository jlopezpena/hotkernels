"""Command-line entry point: uv run judge <command>."""
from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import bench as B
from .build import ARCH, BUILD, ROOT, BuildError, build_asm
from .check import Built, CheckResult, build_all, check, format_result

KERNELS = ROOT / "kernels"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
RESULTS = ROOT / "results"
RUNS = ROOT / ".judge" / "runs"

NUMPY_NOTE = (
    "NumPy is timed from Python and includes its per-call overhead; the kernel is timed "
    "without a binding.\nAt small sizes the NumPy column flatters the kernel. Naive C is the "
    "honest baseline."
)


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def resolve(name: str) -> tuple[Path, bool]:
    """Kernel directory and whether it is a judge fixture."""
    if (KERNELS / name / "spec.py").exists():
        return KERNELS / name, False
    if (FIXTURES / name / "spec.py").exists():
        return FIXTURES / name, True
    sys.exit(f"unknown kernel {name!r}: no kernels/{name}/spec.py")


def locked_paths(name: str) -> list[str]:
    return ["judge", "pyproject.toml", "uv.lock", f"kernels/{name}/spec.py", f"kernels/{name}/naive.c"]


def reference_changes(name: str) -> list[str]:
    """Files an agent must not touch that differ from the main branch."""
    if _git("rev-parse", "--verify", "--quiet", "main").returncode != 0:
        return []
    paths = locked_paths(name)
    changed = _git("diff", "--name-only", "main", "--", *paths).stdout.split()
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", *paths).stdout.split()
    return sorted(set(changed + untracked))


def guard(name: str, is_fixture: bool, unlocked: bool) -> bool:
    """Returns True when the reference files are verified unchanged."""
    if is_fixture:
        return True
    changed = reference_changes(name)
    if not changed:
        return True
    if unlocked:
        print("UNLOCKED: these reference files differ from main; results are not trustworthy "
              "for an agent run:")
        for path in changed:
            print(f"  {path}")
        print()
        return False
    print(f"FAIL {name} ({ARCH}): reference files modified")
    print("  The spec, the naive reference and the judge must match the main branch.")
    for path in changed:
        print(f"  changed: {path}")
    print("  Only kernel.s and attempts.md may be edited during a kernel run.")
    sys.exit(2)


def machine() -> str:
    if platform.system() == "Darwin":
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True).stdout.strip()
        if out:
            return out
    return platform.processor() or platform.machine()


def environment(locked: bool) -> dict:
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


def _progress(dist: str, n: int) -> None:
    print(f"  timed {dist} n={n:,}", file=sys.stderr, flush=True)


def run_bench(built: Built, quick: bool, locked: bool) -> dict:
    k = built.k
    harness = B.Harness(k, built.harness)
    kernel_fn, _keep1 = B.function_address(built.kernel, k.name)
    naive_fn, _keep2 = B.function_address(built.naive, k.name)
    sizes = [1024, 65536] if quick else B.SIZES
    samples = 10 if quick else B.SAMPLES

    before = harness.canary()
    data = B.measure(k, harness, {"kernel": kernel_fn, "naive": naive_fn, "numpy": None},
                     sizes, samples, _progress)
    after = harness.canary()

    report = B.analyse(data, "kernel", ["naive", "numpy"])
    report.update(kernel=k.name, samples=samples, warnings=B.noise_warnings(before, after),
                  environment=environment(locked))
    return report


def print_bench(report: dict) -> None:
    env = report["environment"]
    print(f"{report['kernel']} ({env['arch']}, {env['machine']}), NumPy {env['numpy']}, "
          f"{report['samples']} samples per point\n")
    for warning in report["warnings"]:
        print(f"*** {warning}\n")
    labels = {"kernel": "kernel", "naive": "naive C", "numpy": "NumPy*"}
    print(B.format_table(report, "kernel", ["naive", "numpy"], labels))
    print(f"* {NUMPY_NOTE}")


def save_run(report: dict, record: bool) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    env = report["environment"]
    path = RUNS / f"{report['kernel']}-{env['arch']}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nfull report: {path.relative_to(ROOT)}")
    if record:
        slug = re.sub(r"[^a-z0-9]+", "-", env["machine"].lower()).strip("-")
        target = RESULTS / report["kernel"] / f"{env['arch']}-{slug}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2) + "\n")
        print(f"recorded:    {target.relative_to(ROOT)}")


def cmd_check(args) -> int:
    kernel_dir, is_fixture = resolve(args.kernel)
    guard(args.kernel, is_fixture, args.unlocked)
    result = check(kernel_dir, case=args.case)
    print(format_result(result, args.kernel))
    return 0 if result.ok else 1


def cmd_bench(args) -> int:
    kernel_dir, is_fixture = resolve(args.kernel)
    locked = guard(args.kernel, is_fixture, args.unlocked)
    built, failure, k = build_all(kernel_dir)
    if failure:
        print(format_result(CheckResult(k.name, False, failure), args.kernel))
        return 1
    report = run_bench(built, args.quick, locked)
    print_bench(report)
    save_run(report, record=False)
    return 0


def cmd_run(args) -> int:
    kernel_dir, is_fixture = resolve(args.kernel)
    locked = guard(args.kernel, is_fixture, args.unlocked)
    result = check(kernel_dir)
    print(format_result(result, args.kernel))
    if not result.ok:
        return 1
    print()
    report = run_bench(result.built, args.quick, locked)
    report["correctness"] = {"cases": result.cases, "fresh_seed": result.fresh_seed}
    print_bench(report)
    save_run(report, args.record)
    return 0


def cmd_compare(args) -> int:
    kernel_dir, is_fixture = resolve(args.kernel)
    guard(args.kernel, is_fixture, args.unlocked)
    result = check(kernel_dir)
    print(format_result(result, args.kernel))
    if not result.ok:
        return 1
    built, k = result.built, result.built.k

    rel = built.asm.relative_to(ROOT)
    shown = _git("show", f"{args.against}:{rel.as_posix()}")
    if shown.returncode != 0:
        print(f"\ncannot compare: {rel} does not exist at {args.against}")
        return 2
    old_src = BUILD / k.name / "old.s"
    old_src.write_text(shown.stdout)
    try:
        old_dylib, issues = build_asm(k, old_src, "old")
    except BuildError as e:
        print(f"\ncannot compare: the version at {args.against} does not build: {e}")
        return 2
    if issues:
        print(f"\ncannot compare: the version at {args.against} fails lint: {issues[0]}")
        return 2

    harness = B.Harness(k, built.harness)
    new_fn, _keep1 = B.function_address(built.kernel, k.name)
    old_fn, _keep2 = B.function_address(old_dylib, k.name)
    sizes = [1024, 65536] if args.quick else B.SIZES
    samples = 10 if args.quick else B.SAMPLES
    before = harness.canary()
    data = B.measure(k, harness, {"new": new_fn, "old": old_fn}, sizes, samples, _progress)
    after = harness.canary()
    report = B.analyse(data, "new", ["old"])

    print(f"\n{k.name} ({ARCH}): working copy vs {args.against}\n")
    for warning in B.noise_warnings(before, after):
        print(f"*** {warning}\n")
    print(B.format_table(report, "new", ["old"], {"new": "new", "old": "old"}))
    overall = report["overall"]["vs_old"]
    if not overall["significant"]:
        verdict = "no significant difference"
    elif overall["speedup"] > 1:
        verdict = "significantly faster"
    else:
        verdict = "significantly slower"
    print(f"\nRESULT: {verdict} "
          f"({overall['speedup']:.3f}x, 95% interval {overall['lo']:.3f} – {overall['hi']:.3f})")
    return 0


def cmd_selftest(args) -> int:
    from .selftest import selftest, stability
    if args.stability:
        return stability()
    return selftest(speed=not args.no_speed)


def cmd_leaderboard(args) -> int:
    rows = []
    for path in sorted(RESULTS.glob("*/*.json")):
        r = json.loads(path.read_text())
        env, overall = r["environment"], r["overall"]

        def cell(s):
            return f"{s['speedup']:.2f}x ({s['lo']:.2f} – {s['hi']:.2f})"

        rows.append(f"| `{r['kernel']}` | {env['arch']} | {env['machine']} | "
                    f"{cell(overall['vs_naive'])} | {cell(overall['vs_numpy'])} | "
                    f"{env['numpy']} | {env['timestamp'][:10]} |")
    lines = [
        "# Leaderboard",
        "",
        "Generated by `uv run judge leaderboard` from the files in `results/`. Do not edit.",
        "",
        "Speedups are geometric means across array sizes, with 95% confidence intervals.",
        NUMPY_NOTE.replace("\n", " "),
        "",
        "| Kernel | Arch | Machine | vs naive C | vs NumPy | NumPy | Date |",
        "|---|---|---|---|---|---|---|",
        *(rows or ["| _no recorded results yet_ | | | | | | |"]),
        "",
    ]
    (ROOT / "LEADERBOARD.md").write_text("\n".join(lines))
    print(f"wrote LEADERBOARD.md ({len(rows)} results)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="judge", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def kernel_command(name, func, help_text):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("kernel")
        p.add_argument("--unlocked", action="store_true",
                       help="allow modified reference files (for people creating a kernel or "
                            "changing the judge; never for agent runs)")
        p.set_defaults(func=func)
        return p

    p = kernel_command("check", cmd_check, "lint and correctness only")
    p.add_argument("--case", help="run a single case, as printed by a failure")

    p = kernel_command("bench", cmd_bench, "timing report only")
    p.add_argument("--quick", action="store_true", help="two sizes, few samples (smoke test)")

    p = kernel_command("run", cmd_run, "correctness, then the timing report")
    p.add_argument("--quick", action="store_true", help="two sizes, few samples (smoke test)")
    p.add_argument("--record", action="store_true", help="also save the report under results/")

    p = kernel_command("compare", cmd_compare, "is the working copy faster than an earlier version?")
    p.add_argument("--against", required=True, help="git revision to compare with, e.g. HEAD")
    p.add_argument("--quick", action="store_true", help="two sizes, few samples (smoke test)")

    p = sub.add_parser("selftest", help="prove the judge catches known-bad kernels")
    p.add_argument("--no-speed", action="store_true", help="skip the timing fixture (for CI)")
    p.add_argument("--stability", action="store_true",
                   help="run the timing ten times and report how much it varies")
    p.set_defaults(func=cmd_selftest)

    p = sub.add_parser("leaderboard", help="regenerate LEADERBOARD.md from results/")
    p.set_defaults(func=cmd_leaderboard)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
