"""Compiles kernels, naive references and the timing harness.

The judge owns every compiler flag. A kernel directory cannot change how its
naive reference is built, so the baseline cannot be weakened.
"""
from __future__ import annotations

import platform
import subprocess
from pathlib import Path

from .render import render
from .spec import DTYPES, FLOAT_SCALARS, RETURNS, SCALARS, In, InOut, Kernel, Out, Scalar, Size

ROOT = Path(__file__).resolve().parent.parent
C_DIR = Path(__file__).resolve().parent / "c"
# Compiled libraries and generated sources. Not committed.
BUILD = ROOT / ".judge" / "build"

# Normalise the names different systems use for the same architecture.
_ARCH_ALIASES = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "x86_64", "AMD64": "x86_64"}
ARCH = _ARCH_ALIASES.get(platform.machine(), platform.machine())

# Tune for the processor the judge is running on.
_NATIVE_FLAG = "-mcpu=native" if ARCH == "arm64" else "-march=native"
# The naive reference is always built with exactly these flags.
C_FLAGS = ["-O3", _NATIVE_FLAG, "-std=c11", "-Wall"]

# `nm -g` prints "U _name" for a symbol the object needs from elsewhere, and
# "<address> <type> _name" for one it defines.
_NM_UNDEFINED_FIELDS = 2
_NM_DEFINED_FIELDS = 3
# In `otool -l` output, a section that lives in the writable data segment.
_WRITABLE_SEGMENT_MARKER = "segname __DATA"


class BuildError(Exception):
    """A compiler, assembler or linker step failed. The message is the tool's output."""


def _run(cmd: list) -> str:
    """Run a build tool and return its standard output, raising BuildError on failure."""
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if proc.returncode != 0:
        raise BuildError((proc.stderr or proc.stdout).strip())
    return proc.stdout


def _c_type(arg: In | Out | InOut | Size | Scalar) -> str:
    """Return the C type of one kernel argument."""
    if isinstance(arg, In):
        return f"const {DTYPES[arg.dtype][1]} *"
    if isinstance(arg, (Out, InOut)):
        return f"{DTYPES[arg.dtype][1]} *"
    if isinstance(arg, Scalar):
        return SCALARS[arg.dtype]
    return "size_t"


def _c_literal(scalar: Scalar) -> str:
    """Return a scalar argument's value as a C literal of its type.

    Floats are written in hexadecimal notation, which reproduces the value
    exactly; a decimal literal could round.
    """
    if scalar.dtype in FLOAT_SCALARS:
        suffix = "f" if scalar.dtype == "f32" else ""
        return f"{float(scalar.value).hex()}{suffix}"
    return f"({SCALARS[scalar.dtype]}){int(scalar.value)}"


def _call_arg(arg: In | Out | InOut | Size | Scalar) -> str:
    """Return the C expression the timing loop passes for one kernel argument."""
    if isinstance(arg, Size):
        return "(size_t)n"
    if isinstance(arg, Scalar):
        return _c_literal(arg)
    return f"({_c_type(arg)})p_{arg.name}[p]"


def c_prototype(k: Kernel) -> str:
    """Return the C prototype the spec implies, without a trailing semicolon."""
    params = [f"{_c_type(a)}{'' if _c_type(a).endswith('*') else ' '}{a.name}" for a in k.args]
    return f"{RETURNS[k.returns]} {k.name}({', '.join(params)})"


def _build_dir(k: Kernel) -> Path:
    """Return the kernel's build directory, creating it if needed."""
    d = BUILD / k.name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _object_issues(k: Kernel, obj: Path) -> list[str]:
    """Check an assembled object file against the rules a kernel must obey.

    A kernel must export exactly one symbol (itself), need nothing from outside,
    and contain no writable data.
    """
    issues: list[str] = []
    wanted = f"_{k.name}"
    defined: set[str] = set()
    for line in _run(["nm", "-g", obj]).splitlines():
        fields = line.split()
        if len(fields) == _NM_UNDEFINED_FIELDS and fields[0] == "U":
            issues.append(f"references external symbol {fields[1]}")
        elif len(fields) == _NM_DEFINED_FIELDS:
            defined.add(fields[2])
    if wanted not in defined:
        issues.append(f"does not export {wanted}")
    for extra in sorted(defined - {wanted}):
        issues.append(f"exports extra symbol {extra}")
    if _WRITABLE_SEGMENT_MARKER in _run(["otool", "-l", obj]):
        issues.append("declares writable data")
    return issues


def build_asm(k: Kernel, src: Path, tag: str = "kernel") -> tuple[Path | None, list[str]]:
    """Assemble a kernel into a shared library.

    `tag` names the output files, so several versions of one kernel can be built
    side by side. Returns (library, issues). When the object breaks a rule the
    library is not linked and is returned as None.
    """
    d = _build_dir(k)
    obj, dylib = d / f"{tag}.o", d / f"{tag}.dylib"
    _run(["clang", "-c", src, "-o", obj])
    issues = _object_issues(k, obj)
    if issues:
        return None, issues
    _run(["clang", "-shared", obj, "-o", dylib])
    return dylib, []


def build_naive(k: Kernel, src: Path) -> Path:
    """Compile the naive C reference into a shared library with the fixed flags.

    A generated header with the prototype from the spec is force-included, so a
    `naive.c` whose signature disagrees with the spec fails to compile.
    """
    d = _build_dir(k)
    header = d / "kernel.h"
    header.write_text(render("kernel.h.j2", prototype=c_prototype(k)))
    dylib = d / "naive.dylib"
    _run(["clang", *C_FLAGS, "-shared", "-include", header, src, "-o", dylib])
    return dylib


def _bench_source(k: Kernel) -> str:
    """Generate the C timing loop for this kernel's signature."""
    return render(
        "bench.c.j2",
        ret=RETURNS[k.returns],
        arg_types=[_c_type(a) for a in k.args],
        pointer_names=[a.name for a in k.buffers],
        call_args=[_call_arg(a) for a in k.args],
    )


def build_harness(k: Kernel) -> Path:
    """Build the per-kernel harness library.

    It contains the generated timing loop, the shared timing helpers, and the
    architecture-specific wrapper that checks the calling convention.
    """
    abi = C_DIR / f"abi_{ARCH}.s"
    if not abi.exists():
        raise BuildError(f"the judge does not support the {ARCH} architecture yet")
    d = _build_dir(k)
    bench_c = d / "bench.c"
    bench_c.write_text(_bench_source(k))
    dylib = d / "harness.dylib"
    _run(["clang", *C_FLAGS, "-shared", bench_c, C_DIR / "harness.c", abi, "-o", dylib])
    return dylib
