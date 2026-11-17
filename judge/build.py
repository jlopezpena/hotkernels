"""Compiles kernels, naive references and the timing harness.

The judge owns every compiler flag. A kernel directory cannot change how its
naive reference is built, so the baseline cannot be weakened.

Two platforms are supported: ARM64 on macOS and x86-64 on Linux. What differs
between them is collected in the constants below: the object format, how a
kernel is assembled, and how an object file is inspected.
"""
from __future__ import annotations

import platform
import re
import subprocess
from pathlib import Path

from .render import render
from .spec import DTYPES, RETURNS, SCALARS, In, InOut, Kernel, Out, Scalar, Size

ROOT = Path(__file__).resolve().parent.parent
C_DIR = Path(__file__).resolve().parent / "c"
# Compiled libraries and generated sources. Not committed.
BUILD = ROOT / ".judge" / "build"

# The architectures the judge knows, by the name of their kernel directory.
ARM64, X86_64 = "arm64", "x86_64"
# Normalise the names different systems use for the same architecture.
_ARCH_ALIASES = {"arm64": ARM64, "aarch64": ARM64, "x86_64": X86_64, "AMD64": X86_64}
ARCH = _ARCH_ALIASES.get(platform.machine(), platform.machine())

# The C compiler, on every platform. It also links, and it assembles everything
# except x86-64 kernels (see _X86_64_ASSEMBLER).
CC = "clang"

# macOS writes Mach-O objects and Linux writes ELF. The two name symbols and
# shared libraries differently and are inspected with different tools.
_MACH_O = platform.system() == "Darwin"
# Suffix of a shared library.
SHARED_SUFFIX = ".dylib" if _MACH_O else ".so"
# What the object format puts in front of a C function's name.
SYMBOL_PREFIX = "_" if _MACH_O else ""

# Tune for the processor the judge is running on.
_NATIVE_FLAG = "-mcpu=native" if ARCH == ARM64 else "-march=native"
# ELF code that goes into a shared library has to be position-independent.
# Mach-O code always is, so macOS gets no extra flag.
_PIC_FLAGS = [] if _MACH_O else ["-fPIC"]
# The naive reference is always built with exactly these flags.
C_FLAGS = ["-O3", _NATIVE_FLAG, "-std=c11", "-Wall", *_PIC_FLAGS]
# An ELF library is marked as needing an executable stack unless every object
# in it says otherwise, and current Linux refuses to load one that is. Kernels
# must say so themselves (checked in _elf_issues); this covers the harness.
_LINK_FLAGS = [] if _MACH_O else ["-Wl,-z,noexecstack"]

# x86-64 kernels are assembled by GNU as, not by the C compiler, because it is
# the assembler that can be told which instructions exist and refuse the rest.
_X86_64_ASSEMBLER = ["as", "--64"]
# The instruction-set level an x86-64 kernel may use, by its usual name.
X86_64_LEVEL = "x86-64-v3"
# The same level as GNU as spells it: the 64-bit baseline (which includes SSE2
# and CMOV) plus each extension x86-64-v2 and x86-64-v3 add. MMX and the x87
# stack are switched off: neither is of use to a kernel, and both leave state
# behind that the caller would have to clean up.
_X86_64_LEVEL_FLAG = ("-march=generic64+sse3+ssse3+sse4.1+sse4.2+popcnt+cx16+lahf_sahf"
                      "+avx+avx2+bmi+bmi2+f16c+fma+lzcnt+movbe+nommx+no87")
# How GNU as reports an instruction the stated level does not have:
# "<file>:<line>: Error: `<mnemonic>' is not supported on `<level>'".
_OUTSIDE_LEVEL = re.compile(r":(\d+): Error: `([^']+)' is not supported on")
# Any other error line from GNU as: "<file>:<line>: Error: <message>".
_ASSEMBLER_ERROR = re.compile(r":(\d+): Error: (.+)$")

# `nm -g` prints "U name" for a symbol the object needs from elsewhere, and
# "<address> <type> name" for one it defines.
_NM_UNDEFINED_FIELDS = 2
_NM_DEFINED_FIELDS = 3
# In `otool -l` output, a section that lives in the writable data segment.
_WRITABLE_SEGMENT_MARKER = "segname __DATA"
# One section in `readelf -S -W` output:
# "[Nr] Name Type Address Off Size ES Flg Lk Inf Al". The flags may be empty.
_ELF_SECTION = re.compile(
    r"^\s*\[\s*\d+\]\s+(?P<name>\S+)\s+\S+\s+[0-9a-f]+\s+[0-9a-f]+\s+(?P<size>[0-9a-f]+)"
    r"\s+[0-9a-f]+\s+(?P<flags>[A-Za-z]*)\s+\d+\s+\d+\s+\d+\s*$")
# The flag letter readelf prints for a writable section.
_ELF_WRITABLE = "W"
# readelf prints section sizes in this base.
_HEXADECIMAL = 16
# The empty section by which an ELF object says it does not need an executable stack.
_ELF_STACK_NOTE = ".note.GNU-stack"


class BuildError(Exception):
    """A compiler, assembler or linker step failed. The message is the tool's output."""


def _try(cmd: list) -> subprocess.CompletedProcess:
    """Run a build tool and return the completed process, whether or not it succeeded.

    Args:
        cmd: The program and its arguments; paths are converted to strings.

    Returns:
        The completed process, with its output captured as text.
    """
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True)


def _run(cmd: list) -> str:
    """Run a build tool and return its standard output, raising BuildError on failure.

    Args:
        cmd: The program and its arguments; paths are converted to strings.

    Returns:
        What the tool printed on standard output.
    """
    proc = _try(cmd)
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


def _call_arg(arg: In | Out | InOut | Size | Scalar) -> str:
    """Return the C expression the timing loop passes for one kernel argument.

    A buffer is taken from the pool of inputs. A scalar is a parameter of the
    timing loop itself, named `s_<argument>`, so one compiled loop can time
    the kernel with any scalar values.
    """
    if isinstance(arg, Size):
        return "(size_t)n"
    if isinstance(arg, Scalar):
        return f"s_{arg.name}"
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


def _symbol_issues(k: Kernel, obj: Path) -> list[str]:
    """Check which symbols an assembled kernel exports and needs.

    Args:
        k: The kernel's spec.
        obj: The assembled object file.

    Returns:
        One sentence per broken rule: a kernel must export exactly one symbol,
        itself, and need nothing from outside.
    """
    issues: list[str] = []
    wanted = f"{SYMBOL_PREFIX}{k.name}"
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
    return issues


def _elf_issues(obj: Path) -> list[str]:
    """Check the sections of an ELF object against the rules a kernel must obey.

    Args:
        obj: The assembled object file.

    Returns:
        One sentence per broken rule: no writable data, and a declaration that
        the object does not need an executable stack.
    """
    issues: list[str] = []
    names: set[str] = set()
    for line in _run(["readelf", "-S", "-W", obj]).splitlines():
        section = _ELF_SECTION.match(line)
        if not section:
            continue
        names.add(section["name"])
        # The assembler always emits empty .data and .bss sections; only one
        # with something in it is writable data.
        if _ELF_WRITABLE in section["flags"] and int(section["size"], _HEXADECIMAL):
            issues.append(f"declares writable data (section {section['name']})")
    if _ELF_STACK_NOTE not in names:
        issues.append(f'does not end with `.section {_ELF_STACK_NOTE},"",@progbits`; '
                      "without it the library would need an executable stack")
    return issues


def _object_issues(k: Kernel, obj: Path) -> list[str]:
    """Check an assembled object file against the rules a kernel must obey.

    A kernel must export exactly one symbol (itself), need nothing from outside,
    and contain no writable data.

    Args:
        k: The kernel's spec.
        obj: The assembled object file.

    Returns:
        One sentence per broken rule; empty when the object is acceptable.
    """
    issues = _symbol_issues(k, obj)
    if not _MACH_O:
        return issues + _elf_issues(obj)
    if _WRITABLE_SEGMENT_MARKER in _run(["otool", "-l", obj]):
        issues.append("declares writable data")
    return issues


def _level_issues(src: Path) -> list[str]:
    """Check that an x86-64 kernel uses no instruction outside its level.

    The processor the judge runs on may have newer instructions than the level
    allows, so a kernel that used one would pass every test here and crash on
    an older machine. Assembling it again with the level stated makes the
    assembler refuse them.

    Args:
        src: The kernel's assembly source, which is known to assemble without
            a stated level.

    Returns:
        One sentence per instruction the level does not have; empty when the
        kernel stays inside it.
    """
    proc = _try([*_X86_64_ASSEMBLER, _X86_64_LEVEL_FLAG, src, "-o", "/dev/null"])
    if proc.returncode == 0:
        return []
    issues = []
    for line in proc.stderr.splitlines():
        outside, other = _OUTSIDE_LEVEL.search(line), _ASSEMBLER_ERROR.search(line)
        if outside:
            issues.append(f"line {outside[1]}: {outside[2]} is not in {X86_64_LEVEL}")
        elif other:
            issues.append(f"line {other[1]}: not accepted at {X86_64_LEVEL}: {other[2]}")
    return issues or [f"not accepted at {X86_64_LEVEL}: {proc.stderr.strip()}"]


def _assemble(src: Path, obj: Path) -> list[str]:
    """Assemble a kernel into an object file.

    Args:
        src: The kernel's assembly source.
        obj: Where to write the object file.

    Returns:
        Rule violations the assembler can see; empty when there are none. Only
        x86-64 has any: instructions outside the level.

    Raises:
        BuildError: The source does not assemble at all.
    """
    if ARCH != X86_64:
        _run([CC, "-c", src, "-o", obj])
        return []
    _run([*_X86_64_ASSEMBLER, src, "-o", obj])
    return _level_issues(src)


def build_asm(k: Kernel, src: Path, tag: str = "kernel") -> tuple[Path | None, list[str]]:
    """Assemble a kernel into a shared library.

    `tag` names the output files, so several versions of one kernel can be built
    side by side. Returns (library, issues). When the object breaks a rule the
    library is not linked and is returned as None.
    """
    d = _build_dir(k)
    obj, library = d / f"{tag}.o", d / f"{tag}{SHARED_SUFFIX}"
    issues = _assemble(src, obj) + _object_issues(k, obj)
    if issues:
        return None, issues
    _run([CC, "-shared", obj, "-o", library, *_LINK_FLAGS])
    return library, []


def build_naive(k: Kernel, src: Path) -> Path:
    """Compile the naive C reference into a shared library with the fixed flags.

    A generated header with the prototype from the spec is force-included, so a
    `naive.c` whose signature disagrees with the spec fails to compile.
    """
    d = _build_dir(k)
    header = d / "kernel.h"
    header.write_text(render("kernel.h.j2", prototype=c_prototype(k)))
    library = d / f"naive{SHARED_SUFFIX}"
    _run([CC, *C_FLAGS, "-shared", "-include", header, src, "-o", library, *_LINK_FLAGS])
    return library


def _bench_source(k: Kernel) -> str:
    """Generate the C timing loop for this kernel's signature."""
    return render(
        "bench.c.j2",
        ret=RETURNS[k.returns],
        arg_types=[_c_type(a) for a in k.args],
        pointer_names=[a.name for a in k.buffers],
        scalar_params=[f"{SCALARS[a.dtype]} s_{a.name}" for a in k.scalars],
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
    library = d / f"harness{SHARED_SUFFIX}"
    _run([CC, *C_FLAGS, "-shared", bench_c, C_DIR / "harness.c", abi, "-o", library, *_LINK_FLAGS])
    return library
