"""Build-time generator: turns the kernel specs into what the extension needs.

Run by `meson.build` when the package is configured. It reads every
`kernels/<name>/spec.py` and has two commands:

- `write OUT_DIR` writes `kernels.h`, the C prototype of every kernel from
  its spec, and `kernels.json`, which says per kernel whether it ships as
  assembly or C and whether it is bit-identical to NumPy. The second is
  installed with the package for `report()`.
- `list WHAT` logs one item per line for the build to read: the source file
  of every assembly kernel, of every C kernel, or the compiler flags the judge
  builds a naive reference with, so a C kernel is built exactly as it was
  judged. A wheel, built once for every machine, swaps the judge's
  `-mcpu=native` for a named CPU (`--cpu`).

The specs and the prototype code belong to the judge, so the repository root
is put on the import path. The package is built from a checkout; a build from
anything else has no `kernels/` directory to read.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import typer
from jinja2 import Environment, FileSystemLoader, StrictUndefined

# hotkernels/tools/generate.py -> the repository root, two levels up.
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from judge.build import ARCH, C_FLAGS, c_prototype  # noqa: E402
from judge.spec import EXACT, load_spec  # noqa: E402

KERNELS_DIR = ROOT / "kernels"
TEMPLATES = Path(__file__).resolve().parent / "templates"

# Names used in kernels.json for how a kernel is implemented.
ASSEMBLY = "assembly"
C = "C"

# Arguments of the `list` command.
LIST_ASSEMBLY = "assembly-sources"
LIST_C = "c-sources"
LIST_FLAGS = "c-flags"

# The judge compiles for the machine it runs on. `--cpu native` keeps that flag
# as it is; any other value replaces it with -mcpu=<value>.
NATIVE_CPU = "native"
CPU_FLAG_PREFIX = "-mcpu="

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)


def _kernel_dirs() -> list[Path]:
    """Return every kernel directory, sorted by name."""
    return sorted(p.parent for p in KERNELS_DIR.glob("*/spec.py"))


def _source(kernel_dir: Path) -> tuple[str, Path]:
    """Return how a kernel is implemented on this architecture and its source file.

    Args:
        kernel_dir: `kernels/<name>`.

    Returns:
        (ASSEMBLY, the kernel.s for this architecture) when one exists,
        otherwise (C, the naive reference), which is then the shipped kernel.
    """
    asm = kernel_dir / ARCH / "kernel.s"
    if asm.exists():
        return ASSEMBLY, asm
    return C, kernel_dir / "naive.c"


def _implementations() -> dict[str, tuple[str, Path]]:
    """Return, per kernel name, how it is implemented and its source file."""
    return {kernel_dir.name: _source(kernel_dir) for kernel_dir in _kernel_dirs()}


@app.command()
def write(out_dir: Path) -> None:
    """Write `kernels.h` and `kernels.json` into OUT_DIR.

    Args:
        out_dir: The directory to write into. Created if missing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    env = Environment(loader=FileSystemLoader(TEMPLATES), undefined=StrictUndefined,
                      trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)
    implementations = _implementations()
    prototypes, table = [], {}
    for kernel_dir in _kernel_dirs():
        kernel = load_spec(kernel_dir)
        prototypes.append(c_prototype(kernel))
        table[kernel.name] = {
            "implementation": implementations[kernel.name][0],
            "exact": kernel.mode == EXACT,
        }
    (out_dir / "kernels.h").write_text(env.get_template("kernels.h.j2").render(prototypes=prototypes))
    (out_dir / "kernels.json").write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")


def _c_flags(cpu: str) -> list[str]:
    """Return the judge's C flags, compiled for `cpu`.

    Args:
        cpu: NATIVE_CPU to keep the judge's flags unchanged, or a value for
            clang's -mcpu (such as `apple-m1`) to target that CPU instead.

    Returns:
        The flags, in the judge's order.
    """
    if cpu == NATIVE_CPU:
        return list(C_FLAGS)
    return [CPU_FLAG_PREFIX + cpu if flag.startswith(CPU_FLAG_PREFIX) else flag for flag in C_FLAGS]


@app.command(name="list")
def list_items(what: str, cpu: str = typer.Option(NATIVE_CPU, help="CPU for the C flags.")) -> None:
    """Log one item per line: the sources of a kind of kernel, or the C flags.

    Args:
        what: LIST_ASSEMBLY or LIST_C for the source files of the kernels
            implemented that way, LIST_FLAGS for the judge's compiler flags.
        cpu: For LIST_FLAGS, the CPU to compile for; see `_c_flags`.
    """
    if what == LIST_FLAGS:
        items = _c_flags(cpu)
    else:
        wanted = {LIST_ASSEMBLY: ASSEMBLY, LIST_C: C}[what]
        items = [str(source) for kind, source in _implementations().values() if kind == wanted]
    for item in items:
        log.info("%s", item)


if __name__ == "__main__":
    # The build reads the `list` command's lines from standard output.
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    app()
