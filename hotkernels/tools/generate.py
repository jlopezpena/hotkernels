"""Build-time generator: turns the kernels in `kernels/` into a kernel set.

A kernel set is everything `meson.build` needs to compile the extension, in
one directory:

- `kernels.h`: the C prototype of every kernel, from its spec.
- `kernels.json`: per kernel, whether it ships as assembly or C and whether it
  is bit-identical to NumPy. Installed with the package for `report()`.
- `sources/<name>.s` or `sources/<name>.c`: each kernel's shipped source, its
  ARM64 assembly where it has one and its naive C reference otherwise.
- `assembly-sources.txt`, `c-sources.txt`: the file names in `sources/`, one
  per line.
- `c-flags.txt`: the flags the judge compiles a naive reference with, one per
  line, so a C kernel is built exactly as it was judged. `meson.build` swaps
  their `-mcpu=native` for a named CPU when the `kernel_cpu` option asks.

Two commands write one:

- `write OUT_DIR`, run by `meson.build` when it builds from a checkout.
- `dist`, run by `meson dist` as a dist script when a source distribution is
  made. It writes the set into the distribution's `kernel-set/`, so a build
  from the source distribution needs neither `kernels/` nor the judge.

The specs and the prototype code belong to the judge, so the repository root
is put on the import path. Both commands therefore run only in a checkout.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from pathlib import Path

import typer
from jinja2 import Environment, FileSystemLoader, StrictUndefined

# hotkernels/tools/generate.py -> the repository root, two levels up.
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from judge.build import C_FLAGS, c_prototype  # noqa: E402
from judge.spec import EXACT, load_spec  # noqa: E402

KERNELS_DIR = ROOT / "kernels"
TEMPLATES = Path(__file__).resolve().parent / "templates"

# The architecture the extension is built for. A kernel set always carries the
# ARM64 sources, whichever machine makes it.
TARGET_ARCH = "arm64"

# Names used in kernels.json for how a kernel is implemented.
ASSEMBLY = "assembly"
C = "C"

# The files of a kernel set; meson.build uses the same names.
HEADER = "kernels.h"
TABLE = "kernels.json"
SOURCES_DIR = "sources"
LIST_FILES = {ASSEMBLY: "assembly-sources.txt", C: "c-sources.txt"}
FLAGS_FILE = "c-flags.txt"

# Where the kernel set goes inside a source distribution, relative to the
# package's directory in it. meson.build looks for it there.
DIST_KERNEL_SET = "kernel-set"
# Set by `meson dist` for its dist scripts: the root of the unpacked
# distribution, which is the package's directory.
MESON_DIST_ROOT = "MESON_DIST_ROOT"

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, no_args_is_help=True)


def _kernel_dirs() -> list[Path]:
    """Return every kernel directory, sorted by name."""
    return sorted(p.parent for p in KERNELS_DIR.glob("*/spec.py"))


def _source(kernel_dir: Path) -> tuple[str, Path]:
    """Return how a kernel is implemented and its source file.

    Args:
        kernel_dir: `kernels/<name>`.

    Returns:
        (ASSEMBLY, its ARM64 kernel.s) when one exists, otherwise (C, its
        naive reference), which is then the shipped kernel.
    """
    asm = kernel_dir / TARGET_ARCH / "kernel.s"
    if asm.exists():
        return ASSEMBLY, asm
    return C, kernel_dir / "naive.c"


def _write_kernel_set(out_dir: Path) -> None:
    """Write a complete kernel set into `out_dir`, replacing any that is there.

    Args:
        out_dir: The directory to write into. Created if missing.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    sources_dir = out_dir / SOURCES_DIR
    sources_dir.mkdir(parents=True)
    env = Environment(loader=FileSystemLoader(TEMPLATES), undefined=StrictUndefined,
                      trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)
    prototypes, table = [], {}
    names: dict[str, list[str]] = {ASSEMBLY: [], C: []}
    for kernel_dir in _kernel_dirs():
        kernel = load_spec(kernel_dir)
        kind, source = _source(kernel_dir)
        # Named after the kernel: every source in the set is one flat directory.
        copied = f"{kernel.name}{source.suffix}"
        shutil.copyfile(source, sources_dir / copied)
        names[kind].append(copied)
        prototypes.append(c_prototype(kernel))
        table[kernel.name] = {"implementation": kind, "exact": kernel.mode == EXACT}
    (out_dir / HEADER).write_text(env.get_template("kernels.h.j2").render(prototypes=prototypes))
    (out_dir / TABLE).write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
    lines = env.get_template("lines.txt.j2")
    for kind, list_file in LIST_FILES.items():
        (out_dir / list_file).write_text(lines.render(items=names[kind]))
    (out_dir / FLAGS_FILE).write_text(lines.render(items=C_FLAGS))
    log.info("kernel set: %d assembly, %d C, in %s", len(names[ASSEMBLY]), len(names[C]), out_dir)


@app.command()
def write(out_dir: Path) -> None:
    """Write the kernel set into OUT_DIR, for a build from a checkout.

    Args:
        out_dir: The directory to write into. Created if missing.
    """
    _write_kernel_set(out_dir)


@app.command()
def dist() -> None:
    """Write the kernel set into the source distribution `meson dist` is making."""
    root = os.environ.get(MESON_DIST_ROOT)
    if not root:
        log.error("%s is not set: run this as a meson dist script", MESON_DIST_ROOT)
        raise typer.Exit(code=1)
    _write_kernel_set(Path(root) / DIST_KERNEL_SET)


if __name__ == "__main__":
    # Messages go to standard error: meson shows them, and nothing reads stdout.
    logging.basicConfig(level=logging.INFO, format="generate.py: %(message)s")
    app()
