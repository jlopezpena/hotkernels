"""Builds the study's C: a library of loops called through ctypes, and a
NumPy data allocator built as an extension module.

Both are built once per run into a temporary directory that is removed when
the process ends. The loops are compiled the way the judge compiles C
(`clang -O3`), with `-ffp-contract=off` added so that a multiply followed by
a subtract stays two roundings, as it is in NumPy.
"""
from __future__ import annotations

import atexit
import ctypes
import importlib.util
import shutil
import subprocess
import sysconfig
import tempfile
from functools import cache
from pathlib import Path
from types import ModuleType

import numpy as np

HERE = Path(__file__).resolve().parent
# The loops, and the allocator.
LOOPS_SOURCE = HERE / "loops.c"
ALLOCATOR_SOURCE = HERE / "allocator.c"
# The compiler, as the judge uses it.
CC = "clang"
# Flags for the loops: the judge's optimisation level, and no contraction of a
# multiply and an add into one fused instruction.
LOOP_FLAGS = ("-O3", "-ffp-contract=off", "-shared")
# Flags for the allocator, a Python extension module whose references to the
# interpreter are resolved when the interpreter loads it.
EXTENSION_FLAGS = ("-O2", "-bundle", "-undefined", "dynamic_lookup")
# Name of the allocator's module, as its init function names it.
ALLOCATOR_MODULE = "study_allocator"
# File name of the built loops.
LOOPS_LIBRARY = "libgradient_step_loops.dylib"


@cache
def _directory() -> Path:
    """Return a temporary directory for the built files, removed at exit."""
    path = Path(tempfile.mkdtemp(prefix="gradient-step-"))
    atexit.register(shutil.rmtree, path, ignore_errors=True)
    return path


def _compile(arguments: list[str]) -> None:
    """Run the compiler and raise with its message if it fails.

    Args:
        arguments: Everything after the compiler's name.

    Raises:
        RuntimeError: The compiler failed.
    """
    result = subprocess.run([CC, *arguments], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{CC} failed:\n{result.stderr}")


@cache
def loops() -> ctypes.CDLL:
    """Build loops.c and return it with every function's argument types set.

    Returns:
        The library. Its functions take NumPy arrays' data addresses, as
        `array.ctypes.data`, element counts, and a prefetch distance in bytes.
    """
    library = _directory() / LOOPS_LIBRARY
    _compile([*LOOP_FLAGS, "-o", str(library), str(LOOPS_SOURCE)])
    lib = ctypes.CDLL(str(library))
    pointer, size, double = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_double
    lib.study_subtract.argtypes = [pointer, pointer, size, size]
    lib.study_fused.argtypes = [pointer, pointer, double, size, size]
    lib.study_fused_square_add.argtypes = [pointer, pointer, size]
    for function in (lib.study_subtract, lib.study_fused, lib.study_fused_square_add):
        function.restype = None
    return lib


@cache
def allocator() -> ModuleType:
    """Build allocator.c as an extension module and import it.

    Returns:
        The module, with `install()`, `uninstall()` and `counts()`.
    """
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    target = _directory() / f"{ALLOCATOR_MODULE}{suffix}"
    includes = [f"-I{sysconfig.get_paths()['include']}", f"-I{np.get_include()}"]
    _compile([*EXTENSION_FLAGS, *includes, "-o", str(target), str(ALLOCATOR_SOURCE)])
    spec = importlib.util.spec_from_file_location(ALLOCATOR_MODULE, target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
