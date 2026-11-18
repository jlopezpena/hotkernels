"""Whether hotkernels can be installed in the running interpreter.

The kernels are assembly for two platforms, ARM64 on macOS and x86-64 on
Linux, and the hooks rely on the interpreter lock and on the layout and
results of NumPy 2.3 and later NumPy 2 releases. The x86-64 kernels use the
instructions of the x86-64-v3 level, which not every x86-64 processor has, so
there the processor is asked as well. Anywhere else `install()` does nothing,
and `report()` gives the reason found here.
"""
from __future__ import annotations

import importlib
import platform
import sys
import sysconfig
from pathlib import Path

import numpy as np

# The platforms the kernels are written for: (sys.platform, platform.machine()).
MACOS_ARM64 = ("darwin", "arm64")
LINUX_X86_64 = ("linux", "x86_64")
SUPPORTED_PLATFORMS = (MACOS_ARM64, LINUX_X86_64)
# The instruction sets of the x86-64-v3 level that the x86-64 kernels use, by
# the names Linux gives them in /proc/cpuinfo. `abm` is how it reports LZCNT.
# Linux lists `avx` and `avx2` only when the operating system also saves the
# 256-bit registers across a switch of threads, which the kernels need as much
# as the instructions themselves.
X86_64_V3_FLAGS = ("avx", "avx2", "fma", "bmi1", "bmi2", "abm", "movbe", "popcnt")
# Where Linux describes the processor, and the line that lists what it can do.
CPUINFO = Path("/proc/cpuinfo")
CPUINFO_FLAGS = "flags"
# The Python implementation whose object layout the hooks rely on.
SUPPORTED_IMPLEMENTATION = "cpython"
# The NumPy major version whose C interface the binding is compiled against.
SUPPORTED_NUMPY_MAJOR = 2
# The oldest NumPy whose results the hooks reproduce. Earlier NumPy 2 releases
# answer some calls differently, and the hooks follow the later answer: NumPy
# 2.0's np.clip makes every zero it returns +0.0, where later releases keep the
# sign of the element or bound it came from, and before 2.3 np.count_nonzero
# returns a Python int where later releases return np.intp.
# The self-test at install does not see either difference. Keep this equal to
# the floor of `numpy` in pyproject.toml's dependencies.
OLDEST_NUMPY = "2.3.0"
# Build variable that is 1 on a free-threaded build of CPython.
FREE_THREADED_VARIABLE = "Py_GIL_DISABLED"
# The extension module that holds the binding. Absent where it was not built.
BINDING_MODULE = "hotkernels._binding"


def missing_x86_64_flags() -> list[str] | None:
    """List the x86-64-v3 instruction sets this processor lacks.

    Returns:
        The names from X86_64_V3_FLAGS that /proc/cpuinfo does not list for
        the first processor, in that order; empty when it has them all. None
        when the file cannot be read or has no flags line, in which case
        nothing is known and the kernels must not be run.
    """
    try:
        lines = CPUINFO.read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        name, _, value = line.partition(":")
        if name.strip() == CPUINFO_FLAGS:
            present = set(value.split())
            return [flag for flag in X86_64_V3_FLAGS if flag not in present]
    return None


def unsupported_reason() -> str | None:
    """Say why hotkernels cannot be installed here.

    Returns:
        A sentence fragment naming the first requirement that is not met, to
        follow "not installed: ", or None when everything is supported.
    """
    here = (sys.platform, platform.machine())
    if here not in SUPPORTED_PLATFORMS:
        return f"needs macOS on ARM64 or Linux on x86-64, and this is {sys.platform} on {platform.machine()}"
    if here == LINUX_X86_64:
        missing = missing_x86_64_flags()
        if missing is None:
            return f"could not read what this processor supports from {CPUINFO}"
        if missing:
            return ("needs a processor with the x86-64-v3 instruction sets, and this one lacks "
                    + ", ".join(missing))
    if sys.implementation.name != SUPPORTED_IMPLEMENTATION:
        return f"needs CPython, and this is {sys.implementation.name}"
    if sysconfig.get_config_var(FREE_THREADED_VARIABLE):
        return "free-threaded Python is not supported"
    numpy_major = int(np.__version__.split(".")[0])
    # NumpyVersion orders release candidates and development builds correctly.
    if numpy_major != SUPPORTED_NUMPY_MAJOR or np.lib.NumpyVersion(np.__version__) < OLDEST_NUMPY:
        return f"needs NumPy {OLDEST_NUMPY} or a later NumPy {SUPPORTED_NUMPY_MAJOR}, and this is NumPy {np.__version__}"
    try:
        importlib.import_module(BINDING_MODULE)
    except ImportError as error:
        return f"the extension module could not be loaded ({error})"
    return None
