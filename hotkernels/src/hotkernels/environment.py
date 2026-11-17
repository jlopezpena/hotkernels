"""Whether hotkernels can be installed in the running interpreter.

The kernels are ARM64 assembly for macOS, and the hooks rely on the
interpreter lock and on the layout and results of NumPy 2.3 and later NumPy 2
releases. Anywhere else `install()` does nothing, and `report()` gives the
reason found here.
"""
from __future__ import annotations

import importlib
import platform
import sys
import sysconfig

import numpy as np

# The platform the kernels are written for.
SUPPORTED_SYSTEM = "darwin"
SUPPORTED_MACHINE = "arm64"
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


def unsupported_reason() -> str | None:
    """Say why hotkernels cannot be installed here.

    Returns:
        A sentence fragment naming the first requirement that is not met, to
        follow "not installed: ", or None when everything is supported.
    """
    if sys.platform != SUPPORTED_SYSTEM or platform.machine() != SUPPORTED_MACHINE:
        return f"needs macOS on ARM64, and this is {sys.platform} on {platform.machine()}"
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
