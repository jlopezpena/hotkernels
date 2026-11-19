"""Profiles an application: where the processor's time goes, by native function.

The application runs in a loop in a child process while macOS's `sample`
records the stack of every thread about once a millisecond. The function at
the top of each stack is put in a class: the Python interpreter, NumPy, the
BLAS, the maths library, hotkernels, memory and system, or other compiled
code. One more run under `cProfile` says which Python functions the time is
under. See docs/applications-design.md.
"""
from __future__ import annotations

import cProfile
import pstats
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

from . import apps

# The system's sampling profiler. Only macOS has it.
SAMPLER = "sample"
# Seconds each side is sampled for.
SAMPLE_SECONDS = 6
# Seconds the child keeps running the application: longer than the sampling,
# so the sampler never sees it exit.
CHILD_SECONDS = SAMPLE_SECONDS + 4
# What the child writes on standard output when its data is built and one run is done.
READY = "ready"
# Functions shown per profile.
TOP_FUNCTIONS = 15
# Python functions shown from cProfile.
TOP_PYTHON = 10
# Longest function name shown; C++ names from NumPy's sort run to hundreds of characters.
NAME_WIDTH = 70

# The heading of the part of `sample`'s report that counts samples by the
# function at the top of the stack. It lists functions with five samples or more.
TOP_OF_STACK = "Sort by top of stack"
# One line of that part: "        name  (in image)        count". A function
# with no symbol is named "???" and has its address between the image and the count.
TOP_LINE = re.compile(r"^\s+(?P<name>.+?)\s+\(in (?P<image>[^)]+)\)(?:\s+load address .*\])?\s+(?P<count>\d+)\s*$")
# What `sample` calls a function it has no symbol for.
UNNAMED = "???"
# Functions a thread is in when it is waiting, not working. Their samples are left out.
WAITING = frozenset({
    "__psynch_cvwait", "__workq_kernreturn", "mach_msg2_trap", "__semwait_signal", "kevent", "__ulock_wait",
    "__psynch_mutexwait", "__select", "start_wqthread", "_pthread_wqthread", "semaphore_wait_trap", "read",
    "__read_nocancel", "swtch_pri",
})

# The classes, in the order they are shown.
PYTHON, NUMPY, BLAS, MATHS, HOTKERNELS, SYSTEM, OTHER = (
    "Python interpreter", "NumPy", "BLAS", "Maths library", "hotkernels", "Memory and system", "Other compiled code",
)
CLASSES = (PYTHON, NUMPY, BLAS, MATHS, HOTKERNELS, SYSTEM, OTHER)
# Images of the BLAS: Accelerate's parts, and OpenBLAS as SciPy's wheels ship it.
BLAS_IMAGES = ("libBLAS", "libLAPACK", "libvDSP", "libvMisc", "libBNNS", "libSparse", "libLinearAlgebra",
               "libopenblas", "libscipy_openblas")
# The C maths library: exp, log, tanh.
MATHS_IMAGES = ("libsystem_m",)
# Images of the operating system and the C and C++ runtimes.
SYSTEM_IMAGES = ("libsystem_", "libdyld", "dyld", "libc++", "libobjc", "libdispatch", "libcorecrypto")
# Suffixes of the compiled files of a Python package.
EXTENSION_SUFFIXES = (".so", ".dylib")


@dataclass(frozen=True)
class Function:
    """One native function's share of an application's samples."""

    # Its class, one of CLASSES.
    kind: str
    # The function's name, shortened to NAME_WIDTH.
    name: str
    # The file it is in.
    image: str
    # Share of the samples of working threads, from 0 to 1.
    share: float


@dataclass(frozen=True)
class NativeProfile:
    """Where the time of one side of an application went."""

    # Samples of working threads counted.
    samples: int
    # Share of the samples in each class, in the order of CLASSES.
    classes: dict[str, float]
    # The functions with the largest shares, largest first.
    functions: tuple[Function, ...]


@dataclass(frozen=True)
class PythonFunction:
    """One Python-level function's own time in a run under cProfile."""

    # Where it is: "file:line(name)" for Python code, the built-in's name for C.
    where: str
    # Times it was called.
    calls: int
    # Share of the run spent in the function itself, from 0 to 1.
    share: float


@dataclass(frozen=True)
class AppProfile:
    """One application's profiles."""

    # The application's module name.
    name: str
    # The native profile on plain NumPy.
    plain: NativeProfile
    # The native profile after install().
    installed: NativeProfile
    # The Python functions with the most time of their own, on plain NumPy.
    python: tuple[PythonFunction, ...]


def available() -> bool:
    """Say whether this system has the sampler."""
    return sys.platform == "darwin" and shutil.which(SAMPLER) is not None


def package_images() -> dict[str, str]:
    """Map the compiled files of installed packages to the package they belong to.

    `sample` names an image by its file name alone, so this is how a function
    is known to be NumPy's or a library's.

    Returns:
        File name -> top-level package directory, for every compiled file
        under site-packages.
    """
    site = Path(sysconfig.get_paths()["purelib"])
    images = {}
    for path in site.rglob("*"):
        if path.suffix in EXTENSION_SUFFIXES:
            images.setdefault(path.name, path.relative_to(site).parts[0])
    return images


def classify(image: str, packages: dict[str, str]) -> str:
    """Put a function in a class by the file it is in.

    Args:
        image: The image's file name, as `sample` prints it.
        packages: The result of `package_images`.

    Returns:
        One of CLASSES.
    """
    package = packages.get(image, "")
    if package == "numpy":
        return NUMPY
    if package == "hotkernels":
        return HOTKERNELS
    if image.startswith(BLAS_IMAGES) or (package.endswith(".dylibs") and "openblas" in image):
        return BLAS
    if image.startswith(MATHS_IMAGES):
        return MATHS
    if image.startswith(("python", "libpython")):
        return PYTHON
    if image.startswith(SYSTEM_IMAGES):
        return SYSTEM
    return OTHER


def parse(report: str, packages: dict[str, str]) -> NativeProfile:
    """Read the top-of-stack counts from a report of `sample`.

    Args:
        report: The text `sample` wrote.
        packages: The result of `package_images`.

    Returns:
        The profile. Functions with fewer than five samples are not in the
        report and so not in the shares.
    """
    counted: dict[tuple[str, str], int] = {}
    in_section = False
    for line in report.splitlines():
        if line.startswith(TOP_OF_STACK):
            in_section = True
            continue
        match = TOP_LINE.match(line) if in_section else None
        if in_section and line.strip() and not match:
            break
        if match and match["name"] not in WAITING:
            # Unnamed functions of one image are counted together.
            name = "(functions without symbols)" if match["name"] == UNNAMED else match["name"]
            key = (name, match["image"])
            counted[key] = counted.get(key, 0) + int(match["count"])
    total = sum(counted.values()) or 1
    classes = dict.fromkeys(CLASSES, 0.0)
    functions = []
    for (name, image), count in counted.items():
        kind = classify(image, packages)
        classes[kind] += count / total
        functions.append(Function(kind, name[:NAME_WIDTH], image, count / total))
    functions.sort(key=lambda function: -function.share)
    return NativeProfile(total, classes, tuple(functions[:TOP_FUNCTIONS]))


def sample_side(name: str, mode: str, packages: dict[str, str]) -> NativeProfile:
    """Sample one side of an application.

    Args:
        name: One of `apps.NAMES`.
        mode: "plain" or "installed", passed to the child.
        packages: The result of `package_images`.

    Returns:
        The native profile of the child while it ran the application in a loop.
    """
    child = subprocess.Popen(
        [sys.executable, "-m", "delivery.app_child", name, mode, str(CHILD_SECONDS)],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        # The child writes READY once its data is built and the first run is over.
        if child.stdout.readline().strip() != READY:
            raise RuntimeError(f"{name}: the profiled process did not start")
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "sample.txt"
            subprocess.run([SAMPLER, str(child.pid), str(SAMPLE_SECONDS), "-file", str(report)],
                           check=True, capture_output=True)
            return parse(report.read_text(errors="replace"), packages)
    finally:
        child.kill()
        child.wait()


def _where(function: tuple[str, int, str]) -> str:
    """Name a cProfile entry.

    Args:
        function: cProfile's key: (file, line, name).

    Returns:
        The built-in's name for C code, else the file from its package
        directory on, the line and the name.
    """
    file, line, name = function
    if file == "~":
        return name
    parts = Path(file).parts
    start = parts.index("site-packages") + 1 if "site-packages" in parts else max(len(parts) - 2, 0)
    return f"{'/'.join(parts[start:])}:{line}({name})"


def python_profile(name: str) -> tuple[PythonFunction, ...]:
    """Run an application once on plain NumPy under cProfile.

    Args:
        name: One of `apps.NAMES`.

    Returns:
        The functions with the most time of their own, largest first.
    """
    app = apps.load(name)
    data = app.make_data()
    profiler = cProfile.Profile()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        app.run(*data)
        profiler.enable()
        app.run(*data)
        profiler.disable()
    stats = pstats.Stats(profiler).stats
    total = sum(own for _, _, own, _, _ in stats.values()) or 1.0
    ranked = sorted(stats.items(), key=lambda item: -item[1][2])[:TOP_PYTHON]
    return tuple(PythonFunction(_where(function), calls, own / total)
                 for function, (_, calls, own, _, _) in ranked)


def profile(name: str) -> AppProfile:
    """Profile one application both ways.

    Args:
        name: One of `apps.NAMES`. The sampler must be available.

    Returns:
        Its native profiles and its Python profile.
    """
    packages = package_images()
    return AppProfile(name, sample_side(name, "plain", packages), sample_side(name, "installed", packages),
                      python_profile(name))


def as_row(result: AppProfile) -> dict:
    """Turn a profile into what the JSON record holds."""
    return asdict(result)


def _native(row: dict) -> NativeProfile:
    """Rebuild one side's native profile from its JSON record."""
    return NativeProfile(row["samples"], row["classes"], tuple(Function(**f) for f in row["functions"]))


def from_row(row: dict) -> AppProfile:
    """Rebuild a profile from its JSON record."""
    return AppProfile(row["name"], _native(row["plain"]), _native(row["installed"]),
                      tuple(PythonFunction(**f) for f in row["python"]))
