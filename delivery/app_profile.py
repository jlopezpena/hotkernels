"""Profiles an application: where the processor's time goes, by native function.

The application runs in a loop in a child process while the system's sampling
profiler records the stack of every working thread about once a millisecond:
`sample` on macOS, `perf record -g` on Linux. The function at the top of each
stack is put in a class: the Python interpreter, NumPy, the BLAS, the maths
library, hotkernels, memory and system, or other compiled code. One more run
under `cProfile` says which Python functions the time is under. See
docs/applications-design.md.
"""
from __future__ import annotations

import cProfile
import os
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

# macOS's sampling profiler.
SAMPLER = "sample"
# Linux's sampling profiler.
PERF = "perf"
# The event perf samples on: the processor time of each thread, so a thread
# that waits is not sampled, as `sample`'s waiting threads are left out. It is
# a software event, so it works where the processor's counters are not
# exposed, as in a virtual machine.
PERF_EVENT = "cpu-clock"
# The same event in user space only, which kernel.perf_event_paranoid = 2, the
# kernel's default, allows a user who is not root. The kernel's own time is
# then missing from the shares.
PERF_USER_EVENT = PERF_EVENT + ":u"
# Samples per second perf takes of each thread: once a millisecond, as `sample` does.
PERF_FREQUENCY = 1000
# The fields `perf script` prints: the thread on a line of its own, then with
# -g one line per frame of its stack, the top first: address, symbol, file.
PERF_FIELDS = "comm,tid,ip,sym,dso"
# Where Linux says how much a user who is not root may sample.
PARANOID = Path("/proc/sys/kernel/perf_event_paranoid")
# The highest level at which such a user may sample the kernel as well.
KERNEL_PARANOID = 1
# The highest level at which such a user may sample its own processes in user space.
USER_PARANOID = 2
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
# What perf calls a function or a file it has no name for.
PERF_UNNAMED = "[unknown]"
# One frame of `perf script`'s output: "    7f1a2b3c symbol+0x4e (/path/to/file.so)".
# A symbol may hold parentheses (C++); the file is the last pair on the line.
PERF_FRAME = re.compile(r"^\s*(?P<ip>[0-9a-f]+)\s+(?P<name>.*?)\s+\((?P<image>[^()]*)\)\s*$")
# The offset perf may append to a symbol: "+0x4e".
PERF_OFFSET = re.compile(r"\+0x[0-9a-f]+$")
# The name under which the functions without symbols of one file are counted together.
NO_SYMBOLS = "(functions without symbols)"
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
# Images of the BLAS: Accelerate's parts, OpenBLAS as NumPy's and SciPy's
# wheels ship it, and a system's reference BLAS or MKL on Linux.
BLAS_IMAGES = ("libBLAS", "libLAPACK", "libvDSP", "libvMisc", "libBNNS", "libSparse", "libLinearAlgebra",
               "libopenblas", "libscipy_openblas", "libblas", "liblapack", "libmkl_")
# The C maths library: exp, log, tanh. libsystem_m on macOS; libm and its
# vector variants on Linux.
MATHS_IMAGES = ("libsystem_m", "libm.so", "libm-", "libmvec")
# Images of the operating system and the C and C++ runtimes, on macOS and on
# Linux, where perf names the kernel "[kernel.kallsyms]" and its page of
# fast system calls "[vdso]".
SYSTEM_IMAGES = ("libsystem_", "libdyld", "dyld", "libc++", "libobjc", "libdispatch", "libcorecrypto",
                 "libc.so", "libc-", "ld-linux", "ld-musl", "libstdc++", "libgcc_s", "libpthread", "[kernel",
                 "[vdso]")
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
    # The size it ran at, one of `apps.SIZE_NAMES`.
    size: str
    # The native profile on plain NumPy.
    plain: NativeProfile
    # The native profile after install().
    installed: NativeProfile
    # The Python functions with the most time of their own, on plain NumPy.
    python: tuple[PythonFunction, ...]


@dataclass(frozen=True)
class Sampler:
    """The sampling profiler this system has, and how it is run."""

    # "sample" or "perf".
    tool: str
    # For perf, the event sampled: PERF_EVENT, or PERF_USER_EVENT where only
    # user space may be sampled. Empty for `sample`.
    event: str

    @property
    def description(self) -> str:
        """Return how the samples were taken, as a record states it."""
        if self.tool == SAMPLER:
            return f"`{SAMPLER}`, every thread once a millisecond"
        note = ", user space only: the kernel's time is not in the shares" if self.event == PERF_USER_EVENT else ""
        return f"`{PERF} record -e {self.event} -F {PERF_FREQUENCY} -g`{note}"


def paranoid_level() -> int | None:
    """Read kernel.perf_event_paranoid.

    Returns:
        The level, or None where the file cannot be read: not Linux.
    """
    try:
        return int(PARANOID.read_text().strip())
    except (OSError, ValueError):
        return None


def _perf_allows(event: str) -> str:
    """Try perf on a trivial command with one event.

    Args:
        event: PERF_EVENT or PERF_USER_EVENT.

    Returns:
        An empty string if perf recorded it, else the last line perf wrote
        on its way out.
    """
    with tempfile.TemporaryDirectory() as directory:
        result = subprocess.run(
            [PERF, "record", "-q", "-e", event, "-o", str(Path(directory) / "probe.data"), "--", "true"],
            capture_output=True, text=True,
        )
    if result.returncode == 0:
        return ""
    lines = [line.strip() for line in result.stderr.splitlines() if line.strip()]
    return lines[-1] if lines else f"exit status {result.returncode}"


def find_sampler() -> tuple[Sampler | None, str]:
    """Find the sampling profiler this system has and may use.

    On Linux the event is chosen from kernel.perf_event_paranoid: the kernel
    is sampled too only by root or at a level of 1 or below. At 2 perf would
    leave the kernel out silently, so user space is asked for by name and the
    record says so. perf is then tried on a trivial command, since whether it
    may sample also depends on capabilities and, in a container, on its
    security profile.

    Returns:
        (the sampler, or None if there is none; a sentence saying why there
        is none, empty if there is one).
    """
    if sys.platform == "darwin":
        if shutil.which(SAMPLER):
            return Sampler(SAMPLER, ""), ""
        return None, f"no `{SAMPLER}` on this system"
    if not sys.platform.startswith("linux"):
        return None, f"no sampling profiler is known on {sys.platform}"
    if not shutil.which(PERF):
        return None, (f"no `{PERF}` on this system: install the package that provides it "
                      "(linux-perf on Debian, linux-tools-$(uname -r) on Ubuntu)")
    level = paranoid_level()
    kernel_allowed = os.geteuid() == 0 or (level is not None and level <= KERNEL_PARANOID)
    event = PERF_EVENT if kernel_allowed else PERF_USER_EVENT
    refusal = _perf_allows(event)
    if not refusal:
        return Sampler(PERF, event), ""
    if level is not None and level > USER_PARANOID and os.geteuid() != 0:
        cause = (f"kernel.perf_event_paranoid is {level}, which forbids it. `sudo sysctl "
                 f"kernel.perf_event_paranoid={USER_PARANOID}` lets a user sample its own processes in user space, "
                 f"{KERNEL_PARANOID} adds the kernel")
    else:
        cause = (f"kernel.perf_event_paranoid is {level}, which allows it, so something else forbids "
                 "perf_event_open: in a container, its security profile (run it with --privileged, or with "
                 "--cap-add PERFMON and a seccomp profile that permits the call)")
    return None, f"`{PERF}` may not sample: \"{refusal}\" {cause}"


def available() -> bool:
    """Say whether this system has a sampler it may use."""
    return find_sampler()[0] is not None


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
            name = NO_SYMBOLS if match["name"] == UNNAMED else match["name"]
            key = (name, match["image"])
            counted[key] = counted.get(key, 0) + int(match["count"])
    return summarise(counted, packages)


def parse_perf(script: str, packages: dict[str, str]) -> NativeProfile:
    """Count the samples of `perf script`'s output by the function at the top of the stack.

    The output holds one block per sample: a line naming the thread, then
    one indented line per frame of its stack, the top first, then an empty
    line. Only the first frame of each block is counted. perf samples a
    thread only while it runs, so no sample is of a thread that waits.

    Args:
        script: What `perf script -F comm,tid,ip,sym,dso` wrote for data
            recorded with -g.
        packages: The result of `package_images`.

    Returns:
        The profile. Unlike `sample`'s, every function is counted, however
        few its samples.
    """
    counted: dict[tuple[str, str], int] = {}
    top_seen = True
    for line in script.splitlines():
        if not line.strip():
            continue
        if not line[0].isspace():
            # The thread's line: a new sample begins.
            top_seen = False
            continue
        match = PERF_FRAME.match(line)
        if top_seen or not match:
            continue
        top_seen = True
        name = PERF_OFFSET.sub("", match["name"])
        name = NO_SYMBOLS if name in ("", PERF_UNNAMED) else name
        # perf names a file by its path, `sample` by its file name; the
        # kernel and the like are named in brackets.
        image = Path(match["image"]).name if match["image"].startswith("/") else match["image"]
        key = (name, image)
        counted[key] = counted.get(key, 0) + 1
    return summarise(counted, packages)


def summarise(counted: dict[tuple[str, str], int], packages: dict[str, str]) -> NativeProfile:
    """Turn counts of samples by function into shares by function and by class.

    Args:
        counted: (function name, image file name) -> samples at the top of the stack.
        packages: The result of `package_images`.

    Returns:
        The profile.
    """
    total = sum(counted.values()) or 1
    classes = dict.fromkeys(CLASSES, 0.0)
    functions = []
    for (name, image), count in counted.items():
        kind = classify(image, packages)
        classes[kind] += count / total
        functions.append(Function(kind, name[:NAME_WIDTH], image, count / total))
    functions.sort(key=lambda function: -function.share)
    return NativeProfile(total, classes, tuple(functions[:TOP_FUNCTIONS]))


def record_sample(pid: int, directory: Path) -> str:
    """Sample a process with macOS's `sample`.

    Args:
        pid: The process.
        directory: A directory to write the report in.

    Returns:
        The report's text.
    """
    report = directory / "sample.txt"
    subprocess.run([SAMPLER, str(pid), str(SAMPLE_SECONDS), "-file", str(report)], check=True, capture_output=True)
    return report.read_text(errors="replace")


def record_perf(pid: int, directory: Path, event: str) -> str:
    """Sample a process with perf and print the samples.

    `perf record -p` samples every thread of the process, those it starts
    later too, for as long as the command after `--` runs.

    Args:
        pid: The process.
        directory: A directory to write perf's data in.
        event: PERF_EVENT or PERF_USER_EVENT.

    Returns:
        What `perf script` printed.
    """
    data = directory / "perf.data"
    subprocess.run([PERF, "record", "-q", "-e", event, "-F", str(PERF_FREQUENCY), "-g", "-p", str(pid),
                    "-o", str(data), "--", "sleep", str(SAMPLE_SECONDS)], check=True, capture_output=True)
    script = subprocess.run([PERF, "script", "-i", str(data), "-F", PERF_FIELDS],
                            check=True, capture_output=True, text=True, errors="replace")
    return script.stdout


def sample_side(name: str, size: str, mode: str, sampler: Sampler, packages: dict[str, str]) -> NativeProfile:
    """Sample one side of an application.

    Args:
        name: One of `apps.NAMES`.
        size: One of `apps.SIZE_NAMES`.
        mode: "plain" or "installed", passed to the child.
        sampler: The result of `find_sampler`.
        packages: The result of `package_images`.

    Returns:
        The native profile of the child while it ran the application in a loop.
    """
    child = subprocess.Popen(
        [sys.executable, "-m", "delivery.app_child", name, size, mode, str(CHILD_SECONDS)],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        # The child writes READY once its data is built and the first run is over.
        if child.stdout.readline().strip() != READY:
            raise RuntimeError(f"{name}: the profiled process did not start")
        with tempfile.TemporaryDirectory() as directory:
            if sampler.tool == SAMPLER:
                return parse(record_sample(child.pid, Path(directory)), packages)
            return parse_perf(record_perf(child.pid, Path(directory), sampler.event), packages)
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


def python_profile(name: str, size: str) -> tuple[PythonFunction, ...]:
    """Run an application once on plain NumPy under cProfile.

    Args:
        name: One of `apps.NAMES`.
        size: One of `apps.SIZE_NAMES`.

    Returns:
        The functions with the most time of their own, largest first.
    """
    app = apps.load(name)
    data = app.make_data(size)
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


def profile(name: str, size: str, sampler: Sampler) -> AppProfile:
    """Profile one application at one size both ways.

    Args:
        name: One of `apps.NAMES`.
        size: One of `apps.SIZE_NAMES`.
        sampler: The result of `find_sampler`.

    Returns:
        Its native profiles and its Python profile.
    """
    packages = package_images()
    return AppProfile(name, size, sample_side(name, size, "plain", sampler, packages),
                      sample_side(name, size, "installed", sampler, packages), python_profile(name, size))


def as_row(result: AppProfile) -> dict:
    """Turn a profile into what the JSON record holds."""
    return asdict(result)


def _native(row: dict) -> NativeProfile:
    """Rebuild one side's native profile from its JSON record."""
    return NativeProfile(row["samples"], row["classes"], tuple(Function(**f) for f in row["functions"]))


def from_row(row: dict, size: str) -> AppProfile:
    """Rebuild a profile from its JSON record.

    Args:
        row: One application's profile as `as_row` wrote it.
        size: The size its timing says it ran at, for a record written before
            the profile stated it.

    Returns:
        The profile.
    """
    return AppProfile(row["name"], row.get("size", size), _native(row["plain"]), _native(row["installed"]),
                      tuple(PythonFunction(**f) for f in row["python"]))
