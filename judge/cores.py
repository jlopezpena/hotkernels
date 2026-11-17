"""Keeps the timing thread on one fast core, and says when Linux is holding the machine back.

macOS keeps a thread on the performance cores when it asks for the highest
scheduling class, which the harness does in C. Linux has no such class. On a
processor with performance and efficiency cores it moves a thread between them
as it sees fit, and an efficiency core runs vector code at a fraction of the
speed, so an unpinned timing is not a result. Here the judge pins the timing
thread to one performance core for the rest of the process.

Several judges may run at once, one per agent. Each claims a different core
through a lock file, so they do not time each other.
"""
from __future__ import annotations

import fcntl
import functools
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import IO

# Where Linux lists the CPUs that are performance cores, on a processor that
# also has efficiency cores. Absent on a processor whose cores are all alike.
_PERFORMANCE_CPUS = Path("/sys/devices/cpu_core/cpus")
# The CPUs that are threads of the same physical core as CPU `cpu`.
_SIBLINGS = "/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
# One frequency-scaling setting of CPU `cpu`.
_CPUFREQ = "/sys/devices/system/cpu/cpu{cpu}/cpufreq/{name}"
# "1" when the Intel driver has turbo switched off.
_NO_TURBO = Path("/sys/devices/system/cpu/intel_pstate/no_turbo")
# "0" when another driver has frequency boost switched off.
_BOOST = Path("/sys/devices/system/cpu/cpufreq/boost")
_TURBO_OFF = {_NO_TURBO: "1", _BOOST: "0"}

# Drivers that let the processor choose its own frequency. Under them the
# "powersave" governor is the normal setting and still reaches full speed.
# Under any other driver it pins the processor to its lowest frequency.
_SELF_MANAGING_DRIVERS = {"intel_pstate", "amd-pstate-epp"}
_POWERSAVE_GOVERNOR = "powersave"
# The energy preference that tells a self-managing processor to favour saving power.
_SAVE_POWER_PREFERENCE = "power"

# One-minute load average at or above which other work was clearly running.
# The judge's own timing thread accounts for about 1.
_BUSY_LOAD = 2.0

# One lock file per claimed CPU lives here. A lock is held for as long as the
# claiming process keeps the file open, and is released when it exits.
_LOCK_DIR = Path(tempfile.gettempdir()) / "hotkernels-judge-cores"
# Only Linux can pin a thread to a CPU.
_CAN_PIN = hasattr(os, "sched_setaffinity")
# A judge that starts further processes to time in passes its own CPU on in
# this environment variable. The child pins to it and does not claim one: the
# parent holds the lock and waits while the child runs.
INHERITED_CPU = "HOTKERNELS_JUDGE_CPU"


@dataclass
class Claim:
    """The CPU the timing thread has been pinned to."""

    # The CPU's number, as Linux counts them.
    cpu: int
    # The open lock file that marks the CPU as taken. None when every candidate
    # was already taken by another judge and this CPU is shared with one, and
    # in a process that inherited its CPU.
    lock: IO | None
    # True in a process that was given its parent's CPU and holds no lock.
    inherited: bool = False


def _read(path: Path | str) -> str | None:
    """Read a one-line system file.

    Args:
        path: The file, typically under /sys.

    Returns:
        Its contents without surrounding whitespace, or None if it cannot be read.
    """
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _parse_cpu_list(text: str) -> list[int]:
    """Read a CPU list in the kernel's format.

    Args:
        text: A comma-separated list of numbers and ranges, such as "0-11,14".

    Returns:
        Every CPU number in it, in order.
    """
    cpus: list[int] = []
    for part in filter(None, text.split(",")):
        first, _, last = part.partition("-")
        cpus.extend(range(int(first), int(last or first) + 1))
    return cpus


def _performance_cpus() -> list[int]:
    """List the CPUs the timing thread may be pinned to.

    Returns:
        The performance cores this process is allowed to run on. On a
        processor without efficiency cores, every CPU it is allowed to run on.
    """
    allowed = sorted(os.sched_getaffinity(0))
    listed = _read(_PERFORMANCE_CPUS)
    if not listed:
        return allowed
    performance = set(_parse_cpu_list(listed))
    return [cpu for cpu in allowed if cpu in performance] or allowed


def _one_thread_per_core(cpus: list[int]) -> list[int]:
    """Keep one CPU of each physical core.

    Two threads of one core share its execution units, so two judges pinned to
    them would still time each other.

    Args:
        cpus: Candidate CPU numbers.

    Returns:
        The candidates that are the lowest-numbered thread of their core. A
        CPU whose siblings cannot be read is kept.
    """
    kept = []
    for cpu in cpus:
        siblings = _read(_SIBLINGS.format(cpu=cpu))
        if not siblings or cpu == min(_parse_cpu_list(siblings)):
            kept.append(cpu)
    return kept


def _lock(cpu: int) -> IO | None:
    """Try to take the lock file of one CPU.

    Args:
        cpu: The CPU to claim.

    Returns:
        The open, locked file, or None if another process holds the lock.
    """
    _LOCK_DIR.mkdir(exist_ok=True)
    handle = open(_LOCK_DIR / f"cpu{cpu}.lock", "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


@functools.cache
def claim() -> Claim | None:
    """Pin the calling thread to a performance core no other judge is using.

    The first call pins; later calls in the same process return the same claim.
    Candidates are tried from the highest-numbered down, because the
    lowest-numbered CPUs take most of the machine's interrupts.

    Returns:
        The claim, or None where threads cannot be pinned (macOS). In a
        process started by another judge to time in, the parent's CPU,
        without a lock.
    """
    if not _CAN_PIN:
        return None
    inherited = os.environ.get(INHERITED_CPU)
    if inherited:
        os.sched_setaffinity(0, {int(inherited)})
        return Claim(int(inherited), None, inherited=True)
    candidates = _one_thread_per_core(_performance_cpus())[::-1]
    for cpu in candidates:
        lock = _lock(cpu)
        if lock:
            os.sched_setaffinity(0, {cpu})
            return Claim(cpu, lock)
    os.sched_setaffinity(0, {candidates[0]})
    return Claim(candidates[0], None)


def throttle_warnings(core: Claim | None) -> list[str]:
    """Describe anything on Linux that makes the timings unrepresentative.

    Args:
        core: The claimed core, or None where there is none (macOS).

    Returns:
        One sentence per problem found: a core shared with another judge,
        turbo switched off, a power-saving frequency setting, or other work
        running. Empty when nothing was found, and on macOS.
    """
    if core is None:
        return []
    found = []
    if core.lock is None and not core.inherited:
        found.append("every performance core was already claimed by another judge run, "
                     "so this one shared a core.")
    if any(_read(path) == off for path, off in _TURBO_OFF.items()):
        found.append("turbo is switched off, so the processor is held below its top speed.")
    driver = _read(_CPUFREQ.format(cpu=core.cpu, name="scaling_driver"))
    governor = _read(_CPUFREQ.format(cpu=core.cpu, name="scaling_governor"))
    preference = _read(_CPUFREQ.format(cpu=core.cpu, name="energy_performance_preference"))
    if governor == _POWERSAVE_GOVERNOR and driver not in _SELF_MANAGING_DRIVERS:
        found.append(f"the frequency governor is '{governor}', which holds the processor "
                     "at its lowest speed.")
    if preference == _SAVE_POWER_PREFERENCE:
        found.append("the energy preference is set to save power.")
    load = os.getloadavg()[0]
    if load >= _BUSY_LOAD:
        found.append(f"the load average is {load:.1f}, so other work was running.")
    return found
