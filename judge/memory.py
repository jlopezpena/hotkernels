"""Guarded buffers for correctness testing.

Kernel arguments are not placed in ordinary allocations. Each one sits in an
"arena": a run of normal pages with an inaccessible page on each side. Placing a
buffer flush against one of those pages means that any access outside the
buffer, by even one byte, faults immediately. Inputs are additionally made
read-only for the duration of the call, so writing to an input also faults.
"""
from __future__ import annotations

import ctypes
import mmap

PAGE = mmap.PAGESIZE

# Where a buffer is placed inside its arena:
#   end    ends exactly at the trailing guard page (catches reads past the end)
#   start  starts exactly after the leading guard page (catches reads before the start)
#   midK   starts K elements after a 64-byte boundary (deliberately misaligned)
ARRANGEMENTS = ["end", "start", "mid1", "mid2", "mid3"]
_END, _START = "end", "start"

# The "mid" arrangements start this many bytes into the arena, plus K elements.
_MID_BASE_OFFSET = 64
# Extra bytes reserved per arena so the "mid" arrangements always fit:
# the base offset plus three of the widest (8-byte) elements, rounded up.
_SLACK = 128

_PROT_NONE = 0
_PROT_READ = mmap.PROT_READ
_PROT_READ_WRITE = mmap.PROT_READ | mmap.PROT_WRITE
# mmap reports failure by returning the address -1.
_MAP_FAILED = ctypes.c_void_p(-1).value

_libc = ctypes.CDLL(None, use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long,
]
_libc.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]


def _mprotect(addr: int, size: int, prot: int) -> None:
    """Change the access permissions of a page range, raising OSError on failure."""
    if _libc.mprotect(addr, size, prot) != 0:
        raise OSError(ctypes.get_errno(), "mprotect failed")


class Arena:
    """A run of data pages with an inaccessible guard page on each side.

    Layout:  [guard page][data pages ...][guard page]
                         ^lo             ^hi
    """

    def __init__(self, data_pages: int) -> None:
        """Map the arena and make its first and last pages inaccessible."""
        self.data_pages = data_pages
        self.total = (data_pages + 2) * PAGE
        base = _libc.mmap(None, self.total, _PROT_READ_WRITE,
                          mmap.MAP_PRIVATE | mmap.MAP_ANON, -1, 0)
        if base is None or base == _MAP_FAILED:
            raise OSError(ctypes.get_errno(), "mmap failed")
        self.base = base
        self.lo = base + PAGE
        self.hi = self.lo + data_pages * PAGE
        _mprotect(base, PAGE, _PROT_NONE)
        _mprotect(self.hi, PAGE, _PROT_NONE)

    @staticmethod
    def pages_for(nbytes: int) -> int:
        """Return how many data pages a buffer of `nbytes` needs, including slack."""
        return max(1, -(-(nbytes + _SLACK) // PAGE))

    def place(self, nbytes: int, arrangement: str, itemsize: int) -> int:
        """Return the address for a buffer of `nbytes` under the given arrangement."""
        if arrangement == _END:
            return self.hi - nbytes
        if arrangement == _START:
            return self.lo
        elements = int(arrangement[len("mid"):])
        return self.lo + _MID_BASE_OFFSET + elements * itemsize

    def protect(self, read_only: bool) -> None:
        """Make the data pages read-only, or readable and writable again."""
        _mprotect(self.lo, self.hi - self.lo, _PROT_READ if read_only else _PROT_READ_WRITE)

    def close(self) -> None:
        """Unmap the arena."""
        _libc.munmap(self.base, self.total)


class ArenaPool:
    """Keeps one arena per kernel argument, remapping only when the size changes."""

    def __init__(self) -> None:
        """Start with no arenas."""
        self._arenas: dict[str, Arena] = {}

    def get(self, name: str, nbytes: int) -> Arena:
        """Return an arena for the named argument that fits `nbytes`."""
        pages = Arena.pages_for(nbytes)
        arena = self._arenas.get(name)
        if arena is None or arena.data_pages != pages:
            if arena is not None:
                arena.close()
            arena = self._arenas[name] = Arena(pages)
        return arena
