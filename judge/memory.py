"""Guarded buffers: any access outside a buffer, or any write to an input, faults."""
from __future__ import annotations

import ctypes
import mmap

PAGE = mmap.PAGESIZE
ARRANGEMENTS = ["end", "start", "mid1", "mid2", "mid3"]
_SLACK = 128  # room for the "mid" arrangements

_libc = ctypes.CDLL(None, use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long,
]
_libc.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

_RW = mmap.PROT_READ | mmap.PROT_WRITE


def _mprotect(addr: int, size: int, prot: int) -> None:
    if _libc.mprotect(addr, size, prot) != 0:
        raise OSError(ctypes.get_errno(), "mprotect failed")


class Arena:
    """A run of data pages with an inaccessible page on each side."""

    def __init__(self, data_pages: int) -> None:
        self.data_pages = data_pages
        self.total = (data_pages + 2) * PAGE
        base = _libc.mmap(None, self.total, _RW, mmap.MAP_PRIVATE | mmap.MAP_ANON, -1, 0)
        if base is None or base == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_errno(), "mmap failed")
        self.base = base
        self.lo = base + PAGE
        self.hi = self.lo + data_pages * PAGE
        _mprotect(base, PAGE, 0)
        _mprotect(self.hi, PAGE, 0)

    @staticmethod
    def pages_for(nbytes: int) -> int:
        return max(1, -(-(nbytes + _SLACK) // PAGE))

    def place(self, nbytes: int, arrangement: str, itemsize: int) -> int:
        """Address for a buffer of `nbytes` under the given arrangement."""
        if arrangement == "end":
            return self.hi - nbytes  # ends exactly at the trailing guard page
        if arrangement == "start":
            return self.lo  # starts exactly after the leading guard page
        k = int(arrangement[3:])
        return self.lo + 64 + k * itemsize  # deliberately misaligned

    def protect(self, read_only: bool) -> None:
        _mprotect(self.lo, self.hi - self.lo, mmap.PROT_READ if read_only else _RW)

    def close(self) -> None:
        _libc.munmap(self.base, self.total)


class ArenaPool:
    """One arena per argument, reallocated only when the needed size changes."""

    def __init__(self) -> None:
        self._arenas: dict[str, Arena] = {}

    def get(self, name: str, nbytes: int) -> Arena:
        pages = Arena.pages_for(nbytes)
        arena = self._arenas.get(name)
        if arena is None or arena.data_pages != pages:
            if arena is not None:
                arena.close()
            arena = self._arenas[name] = Arena(pages)
        return arena
