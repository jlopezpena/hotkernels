"""Decides where a timed call's output buffers sit relative to its inputs.

On x86-64 the processor compares only the address bits within a memory page
when it decides whether a load has to wait for an earlier store. If an output
buffer starts a few bytes after an input, counted within the page, every store
to it looks like a store to the input elements about to be loaded, and the
loop stalls. Measured on an Intel Core i5-14600KF at 1,048,576 elements: a
naive C running sum takes 1.4 ns per element instead of 0.3, and NumPy's
`clip` 1.9 instead of 0.24.

Left to the allocator, which case a call is timed in changes from one size to
the next and from one run to the next. Two arrays of a power-of-two size made
one after the other often sit exactly 16 bytes apart within the page. So the
judge chooses: a timing is taken with the output clear of the inputs, and a
second table of `judge run` shows the bad case on purpose.

The same rule is applied on every architecture. It costs nothing where the
processor compares whole addresses.
"""
from __future__ import annotations

import numpy as np

# Size of a memory page, in bytes: the part of an address the processor compares.
PAGE_BYTES = 4096
# An output that starts this many bytes or more after an input, counted within
# the page, is out of its way. Measured: the slowdown is largest at 8 to 32
# bytes, still visible at 128 and gone at 256.
CLEAR_BYTES = 256
# Where the bad case puts an output: this many bytes after its input within
# the page. It is where two power-of-two arrays allocated one after the other
# end up, because the allocator puts a 16-byte header in front of each.
CLOSE_BYTES = 16
# Outputs start on a multiple of this, as memory from the C allocator does.
ALIGN_BYTES = 16
# Size of a cache line. Where a buffer starts within one decides how many of a
# kernel's vector loads and stores straddle two lines, which changes its time
# by a tenth or more. Both placements therefore start an output at the same
# position within a line, so that they differ in nothing but the distance
# within the page.
LINE_BYTES = 64

# The two placements. APART is how every timing is taken; CLOSE is the bad case.
APART, CLOSE = "apart", "close"


def gaps(address: int, inputs: list[int]) -> list[int]:
    """Measure how far after each input a buffer starts, within the page.

    Args:
        address: Where the buffer starts.
        inputs: Where the input buffers start.

    Returns:
        One distance in bytes per input, each from 0 to PAGE_BYTES - 1.
    """
    return [(address - start) % PAGE_BYTES for start in inputs]


def clear_of(address: int, inputs: list[int]) -> bool:
    """Say whether an output buffer at an address is out of the way of the inputs.

    Args:
        address: Where the output buffer would start.
        inputs: Where the input buffers start.

    Returns:
        True when, counted within a page, the output starts either exactly
        where each input does or at least CLEAR_BYTES after it.
    """
    return all(gap == 0 or gap >= CLEAR_BYTES for gap in gaps(address, inputs))


def _aligned(address: int) -> int:
    """Round an address up to the next multiple of ALIGN_BYTES.

    Args:
        address: Any address.

    Returns:
        The smallest multiple of ALIGN_BYTES that is not below it.
    """
    return -(-address // ALIGN_BYTES) * ALIGN_BYTES


def next_address(cursor: int, inputs: list[int], anchor: int | None, placement: str) -> int:
    """Find where the next output buffer goes.

    Args:
        cursor: The first free address of the block the outputs are carved from.
        inputs: Where the input buffers of the same call start.
        anchor: The input the output is placed against in the CLOSE placement:
            the one the kernel reads in step with its writes. None when the
            call has no input.
        placement: APART or CLOSE.

    Returns:
        The lowest address at or after the cursor that sits where the anchor
        plus CLOSE_BYTES does within a cache line and that, for CLOSE, also
        sits there within the page, and for APART, is clear of every input.
        If a call has so many inputs that no place is clear of them all, the
        lowest address with the right position in the line. The cursor itself,
        aligned, when there is no input. The result is less than a page and a
        line above the cursor.
    """
    if anchor is None:
        return _aligned(cursor)
    close = _aligned(anchor) + CLOSE_BYTES
    start = cursor + (close - cursor) % LINE_BYTES
    if placement == CLOSE:
        return start + (close - start) % PAGE_BYTES
    candidates = range(start, start + PAGE_BYTES, LINE_BYTES)
    return next((address for address in candidates if clear_of(address, inputs)), start)


def anchor_of(count: int, inputs: list[np.ndarray]) -> int | None:
    """Choose the input an output buffer is placed against.

    Args:
        count: Number of elements in the output buffer.
        inputs: The input arrays of the call, in argument order.

    Returns:
        The address of the first input with as many elements as the output,
        since that is the one a kernel reads in step with its writes; of the
        first input if none has; None if there is no input.
    """
    if not inputs:
        return None
    in_step = [a for a in inputs if a.size == count]
    return (in_step or inputs)[0].ctypes.data


def place(calls: list[list[np.ndarray]], outputs: list[tuple[int, np.dtype]],
          placement: str) -> list[list[np.ndarray]]:
    """Make the output buffers of several calls, each placed against its own inputs.

    All of them are carved from one block, one after the other, so the outputs
    of a pool of small calls stay as close together as their inputs are.

    Args:
        calls: The input arrays of each call, in argument order.
        outputs: The element count and element type of each output buffer a
            call has. Every call has the same ones.
        placement: APART or CLOSE.

    Returns:
        For each call, its uninitialised output arrays in the order of
        `outputs`. Each is a view into the shared block, which stays allocated
        for as long as any of them does.
    """
    sizes = [count * np.dtype(dtype).itemsize for count, dtype in outputs]
    # Each buffer can be pushed up by less than a page and a line to reach its place.
    slack = (PAGE_BYTES + LINE_BYTES) * len(outputs)
    block = np.empty(len(calls) * (sum(sizes) + slack), np.uint8)
    base = cursor = block.ctypes.data
    placed = []
    for inputs in calls:
        starts = [a.ctypes.data for a in inputs]
        made = []
        for (count, dtype), size in zip(outputs, sizes):
            address = next_address(cursor, starts, anchor_of(count, inputs), placement)
            made.append(block[address - base:address - base + size].view(dtype))
            cursor = address + size
        placed.append(made)
    return placed
