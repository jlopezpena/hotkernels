"""Whether each row of a boolean matrix holds a True: `m.any(axis=1)` for a two-dimensional array.

`m` holds `n` elements: `n // cols` whole rows of `cols` elements in C order,
and possibly a partial row after them, which is ignored. The kernel writes one
result per whole row and returns how many rows there were. When `cols` is
larger than `n` there are no rows: it returns 0 and writes nothing.

The kernel is defined for `cols >= 1`. Every byte of `m` is 0 or 1; a caller
that may hold other non-zero bytes checks for them before the call. Every byte
written is exactly 0 or 1.

The judge sweeps `n` for each width in SCALAR_SETS, so no rows, one row and a
partial last row are covered at every width. It times the default width and
the sets marked `timed`.
"""
from judge.spec import USED_RETURN, In, Kernel, Out, Scalar, ScalarSet, Size

# Row width in the default scalar set: the shape the kernel was asked for, and
# one that is neither a power of two nor a divisor of a vector width.
DEFAULT_COLS = 22

# Further row widths that are checked and not timed: the narrowest, and the
# widths either side of each vector and block size a kernel may work in (8,
# 16, 32 and 64 bytes), then two that are several vectors long.
CHECKED_COLS = (1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 65, 100, 1000)
# Row widths that are timed as well: a narrow row, where the work per row is
# everything, and one of two 32-byte vectors.
TIMED_COLS = (4, 64)

# Share of elements that are True in each timing distribution. With the sparse
# share most rows of the default width hold no True and must be read to the
# end to find that out.
SPARSE_SHARE = 0.03
DENSE_SHARE = 0.5


def _scalar_sets():
    """Return one scalar set per row width, named `cols_<width>`, ascending by width."""
    widths = sorted(CHECKED_COLS + TIMED_COLS)
    return [ScalarSet(f"cols_{width}", {"cols": width}, timed=width in TIMED_COLS) for width in widths]


def oracle(m, cols):
    """Whether each whole row of `cols` elements of m holds a True.

    Args:
        m: The boolean elements, row after row.
        cols: Elements per row, at least 1.

    Returns:
        One boolean per whole row: an array of `len(m) // cols` elements,
        empty when m is shorter than one row.
    """
    rows = len(m) // cols
    return m[:rows * cols].reshape(rows, cols).any(axis=1)


def _filled(share):
    """Return a timing distribution in which `share` of the elements are True.

    Args:
        share: Probability that an element is True, from 0 to 1.

    Returns:
        A function of a random generator and an element count that builds the
        input of one timed call.
    """
    def make(rng, n):
        """Build one random matrix, as its n elements in C order."""
        return {"m": rng.random(n) < share}
    return make


KERNEL = Kernel(
    name="any_rows_bool",
    returns="usize",  # number of rows, which is the number of results written
    args=[
        In("m", "bool"),
        Out("out", "bool", used=USED_RETURN),
        Size("n"),
        Scalar("cols", "usize", DEFAULT_COLS),
    ],
    oracle=oracle,
    scalar_sets=_scalar_sets(),
    distributions={"sparse": _filled(SPARSE_SHARE), "dense": _filled(DENSE_SHARE)},
)
