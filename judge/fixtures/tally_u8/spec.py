"""Fixture kernel: tests the judge on scalar sets and on an output sized by a scalar."""
import numpy as np

from judge.spec import In, Kernel, Out, Scalar, ScalarSet, Size

# Bucket count in the default scalar set. A power of two, so a kernel that
# hard-codes it can use a mask and still pass every default case.
BUCKETS = 16
# The other scalar sets. `more` comes first: a kernel that assumes BUCKETS
# stays inside the larger table and gives wrong counts. `fewer` makes the table
# smaller than the default one, so a kernel that clears BUCKETS entries
# regardless writes past it.
MORE_BUCKETS = 24
FEWER_BUCKETS = 5


def oracle(x, nbuckets):
    """How many elements of x leave each remainder when divided by nbuckets."""
    return np.bincount(x % nbuckets, minlength=nbuckets)


KERNEL = Kernel(
    name="tally_u8",
    returns="void",
    args=[
        In("x", "u8"),
        Out("counts", "i64", capacity="nbuckets"),
        Size("n"),
        Scalar("nbuckets", "usize", BUCKETS),
    ],
    oracle=oracle,
    scalar_sets=[
        ScalarSet("more", {"nbuckets": MORE_BUCKETS}),
        ScalarSet("fewer", {"nbuckets": FEWER_BUCKETS}),
    ],
)
