"""Binary search from the right: `np.searchsorted(a, v, side='right')` for n sorted values and any number of needles.

The twin of `searchsorted_f64`, which is the left side. The two differ only for a
needle that equals an element, where this one answers the index after the last
equal element, and for a NaN needle, which goes after everything: its answer is n.
"""
import numpy as np

from judge.spec import In, Kernel, Out, Scalar, ScalarSet, Size

# Needles in the default scalar set: one full group of the widest path that
# works inside the cache. This set gets the full sweep and is timed.
NEEDLES = 64
# Needles in the second timed set: several groups and a leftover in every
# path (1,000 leaves 8 over groups of 16, 40 over 64 and 104 over 128).
MANY_NEEDLES = 1000
# The kernel searches for 16, 64 or 128 needles at once and handles leftovers
# by redoing the last full group, so the counts either side of each width are
# the ones that matter. Set name -> number of needles.
BOUNDARY_NEEDLES = {
    "none": 0, "one": 1,
    "k15": 15, "k16": 16, "k17": 17,
    "k63": 63, "k65": 65,
    "k127": 127, "k128": 128, "k129": 129,
}


def oracle(a, v, needles):
    """For each needle, the last index at which it could be inserted to keep `a` sorted."""
    return np.searchsorted(a, v, side="right")


KERNEL = Kernel(
    name="searchsorted_right_f64",
    returns="void",
    args=[
        # NumPy sorts NaN last and searches with the same ordering, so a sorted
        # array may end in NaNs and the answer is still unique.
        In("a", "f64", prepare=np.sort),
        # Drawn from `a` as well as random: a needle equal to an element is
        # where the two sides of the search differ, and random data never
        # gives one.
        In("v", "f64", len="needles", draws_from="a"),
        Out("out", "i64", capacity="needles"),
        # The length of a. It selects the kernel's code path and the depth of
        # the search, so it is the size that is swept.
        Size("n"),
        Scalar("needles", "usize", NEEDLES),
    ],
    oracle=oracle,
    scalar_sets=[ScalarSet(name, {"needles": count}) for name, count in BOUNDARY_NEEDLES.items()]
    + [ScalarSet("many", {"needles": MANY_NEEDLES}, timed=True)],
)
