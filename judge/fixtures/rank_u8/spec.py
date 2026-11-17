"""Fixture kernel: tests the judge on an input whose length is a scalar argument."""
import numpy as np

from judge.spec import In, Kernel, Out, Scalar, ScalarSet, Size

# Number of needles in the default scalar set.
NEEDLES = 3
# The other scalar sets. `more` comes first: a kernel that hard-codes NEEDLES
# stops short and leaves outputs unwritten. `one` and `none` make the needle
# buffer shorter than the default, down to nothing.
MORE_NEEDLES = 7
ONE_NEEDLE = 1
NO_NEEDLES = 0
# The kernel compares every needle with every element, so the sweep stops
# here: one past a power of two, and still thousands of cases.
CHECK_MAX_N = 1025


def oracle(a, v, needles):
    """For each needle, how many elements of `a` are smaller than it."""
    return (a[None, :] < v[:, None]).sum(axis=1)


KERNEL = Kernel(
    name="rank_u8",
    returns="void",
    args=[
        In("a", "u8"),
        # Needles drawn from the array as well: the selftest checks that the
        # judge generates them at every size and in every scalar set.
        In("v", "u8", len="needles", draws_from="a"),
        Out("out", "i64", capacity="needles"),
        Size("n"),
        Scalar("needles", "usize", NEEDLES),
    ],
    oracle=oracle,
    check_max_n=CHECK_MAX_N,
    scalar_sets=[
        ScalarSet("more", {"needles": MORE_NEEDLES}),
        ScalarSet("one", {"needles": ONE_NEEDLE}),
        ScalarSet("none", {"needles": NO_NEEDLES}),
    ],
)
