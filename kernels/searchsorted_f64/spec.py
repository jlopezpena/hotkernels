"""Binary search: `np.searchsorted(a, v)` with n sorted values and n needles."""
import numpy as np

from judge.spec import In, Kernel, Out, Size


def oracle(a, v):
    """For each needle, the first index at which it could be inserted to keep `a` sorted."""
    return np.searchsorted(a, v)


KERNEL = Kernel(
    name="searchsorted_f64",
    returns="void",
    args=[
        # NumPy sorts NaN last and searches with the same ordering, so a sorted
        # array may end in NaNs and the answer is still unique.
        In("a", "f64", prepare=np.sort),
        In("v", "f64"),
        Out("out", "i64"),
        Size("n"),
    ],
    oracle=oracle,
)
