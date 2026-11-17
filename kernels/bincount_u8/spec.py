"""Count occurrences of each byte value: `np.bincount(x, minlength=256)` for uint8 data."""
import numpy as np

from judge.spec import In, Kernel, Out, Size

# A uint8 has this many possible values, so the counts always fit.
BINS = 256


def oracle(x):
    """How many times each value from 0 to 255 occurs in x."""
    return np.bincount(x, minlength=BINS)


KERNEL = Kernel(
    name="bincount_u8",
    returns="void",
    args=[In("x", "u8"), Out("counts", "i64", capacity=BINS), Size("n")],
    oracle=oracle,
)
