"""Type conversion: `x.astype(np.float32)` for uint8 data."""
import numpy as np

from judge.spec import In, Kernel, Out, Size


def oracle(x):
    """Every element of x as a float32."""
    return x.astype(np.float32)


KERNEL = Kernel(
    name="cast_u8_f32",
    returns="void",
    args=[In("x", "u8"), Out("out", "f32"), Size("n")],
    oracle=oracle,
)
