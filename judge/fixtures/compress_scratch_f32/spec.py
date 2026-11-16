"""Fixture kernel: tests the judge on a kernel with a scratch output buffer.

The operation is the same as `compress_f32`. The difference is that the output
is declared `scratch=True`, so the kernel and the naive C may store every
element and only advance past the selected ones, leaving a stray element after
the reported output.
"""
from judge.spec import USED_RETURN, In, Kernel, Out, Size

# Share of mask elements that are True in each timing distribution.
HALF_SELECTED = 0.5
FEW_SELECTED = 0.05


def oracle(x, mask):
    """The elements of x where mask is True, as NumPy selects them."""
    return x[mask]


def _masked(fraction):
    """Return a timing distribution in which `fraction` of the mask is True."""
    def make(rng, n):
        """Build random data and a random mask for one timing input."""
        return {"x": rng.standard_normal(n).astype("float32"), "mask": rng.random(n) < fraction}
    return make


KERNEL = Kernel(
    name="compress_scratch_f32",
    returns="usize",  # number of elements written
    args=[
        In("x", "f32"),
        In("mask", "bool"),
        Out("out", "f32", used=USED_RETURN, scratch=True),
        Size("n"),
    ],
    oracle=oracle,
    distributions={"half_selected": _masked(HALF_SELECTED), "few_selected": _masked(FEW_SELECTED)},
)
