"""Fixture kernel: tests the judge on a kernel that writes variable-length output."""
from judge.spec import In, Kernel, Out, Size


def oracle(x, mask):
    return x[mask]


def _masked(fraction):
    def make(rng, n):
        return {"x": rng.standard_normal(n).astype("float32"), "mask": rng.random(n) < fraction}
    return make


KERNEL = Kernel(
    name="compress_f32",
    returns="usize",  # number of elements written
    args=[
        In("x", "f32"),
        In("mask", "bool"),
        Out("out", "f32", used="return"),
        Size("n"),
    ],
    oracle=oracle,
    mode="exact",
    distributions={"half_selected": _masked(0.5), "few_selected": _masked(0.05)},
)
