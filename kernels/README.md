# Kernels

One directory per kernel:

```
kernels/<name>/
  spec.py          signature, NumPy oracle, correctness mode
  naive.c          the simplest correct C loop
  arm64/kernel.s   hand-written NEON; absent for a kernel that is its naive C
  arm64/attempts.md    log of what was tried on ARM64 and what the judge said
  x86_64/kernel.s  hand-written AVX2
  x86_64/attempts.md   the same for x86-64
```

A kernel without a `kernel.s` for the architecture of the machine is a C kernel there: `uv run judge check <name>` verifies its `naive.c` against NumPy, and `uv run judge run <name>` reports the C against NumPy.

## Adding a kernel

1. A person writes `spec.py` and `naive.c`. Working examples are in `judge/fixtures/`.
2. Check the reference: `uv run judge check <name> --unlocked` builds `naive.c` against the spec and runs every correctness case on it. It must pass before any assembly is written.
3. Merge `spec.py` and `naive.c` to `main` through a pull request. From then on they are locked.
4. Start a `/goal` run using [docs/goal-spec-template.md](../docs/goal-spec-template.md).

## Scalar arguments

A `Scalar` has a default value, and the oracle receives the scalars after the arrays: `oracle(x, lo, hi)`. A kernel is verified only for the combinations of scalar values its spec lists, so list the ones that matter in `scalar_sets`: values that select a different path through the kernel, zeros of both signs, infinities, the smallest and largest sizes. Keep every combination inside the kernel's domain, and say in the spec's docstring what that domain is. Mark a set `timed=True` when its speed can differ from the default's. `kernels/clip_f64/spec.py` and `kernels/histogram_f64/spec.py` are working examples; the design is in [docs/judge-design.md](../docs/judge-design.md#scalar-sets).

## A second length

A kernel with two lengths that vary independently, such as a sorted array and the needles searched for in it, declares one of them as its `Size` and the other as an integer `Scalar` that sizes a buffer: `In("v", "f64", len="needles")`, `Out("out", "i64", capacity="needles")`, `Scalar("needles", "usize", 64)`. Make the `Size` the length that selects the kernel's code path, because it is the one swept exhaustively and timed across sizes. List in `scalar_sets` every value of the other length that matters: zero, one, and the counts either side of each group width the kernel works in. `kernels/searchsorted_f64/spec.py` is a working example; the design is in [docs/judge-design.md](../docs/judge-design.md#a-second-length).

## Values drawn from another argument

Random needles never equal an element of a random array, and a search answers differently for a needle that does. Declare `In("v", "f64", len="needles", draws_from="a")` and the judge also fills `v` with elements of `a` as the kernel receives it (`v=from_a` in a case description), at every size and in every scalar set, from an `a` with repeated values as well as a random one. The source must be another input of the same element type. `kernels/searchsorted_f64/spec.py` and `kernels/interp_f64/spec.py` are working examples; the design is in [docs/judge-design.md](../docs/judge-design.md#values-drawn-from-another-argument).

## Scratch output

By default a kernel may write only the output elements it reports. Declaring an output as `Out(..., used=USED_RETURN, scratch=True)` lets the kernel and `naive.c` write anywhere inside the buffer's capacity; only the reported elements are compared with NumPy. Use it when the fast way to produce variable-length output stores more than it keeps, as mask selection does. `judge/fixtures/compress_scratch_f32/` is a working example.

A kernel that needs working memory in proportion to its input gets it the same way, with a capacity that is a multiple of `n`: `Out("out", "i64", capacity=Times(3), used=USED_RETURN, scratch=True)` is room for three elements per element of input, of which the kernel reports the first `n`. `kernels/argsort_f64/spec.py` is a working example.

It has a cost: the rest of the buffer holds leftover values afterwards, so whatever calls the kernel must allocate the full capacity, return only the reported elements, and never run a scratch kernel on a buffer that belongs to the caller. The trade-offs are set out in [docs/judge-design.md](../docs/judge-design.md#scratch-output-writing-inside-the-capacity).
