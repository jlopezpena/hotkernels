# Kernels

One directory per kernel:

```
kernels/<name>/
  spec.py          signature, NumPy oracle, correctness mode
  naive.c          the simplest correct C loop
  arm64/kernel.s   hand-written NEON; absent for a kernel that is its naive C
  attempts.md      log of what was tried and what the judge said
```

A kernel without `arm64/kernel.s` is a C kernel: `uv run judge check <name>` verifies its `naive.c` against NumPy, and `uv run judge run <name>` reports the C against NumPy.

## Adding a kernel

1. A person writes `spec.py` and `naive.c`. Working examples are in `judge/fixtures/`.
2. Check the reference: `uv run judge check <name> --unlocked` builds `naive.c` against the spec and runs every correctness case on it. It must pass before any assembly is written.
3. Merge `spec.py` and `naive.c` to `main` through a pull request. From then on they are locked.
4. Start a `/goal` run using [docs/goal-spec-template.md](../docs/goal-spec-template.md).

## Scratch output

By default a kernel may write only the output elements it reports. Declaring an output as `Out(..., used=USED_RETURN, scratch=True)` lets the kernel and `naive.c` write anywhere inside the buffer's capacity; only the reported elements are compared with NumPy. Use it when the fast way to produce variable-length output stores more than it keeps, as mask selection does. `judge/fixtures/compress_scratch_f32/` is a working example.

It has a cost: the rest of the buffer holds leftover values afterwards, so whatever calls the kernel must allocate the full capacity, return only the reported elements, and never run a scratch kernel on a buffer that belongs to the caller. The trade-offs are set out in [docs/judge-design.md](../docs/judge-design.md#scratch-output-writing-inside-the-capacity).
