# Kernels

One directory per kernel:

```
kernels/<name>/
  spec.py          signature, NumPy oracle, correctness mode
  naive.c          the simplest correct C loop
  arm64/kernel.s   hand-written NEON
  attempts.md      log of what was tried and what the judge said
```

There are no kernels yet. The survey in phase 2 decides which ones are worth writing.

## Adding a kernel

1. A person writes `spec.py` and `naive.c`. Two working examples are in `judge/fixtures/`.
2. Check the reference: `uv run judge check <name> --unlocked` will report a missing kernel, but it builds `naive.c` against the spec first, so a signature mismatch shows up here.
3. Merge `spec.py` and `naive.c` to `main` through a pull request. From then on they are locked.
4. Start a `/goal` run using [docs/goal-spec-template.md](../docs/goal-spec-template.md).
