# hotkernels

Hand-written assembly kernels for the NumPy operations where assembly can actually win, delivered as a zero-friction drop-in. ARM64 NEON first, x86-64 planned.

The kernels are built by coding agents iterating against a fixed judge: a change is kept only if it is correct against NumPy and measurably faster than both NumPy and naive `clang -O3` C.

**Status:** phase 1 of the proof of concept. The judge exists; there are no kernels yet.

- [docs/core-design.md](docs/core-design.md): the proof-of-concept design
- [docs/judge-design.md](docs/judge-design.md): how kernels are checked and timed
- [AGENTS.md](AGENTS.md): rules for coding agents working on kernels

## The judge

Requires macOS on Apple Silicon, the Xcode command line tools, and [uv](https://docs.astral.sh/uv/).

```bash
uv run judge selftest            # prove the judge catches known-bad kernels
uv run judge check argmax_f32    # lint and correctness for one kernel
uv run judge run argmax_f32      # correctness, then a timing report
```

`argmax_f32` and `compress_f32` are fixtures used to test the judge, not kernels meant for shipping.

## Licence

MIT
