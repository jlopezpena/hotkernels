# hotkernels

Hand-written assembly kernels for the NumPy operations where assembly can actually win, delivered as a zero-friction drop-in. ARM64 NEON first, x86-64 planned.

The kernels are built by coding agents iterating against a fixed judge: a change is kept only if it is correct against NumPy and measurably faster than both NumPy and naive `clang -O3` C.

**Status:** planning. See [docs/core-design.md](docs/core-design.md) for the proof-of-concept design.

## Licence

MIT
