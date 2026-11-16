# kernel-arena

Hand-written ARM64 NEON kernels for the NumPy operations where assembly can actually win, delivered as a zero-friction drop-in.

The kernels are built by coding agents competing against a fixed judge: a change is kept only if it is bit-identical to NumPy and measurably faster than both NumPy and `clang -O3` C.

**Status:** planning. See the open pull request for the proposed phases.
