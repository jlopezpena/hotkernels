# nanmin_cols_f64 attempts (arm64)

There is no `kernel.s` beside this log: on ARM64 the kernel ships as its naive C for now. No assembly has been tried yet; the one attempt is the C itself.

The naive C carries 32 columns at once, a row at a time, which keeps bottleneck's order within each column and lets clang make vector code of the row. Its speed against bottleneck, from the prototype it was taken from, is in [docs/bottleneck-design.md](../../../docs/bottleneck-design.md#a-prototype): 3.3x to 7.1x at 252 × 4,789 on an Apple M1 Pro. An assembly run starts once `spec.py` and `naive.c` are on `main`; for `nanmin` and `nanmax` the target is NumPy's `np.fmin.reduce`, which is 1.5x faster than this C along axis 0.

## 1. The naive C: 32 columns at a time, a row at a time

- **Idea:** bottleneck reduces one column at a time, each addition (or comparison) waiting for the one before. Carry 32 columns together, one accumulator each, and walk the rows in order: each column keeps bottleneck's order, so the result is the same bits, and the 32 operations of a row are independent, so clang makes NEON code of them.
- **check:** pass (`--unlocked`: the spec is not on `main`)
- **compare:** n/a (first version)
- **Kept:** yes
- **Notes:** from Python at 252 × 4,789 against bottleneck's own call, best of 15, Apple M1 Pro at a load average of 14: see [docs/kernel-findings.md](../../../docs/kernel-findings.md#bottlenecks-nan-aware-reductions).
