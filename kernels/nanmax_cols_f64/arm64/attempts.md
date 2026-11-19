# nanmax_cols_f64 attempts (arm64)

There is no `kernel.s` beside this log: on ARM64 the kernel ships as its naive C for now. No assembly has been tried yet.

The naive C carries 32 columns at once, a row at a time, which keeps bottleneck's order within each column and lets clang make vector code of the row. Its speed against bottleneck, from the prototype it was taken from, is in [docs/bottleneck-design.md](../../../docs/bottleneck-design.md#a-prototype): 3.3x to 7.1x at 252 × 4,789 on an Apple M1 Pro. An assembly run starts once `spec.py` and `naive.c` are on `main`; for `nanmin` and `nanmax` the target is NumPy's `np.fmin.reduce`, which is 1.5x faster than this C along axis 0.
