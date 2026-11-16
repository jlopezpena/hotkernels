# maskfill_f64 attempts

All timings were taken with `--unlocked` (the spec was not on `main` yet) while
other agents were timing other kernels on the same machine, so intervals are
wider than usual.

**What the store-back means for a caller.** From attempt 1 on, the kernel
loads elements of `x`, blends in `value`, and stores the result back. Elements
the mask did not select are therefore written with their own bits. The result
is bit-identical to `x[mask] = value`, but it is not the same memory behaviour:
another thread that writes one of those elements at the same time can have its
write undone, the pages of `x` are dirtied even when nothing was selected, and
`x` must be writable everywhere, not only where the mask is true. The naive C
loop (and NumPy) only store to selected elements.

## 1. Scalar branch-free baseline

- **Idea:** simplest correct version: per element, load `x[i]`, `csel` between it and the value bits on the mask byte, store back. No data-dependent branch.
- **check:** pass
- **compare:** n/a (first version). `run`: 1.34x [1.31 – 1.35] overall vs naive C; half_selected 2.67x [2.58 – 2.71], few_selected 0.67x [0.65 – 0.68].
- **Kept:** yes
- **Notes:** about 0.33 ns per element at every size. Naive C mispredicts at half_selected only from 65,536 elements up (2.7 ns per element there; below that the predictor memorises the cycled inputs and it runs at 0.4 ns). At few_selected naive C runs at 0.17 ns per element, twice as fast as this scalar loop.
