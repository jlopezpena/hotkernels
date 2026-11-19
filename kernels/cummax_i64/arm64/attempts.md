# cummax_i64 attempts (arm64)

All timings on an Apple M1 Pro, macOS, native. Another kernel run was timing
in another worktree during these runs, so sizes from about 100,000 elements up
are noisier than on a quiet machine.

Reference point: the naive C (`clang -O3`) keeps the running maximum in one
register, a `cmp` and a `csel` per element on one chain: 0.65 ns per element,
two cycles, at every size from 1,024 up, on both distributions. NumPy takes
2.5 ns per element.

## 1. Simplest scalar loop

- **Idea:** one `ldr`, `cmp`, `csel`, `str` per element with post-indexed
  pointers; the starting point, to get `check` passing.
- **check:** pass
- **compare:** n/a (first version). `run`: 0.98x [0.98 – 0.98] of naive C
  overall, on both distributions.
- **Kept:** yes (baseline)
- **Notes:** 0.90x at 16, 1.00x from 4,096 up: both are held by the
  two-cycle `cmp`/`csel` chain, 0.65 ns per element.
