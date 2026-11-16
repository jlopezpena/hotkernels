# interp_f64 attempts (arm64)

All judge commands in this log were run with `--unlocked` (the spec is not on
`main` yet), while other agents were timing other kernels on the same machine.

## 1. Baseline: naive.c translated line by line

- **Idea:** one query at a time, branchy binary search, every special case as
  a branch in the same order as `naive.c`.
- **check:** pass
- **compare:** not applicable (first version)
- **Kept:** yes
- **Notes:** starting point only. In a local harness it is about 4x slower
  than naive C from 64 elements up (the compiler makes the naive search
  branchless; this one mispredicts every other probe).

## 2. Two stages per group of 64 queries: branchless search, then branchless arithmetic

- **Idea:** reuse the search core of `searchsorted_f64` (64 searches at once,
  bases as 32-bit lanes of sixteen vector registers, `fcmge` for the `>=` of
  the naive loop, 16 instructions per four probes). The search runs over
  `xp[1 .. n-2]` only, so its count is NumPy's segment index already clamped
  to [0, n-2] and the four loads `xp[jc], xp[jc+1], fp[jc], fp[jc+1]` are
  always in bounds. The arithmetic is then branch-free per query: the fma
  value, replaced by `fp[jc]` when `!(v - xp[jc] > 0)` (exact hit or below
  the range) and by `fp[jc+1]` when `v >= xp[jc+1]` (at or above the last
  point). Whenever the fma result is NaN (NaN query, infinities, NumPy's two
  fallbacks) the four queries of that iteration are redone by a line-by-line
  copy of `naive.c`. Groups of 16 when n < 64; n < 16 goes to the copy of
  `naive.c`.
- **check:** pass
- **compare:** significantly faster, 10.7x [10.63 – 10.78]
- **Kept:** yes
- **Notes:** 1.72x at 16, 10.9x to 16.9x from 64 to 262,144, about 10x from
  1M. Locally about 1.2x over naive C at 16 and 3x to 6x above.
