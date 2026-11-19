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

## 2. Scalar groups of four with M off the chain inside the group

- **Idea:** the maxima of x0, x0..x1, x0..x2, x0..x3 of a group of four do
  not depend on M, so each output is one `cmp`/`csel` against M and the chain
  from group to group is one `cmp` and one `csel` (two cycles per four
  elements instead of per element). Two groups per iteration, one more group,
  then the scalar loop of attempt 1 for 0 to 3 elements.
- **check:** pass
- **compare:** significantly faster, 1.34x [1.34 – 1.34]
- **Kept:** yes
- **Notes:** 1.40x to 1.57x from 256 up on both distributions, 0.41 ns per
  element (1.3 cycles): no longer the chain but the 14 `cmp`/`csel` per four
  elements. 0.72x at 16 (10.1 ns against 7.3) and 1.11x at 64.
