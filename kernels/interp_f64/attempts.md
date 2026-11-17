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

## 3. Stage 2 with two queries per vector

- **Idea:** scalar `fdiv` runs at one per cycle on this core and the two-lane
  `fdiv` also at one per cycle, so do the arithmetic for two queries at once:
  load `xp[jc], xp[jc+1]` and `fp[jc], fp[jc+1]` of each query as one 16-byte
  vector, `zip1`/`zip2` them into lane form, then `fsub`, `fdiv`, `fmla` (the
  fused multiply-add), the two compares and the two selects as vector
  operations.
- **check:** pass
- **compare:** significantly faster, 1.07x [1.07 – 1.07]
- **Kept:** yes
- **Notes:** 1.13x at 64 falling to 1.03x at 262,144, 1.04x to 1.05x from 1M,
  1.08x at 16. Locally, gathering with `ldp d` + `ins` instead of 16-byte
  loads + `zip` was 1% slower.

## 4. Counting instead of searching when n < 64

- **Idea:** with sixteen queries per group the searches wait on each other's
  steps (about 21 cycles per step for only sixteen probes). For n < 64, load
  the sixteen queries into eight vector registers and, for every element of
  `xp[1 .. n-2]`, compare it with all of them and add up the masks: no step
  depends on the one before.
- **check:** pass
- **compare:** significantly faster, 1.02x [1.02 – 1.02]
- **Kept:** yes
- **Notes:** 1.26x at 16, every other size unchanged (different code path).
  The same idea in `searchsorted_f64` was 4% slower than the code it would
  replace there, which already counts over a window of eight.

## 5. Detect NaN results with one `fmax` per pair

- **Idea:** stage 2 flagged NaN fma results with `fcmeq r, r` and an `and`
  into an accumulator, then moved both lanes to integer registers. Accumulate
  `fmax(acc, r)` instead (it stays NaN once any result is NaN) and test it
  with `fmaxp` + `fcmp` once per four queries: one vector operation per pair
  fewer, and no moves to integer registers.
- **check:** pass
- **compare:** significantly faster, 1.01x [1.01 – 1.01]
- **Kept:** yes
- **Notes:** 1.02x to 1.03x up to 4,096, 1.00x to 1.01x above.

## 6. From 1M points, prefetch `fp` for the whole group before stage 2

- **Idea:** far beyond the cache every `fp[jc]` load of stage 2 is a miss and
  only about thirty queries fit in the out-of-order window. Once stage 1 has
  written the 64 indices, issue `prfm` for all 64 `fp[jc]` before the
  arithmetic starts. Only when n >= 1,048,576.
- **check:** pass
- **compare:** significantly faster, 1.01x [1.01 – 1.01]
- **Kept:** yes
- **Notes:** 1.02x to 1.03x from 1M, unchanged below. Without the size test
  it was 0.92x to 0.97x locally from 64 to 262,144 (the prefetches take load
  ports and the data is already in the cache), so the test stays.

## 7. Two search steps per load of the queries (rejected on local timing)

- **Idea:** as attempt 8 of `searchsorted_f64`: do two consecutive search
  steps for each four queries so the queries are loaded once per two steps.
- **check:** not run through the judge (local harness: correct)
- **compare:** not run through the judge; locally 0.96x to 0.97x from 64 to
  65,536
- **Kept:** no
- **Notes:** the same result as in `searchsorted_f64`: the doubled dependency
  chain per register costs more than the saved loads.

## Final numbers

`uv run judge run interp_f64 --unlocked` on the kernel of attempt 6 (check:
pass, 53,227 cases), Apple M1 Pro, NumPy 2.5.5, 30 samples per point, with
other agents timing on the machine:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 34.6 ns | 62.6 ns | 1.81x [1.81 – 1.82] | 773.7 ns | 22.4x [22.31 – 22.42] |
| 64 | 122.1 ns | 408.5 ns | 3.34x [3.34 – 3.35] | 2.4 us | 20.0x [19.96 – 20.08] |
| 256 | 583.8 ns | 2.2 us | 3.76x [3.76 – 3.76] | 11.2 us | 19.2x [19.15 – 19.17] |
| 1,024 | 2.8 us | 12.1 us | 4.39x [4.39 – 4.39] | 53.8 us | 19.6x [19.54 – 19.57] |
| 4,096 | 12.8 us | 64.2 us | 5.02x [5.02 – 5.02] | 251.2 us | 19.6x [19.63 – 19.67] |
| 16,384 | 68.0 us | 371.7 us | 5.47x [5.38 – 5.47] | 1.31 ms | 19.3x [19.26 – 19.34] |
| 65,536 | 426.1 us | 2.47 ms | 5.81x [5.80 – 6.22] | 8.00 ms | 18.8x [18.68 – 18.86] |
| 262,144 | 2.30 ms | 13.03 ms | 5.65x [5.64 – 6.14] | 39.19 ms | 17.0x [16.96 – 17.16] |
| 1,048,576 | 14.78 ms | 100.80 ms | 6.82x [6.79 – 6.85] | 343.96 ms | 23.3x [23.12 – 23.38] |
| 4,194,304 | 142.55 ms | 1184.57 ms | 8.31x [8.26 – 8.34] | 4206.04 ms | 29.5x [29.33 – 29.72] |
| 16,777,216 | 853.85 ms | 7409.28 ms | 8.68x [8.62 – 8.74] | 33346.41 ms | 39.1x [38.90 – 39.27] |
| overall | | | 4.96x [4.95 – 5.00] | | 21.9x [21.84 – 21.89] |

NumPy is timed from Python and includes its per-call overhead.

Not tried, and why:

- The ideas listed as untried in `searchsorted_f64/attempts.md` apply to
  stage 1 here in the same way, with the same expectations.
- At 16, one search step followed by a count over a window of eight (as
  `searchsorted_f64` does below 64) would need about 150 vector operations
  per sixteen queries against 224 for the full count; estimated at 15% at
  that one size.
- The judge's correctness check stops at 65,537 elements for this kernel
  (the spec's `check_max_n`), so the prefetch of attempt 6, which only runs
  from 1,048,576, is not exercised by it. It only issues `prfm` on addresses
  inside `fp` and changes no result. As a substitute, a local program
  compared the kernel's output with `naive.c` byte for byte at 1,048,576,
  1,048,613 and 2,097,157 elements, with plain random data and with NaN,
  infinite and repeated values mixed in: identical in all six runs. That
  check has no guard pages, so it says nothing about out-of-bounds reads at
  those sizes.
