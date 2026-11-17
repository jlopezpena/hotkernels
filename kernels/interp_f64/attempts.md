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

## 8. A second length: the number of needles is its own argument

This is not a tuning attempt. The spec changed ([issue 16](https://github.com/sofeikov/hotkernels/issues/16),
[docs/two-sizes-design.md](../../docs/two-sizes-design.md)): `n` is now the length of the
sorted array only, and the number of queries arrives separately. The kernel was
changed to match and nothing else.

- **Idea:** the path is still chosen by `n`; "the last full group", which
  leftovers are covered by, is computed from the new count. A path needs one
  full group, so the entry also tests the count and sends a call with too few
  to a narrower path.
- **check:** pass, 155,063 cases (`--unlocked`: the spec and the judge differ from `main`)
- **compare:** not applicable: the previous kernel has another signature
- **Kept:** yes
- **Notes:** no instruction in either stage changed. With fewer than 16 queries, or
  fewer than 64 against 64 or more sample points, queries go one at a time
  through `Lslow`. One sample point now writes `fp[0]` for every query, where
  it used to write one. Indicative timing (unlocked, machine in use), 1,000
  queries: 6.25x over naive C and 19.6x over NumPy at 1,048,576 sample points;
  64 queries: 7.46x and 15.7x.

## Final numbers

These were measured before attempt 8, with as many queries as elements. The search
loops are the same code now.

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

## Queries on sample points: the judge's drawn class

The search in stage 1 is `searchsorted_f64`'s turned to the right side, and
the judge had the same gap for it: random queries never land on a sample
point. [Issue 48](https://github.com/sofeikov/kernel-arena/issues/48) gave the
judge a value class drawn from another argument; the spec now declares
`In("x", ..., draws_from="xp")`, so queries are drawn from the sample points,
with `xp` holding repeated values (`ties`), at every size up to 65,537 and in
every scalar set. The comparisons were mutated one part at a time on a copy of
the kernel (`fcmge` to `fcmgt`) and checked with `uv run judge check
interp_f64 --unlocked`, with the spec on `main` (five runs each, since only
the fresh cases differed) and with the new one. The kernel was restored after
each.

| Changed | Spec on `main` | With queries drawn from `xp` |
|---|---|---|
| First step of the 64-query search (`Lg64`, 32 instructions) | Passes | Passes |
| Later steps of it (`Lg64_pass`, 32) | Passes every fixed case; a fresh case caught it in 5 of 5 runs | **Fails**: `n=64 arr=mid3 seed=1 x=from_xp xp=ties fp=random` |
| The sixteen-query count (`Lg16_element`, 8) | Passes every fixed case; a fresh case caught it in 4 of 5 runs | **Fails**: `n=16 arr=start seed=1 x=from_xp xp=ties fp=random` |

The first step cannot be caught, because it cannot change a result. For a
query equal to `a[p - 1]` the mutation takes the lower window, and the search
ends at `p`, inside the run of sample points equal to the query. Stage 2 then
divides by a zero width, the result is NaN, and the four queries are
recomputed by `Lslow`, which is `naive.c` line by line. The unmutated kernel
passes 157,062 cases.

## Ten-run means, locked

`uv run judge run interp_f64`, locked, ten times, one run at a time, on a machine that was
not fully quiet (one-minute load average between 2.2 and 9.1 when a run started,
about 3 for most; a run the judge flagged as noisy was repeated). Check: pass,
155,063 cases. Mean of the ten, with the lowest and highest run in brackets. These
replace the indicative and single-pass figures above wherever they are quoted.

| Case | vs naive C, all sizes | at 1,048,576 | vs NumPy, all sizes | at 1,048,576 |
|---|---|---|---|---|
| 64 queries | 5.17x [5.04 – 5.26] | 7.85x [7.57 – 8.10] | 16.2x [16.0 – 16.3] | 15.7x [15.6 – 15.8] |
| 1,000 queries | 5.00x [4.93 – 5.03] | 6.40x [6.23 – 6.61] | 15.3x [15.0 – 16.0] | 16.8x [16.3 – 17.8] |

## Fewer queries than a group (issue 39)

[Issue 39](https://github.com/sofeikov/kernel-arena/issues/39): a call with
fewer queries than one group (fewer than 16, or fewer than 64 from 64 sample
points) went one query at a time through `Lslow`, a branchy copy of
`naive.c`, and so did every call with 2 to 15 sample points (one sample
point has its own path).

What `judge compare` times here: the default scalar set (64 queries) and
`many` (1,000), at every size from 16 to 16,777,216 sample points. Both take a
group path at every size, so no timed call reaches `Lslow`. The sets with 1,
15, 17 and 63 queries are checked, not timed, and the sweep starts above 15
sample points. A change to the few-query path therefore always reports "no
significant difference" and cannot be kept under the one rule. This is
[issue 65](https://github.com/sofeikov/kernel-arena/issues/65), found by the
same run on `searchsorted_f64`. The judge also checks no call with 2 to 14
queries, which is where a padded short group is
([issue 67](https://github.com/sofeikov/kernel-arena/issues/67)).

So the attempts below were judged by `check` and `compare`, and their speed
by a local C harness: the kernel on `HEAD` against the working copy, fresh
queries on every call (a pool of a million, so the branch predictor cannot
learn them), best of seven runs of 20,000 calls, the two interleaved; Apple M1
Pro, with the self-hosted CI runner and a desktop session on the machine. The
harness also compared the two outputs bit for bit. Each attempt was checked
as well by a local program against `naive.c`, bit for bit, for every query
count from 0 to 80 at every length from 1 to 300 sample points and at
511 to 1,048,577, on random, tied and special values (NaN, infinities, signed
zeros among the queries, the sample points and the values), with every buffer
ending at an inaccessible page: 150,366 cases, all identical.

## 9. Eight queries at once, a short group padded with its last query

- **Idea:** as attempt 12 of `searchsorted_f64`. Stage 1 is this kernel's
  search (count of `xp[1 .. n-2]` that are `<= v`) for eight queries at once,
  with the bases in x5-x12, the queries in d0-d7, `fcmp` + `csel ls` per
  probe and the step size shared. Stage 2 is the same arithmetic as the group
  path, one query at a time in scalar registers (`fdiv`, `fmadd`, two
  `fcsel`), the result replacing the query in its register, and a NaN fma
  result sends the call from this group to the end to `Lslow`. From eight
  queries, groups of eight with leftovers covered by redoing the last eight;
  below eight, the missing lanes load the last query again (`csel` on the
  index) and only the real results are stored. It serves every call that took
  `Lslow` except two sample points (nothing to search) and, as in
  `searchsorted_f64`, one query against 1,048,576 points or more.
- **check:** pass, 157,062 cases
- **compare:** no significant difference, 1.000x [0.999 – 1.000]: no timed
  call takes the path
- **Kept:** no
- **Notes:** locally against `HEAD`, from 64 to 16,384 points: 8 to 63
  queries 5.3x to 8.8x, 4 queries 2.9x to 4.4x; from 1,048,576, 3.4x to 5.9x
  and 2.0x to 3.2x. But one query was 0.64x at 16 points and 0.76x at 64 (36
  against 23 ns): padding one query to eight lanes costs eight searches'
  worth of instructions, and at small sizes the eight lanes are bound by
  throughput, not by the latency of a step. A mutation of one lane's `csel`
  (`ls` to `mi`) fails `check` at once.

## 10. One query: the same two stages for it alone

- **Idea:** attempt 9, with a copy of the two stages for exactly one query
  (`Lfew_one`): one base, one probe per step, the arithmetic once, and a NaN
  fma result handing the call to `Lslow`.
- **check:** pass, 157,062 cases
- **compare:** not run separately (the path is not timed; see above)
- **Kept:** no
- **Notes:** one query locally, against `HEAD`: 2.5x to 3.3x from 16 to
  262,144 points (12 against 29 ns at 16, 19 against 61 ns at 4,096).

## 11. Two to four queries: four lanes

- **Idea:** attempt 10, with a four-lane copy of the two stages for two to
  four queries (`Lfew4`), padded to four by repeating the last query.
- **check:** pass, 157,062 cases. The judge checks no call with 2 to 14
  queries, so this path was not run by it: replacing its store of the third
  result by `nop` still passes (157,062 cases). The local program against
  `naive.c` catches that mutation (`n=3 q=3`); issue 67 asks for the sets.
- **compare:** not run separately (the path is not timed)
- **Kept:** no
- **Notes:** against attempt 10 locally, two to four queries are 1.9x faster
  at 16 points, 1.6x at 4,096 and 1.2x to 1.35x from 1,048,576; five and seven
  queries are unchanged (same code). So the eight lanes are bound by
  throughput even far beyond the cache.

## 12. One query against a large array stays branchless

- **Idea:** attempt 9 kept one query against 1,048,576 points or more in
  `Lslow`, because the branchy search was faster there in `searchsorted_f64`
  (attempt 14 of its log). Measure it here and drop the test if not. Also: the
  eight-lane padded group now only serves five to seven queries, so its first
  five queries are loaded and stored directly; and the test for two sample
  points moved into the few-query path, so the group paths' entry is the same
  instructions as on `HEAD` (with the test in front, `compare` once reported
  "significantly slower", 1.000x [0.999 – 1.000], 0.2 ns at 16 points).
- **check:** pass, 157,062 cases
- **compare:** no significant difference, 1.000x [1.000 – 1.000]
- **Kept:** no. The kernel is unchanged on this run's branch; attempts 9 to
  12 together are on the branch `claude/interp_f64-few-needles-attempt`, to be
  judged once the spec times few queries (issue 65) and checks 2 to 14
  (issue 67).
- **Notes:** one query against `HEAD` locally, with the test and without it:
  1.00x / 2.0x at 1,048,576, 1.00x / 1.5x at 4,194,304, 1.00x / 1.2x at
  16,777,216. Here the branchless search wins at every size, unlike in
  `searchsorted_f64`; why the two kernels differ was not looked into.

  The final kernel against `HEAD` locally, nanoseconds per call, old / new:

  | n | 1 query | 2 | 4 | 5 | 8 | 15 | 32 | 63 |
  |---|---|---|---|---|---|---|---|---|
  | 4 | 14 / 7, 1.9x | 23 / 18, 1.3x | 37 / 16, 2.3x | 42 / 27, 1.6x | 62 / 25, 2.5x | 95 / 41, 2.3x | 190 / 77, 2.5x | 369 / 154, 2.4x |
  | 16 | 19 / 7, 2.8x | 35 / 14, 2.4x | 71 / 14, 4.9x | 87 / 27, 3.2x | 135 / 27, 5.1x | 248 / 53, 4.7x | group path | group path |
  | 64 | 27 / 9, 2.9x | 50 / 18, 2.7x | 105 / 19, 5.5x | 125 / 34, 3.7x | 197 / 34, 5.8x | 363 / 66, 5.5x | 771 / 131, 5.9x | 1,508 / 260, 5.8x |
  | 4,096 | 61 / 18, 3.3x | 122 / 33, 3.7x | 239 / 33, 7.2x | 297 / 54, 5.5x | 472 / 54, 8.7x | 879 / 107, 8.2x | 1,868 / 214, 8.7x | 3,670 / 425, 8.6x |
  | 65,536 | 105 / 40, 2.6x | 209 / 72, 2.9x | 414 / 73, 5.7x | 498 / 99, 5.0x | 789 / 103, 7.7x | 1,475 / 200, 7.4x | 3,139 / 408, 7.7x | 6,168 / 808, 7.6x |
  | 1,048,576 | 187 / 89, 2.1x | 372 / 165, 2.3x | 725 / 178, 4.1x | 905 / 240, 3.8x | 1,435 / 256, 5.6x | 2,664 / 491, 5.4x | 5,452 / 958, 5.7x | 10,707 / 1,880, 5.7x |
  | 16,777,216 | 585 / 471, 1.2x | 1,155 / 863, 1.3x | 2,276 / 888, 2.6x | 2,735 / 1,146, 2.4x | 4,354 / 1,245, 3.5x | 8,134 / 2,435, 3.3x | 17,335 / 4,898, 3.5x | 34,104 / 9,700, 3.5x |

  No cell is slower than `HEAD`. With 64 queries at 3 to 15 points (now eight
  at a time instead of one at a time) it is 2.5x at 3 points, 3.4x at 8 and
  4.3x at 15.

Not tried, and why:

- Two four-lane groups for five to seven queries instead of one padded group
  of eight: two four-query calls cost more than one padded group of eight at
  every size measured (for example 2 x 19 against 34 ns for five queries at
  64 points, 2 x 888 against 1,146 ns at 16,777,216).
- More than eight lanes in general registers: x0-x17 are all in use (eight
  bases, the step, the pass count, the probe pointer, a scratch, the group,
  out, the last group, xp, fp and n). Moving state to vector registers, as
  the group paths do, is what those paths already are.
