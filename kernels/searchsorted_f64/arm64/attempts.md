# searchsorted_f64 attempts (arm64)

All judge commands in this log were run with `--unlocked` (the spec is not on
`main` yet), while other agents were timing other kernels on the same machine.

## 1. Baseline: branchy binary search, one needle at a time

- **Idea:** direct translation of `naive.c`, with a separate loop for NaN needles.
- **check:** pass
- **compare:** not applicable (first version)
- **Kept:** yes
- **Notes:** starting point only. It is about 2x slower than naive C at most
  sizes, because the compiler turns the naive loop into a conditional select
  while this version takes an unpredictable branch per probe.

## 2. Branchless lower bound, eight needles interleaved

- **Idea:** keep `base` per needle and a shared `len` (`half = len >> 1; if
  (a[base + half - 1] < key) base += half; len -= half`, then one last probe),
  with `fcmp` + `csel`. `len` depends only on n, so eight searches run in step
  and their loads overlap. A group containing a NaN needle (found with an `fmax`
  tree) and all needles when n < 8 go to the scalar search from attempt 1;
  leftovers are covered by one overlapping group ending at the last needle.
- **check:** pass
- **compare:** significantly faster, 7.25x [7.21 – 7.31]
- **Kept:** yes
- **Notes:** 10x to 12x from 64 to 16,384 elements, 2.7x at 16, 4x at 4M and
  16M. A local harness puts it at about 4x to 6.6x over naive C.
  Micro-benchmarks on this core: `fcmp` runs at one per cycle and the chain
  load -> `fcmp` -> `csel` is about 11.4 cycles, so this form is limited to
  about one probe per cycle however many needles are interleaved.

## 3. Sixteen needles, compare-to-mask instead of flags, one code block per level

- **Idea:** replace `fcmp` + `csel` with `fcmgt` (mask in a vector register),
  `fmov` to an integer register, and `sub base, base, mask, lsl #log2(h)`. That
  is still four instructions per probe but none of them is limited to one per
  cycle (`fmov x, d` runs at two per cycle). The shift must be a constant, so
  the search uses power-of-two steps (first step `base = a[p-1] < key ? n-p :
  0` with p the largest power of two <= n, skipped when n is a power of two)
  and the code for each step size is written out once; a group jumps to the
  block for h = p/2. With no register needed for the step size, all of x0-x15
  hold bases (sixteen needles per group) and the pointers and constants are
  parked in vector registers.
- **check:** pass
- **compare:** significantly faster, 1.30x [1.29 – 1.31]
- **Kept:** yes
- **Notes:** 1.21x to 1.38x from 64 to 1M elements, 1.5x at 4M and 16M (twice
  as many cache misses in flight), 1.01x at 16 (a single group per call, so it
  is limited by the length of the dependency chain, about 13.6 cycles per
  level in this form against 11.4 with `fcmp`).

## 4. From 1M elements: blocks of 128 needles swept once per step

- **Idea:** when the array is much larger than the cache the time goes on
  cache misses, and sixteen interleaved searches keep only sixteen in flight.
  Take 128 needles at a time, keep their bases in `out`, and sweep the block
  once per step (load base, probe, compare, store base). Iterations of a sweep
  are independent, so the out-of-order window holds the probes of about a
  hundred needles. NaN needles are found per block with a vector `fmax` and
  fixed up afterwards.
- **check:** pass
- **compare:** significantly faster, 1.12x [1.12 – 1.13]
- **Kept:** yes
- **Notes:** 1.82x at 4M and 16M, 1.09x at 1M, unchanged below (different
  code path). Local timing of other block sizes: 32 is far slower (0.3x to
  0.7x of 128), 512 is the same as 128. With the threshold at 65,536 the gain
  at 65,536 and 262,144 was only 1.02x to 1.05x locally, so the threshold
  stays at 1M.

## 5. Prefetch in the block sweeps (rejected on local timing)

- **Idea:** (a) after updating a base, `prfm` the next step's probe for that
  needle; (b) a separate cheap sweep of `prfm` over the block before each
  probing sweep, with blocks of 512 or 2,048 so the prefetches get a head start.
- **check:** not run
- **compare:** not run through the judge; in the local harness (a) was 0.65x
  to 0.9x of the version without prefetch from 65,536 to 16M, (b) 0.44x to 0.85x
- **Kept:** no
- **Notes:** slower at every size tried. The sweeps are already limited by
  the three load ports while the data is in cache, and a `prfm` takes one;
  when the data is not in cache the out-of-order window already has the next
  hundred probes issued.

## 6. 64 needles at once with bases in vector registers; window count for n < 64

- **Idea:** attempt 3 is limited by its dependency chain: sixteen searches,
  about 14.6 cycles per step, and the scheduler cannot reach into the next
  group, so about 0.94 cycles per probe. Two changes, measured together:
  (a) From 64 elements, keep the bases as 32-bit lanes of sixteen vector
  registers (64 searches in flight). One step for four needles: two moves to
  integer registers, four scalar loads of the probes, two `ins`, two loads of
  the needles from `v`, two `fcmgt`, `uzp1` to 32-bit masks, `mls` to add h
  where the mask is set: 16 instructions per four probes, h in a register,
  one loop over the steps instead of a block per step. NaN needles are found
  with `fmax` during a group's first step and fixed up afterwards.
  (b) Below 64 elements keep the sixteen-needle code but replace the last four
  probes, which all fall in `a[base .. base+7]`, by loading those eight
  elements and counting the ones below the key (one step in the chain
  instead of four).
- **check:** pass
- **compare:** significantly faster, 1.17x [1.16 – 1.17]
- **Kept:** yes
- **Notes:** 1.23x at 16 (from (b)), 1.42x at 64 falling to 1.10x at 262,144
  (from (a)), unchanged from 1M (still the block sweeps of attempt 4; locally
  the 64-needle path is equal to them at 1M and 4M and 6% slower at 16M).
  Locally, (b) applied to every size in the sixteen-needle code was 0.97x to
  1.01x from 64 up, and `and` + `add` in place of `mls` was 3% to 7% slower.
  An experiment that dropped the needle loads (wrong results, timing only)
  ran 1.3x to 1.4x faster, so the step is limited by its loads and
  instruction count: 1.5 loads per probe against three load ports.

## 7. Counting over all of `a` for 16 <= n < 64 (rejected on local timing)

- **Idea:** drop the sixteen-needle binary search and, for sixteen needles at
  once, compare every element of `a` with all of them and add up the masks
  (no dependency chain at all, and a much shorter kernel).
- **check:** not run through the judge (local harness: correct)
- **compare:** not run through the judge; locally 0.96x at 16, unchanged at 64
- **Kept:** no
- **Notes:** 16 vector operations per element for sixteen needles, and the
  core has four vector ports, so 16 elements take about 64 cycles: no better
  than the search it would replace.

## 8. Two search steps per load of the needles (rejected on local timing)

- **Idea:** in the 64-needle path, do two consecutive steps for each four
  needles before moving to the next four, so the needles are loaded from `v`
  once per two steps (1.25 loads per probe instead of 1.5).
- **check:** not run through the judge (local harness: correct)
- **compare:** not run through the judge; locally 1.00x to 1.01x from 256 to
  16,384, 0.97x at 64, 0.90x at 65,536 and 262,144, 0.99x from 1M
- **Kept:** no
- **Notes:** the saved loads are cancelled by the doubled dependency chain
  per register. See the notes of attempt 10 for why the loads were not the
  limit after all.

## 9. Lane load instead of load plus insert (rejected on local timing)

- **Idea:** `add x7, x17, x5, lsr #29` + `ld1 {v24.d}[1], [x7]` in place of
  `lsr` + `ldr d25` + `ins`, one instruction fewer per two probes.
- **check:** not run through the judge (local harness: correct)
- **compare:** not run through the judge; locally 1.00x at every size from
  16 to 65,536
- **Kept:** no
- **Notes:** the lane load is evidently a load plus an insert internally.

## 10. Diagnosis of the 64-needle path (no change to the kernel)

- **Idea:** find out what limits the path at about 0.74 cycles per probe
  when its instruction count allows about 0.5.
- **check:** not applicable
- **compare:** not applicable
- **Kept:** no change
- **Notes:** timing-only experiments in the local harness. Comparing against
  a constant instead of the loaded needles (so every search takes the same
  path and probes the same elements, with all the same instructions executed)
  ran 1.06x faster at 64, 1.25x at 1,024 and 1.33x at 4,096: the time depends
  on which elements of `a` are probed, not on the instruction count. The
  timing pool at these sizes holds about 96 KB of page-aligned arrays against
  a 128 KB first-level cache, so part of the random probes miss it. With 32
  searches per group instead of 64 the path is 0.83x to 0.96x, so it is only
  mildly limited by how many searches are in flight; doubling them again
  would need 16-bit bases (n <= 65,536 only) and was not tried.

## 11. A second length: the number of needles is its own argument

This is not a tuning attempt. The spec changed ([issue 16](https://github.com/sofeikov/hotkernels/issues/16),
[docs/two-sizes-design.md](../../docs/two-sizes-design.md)): `n` is now the length of the
sorted array only, and the number of needles arrives separately. The kernel was
changed to match and nothing else.

- **Idea:** the path is still chosen by `n`; "the last full group", which
  leftovers are covered by, is computed from the new count. A path needs one
  full group, so the entry also tests the count and sends a call with too few
  to a narrower path.
- **check:** pass, 141,872 cases (`--unlocked`: the spec and the judge differ from `main`)
- **compare:** not applicable: the previous kernel has another signature
- **Kept:** yes
- **Notes:** no instruction inside a search loop changed. With fewer than 64 needles
  against an array of 64 or more, needles go one at a time through `Lscalar`:
  correct, and the obvious thing for a tuning run to improve. With 64 to 127
  needles against an array of 1,048,576 or more the 64-needle path is used; its
  32-bit bases reach any array below 2^32 elements. Indicative timing (unlocked,
  machine in use), 1,000 needles: 8.94x over naive C and 6.53x over NumPy at
  1,048,576 elements; 64 needles: 10.1x and 2.38x.

## Final numbers

These were measured before attempt 11, with as many needles as elements. The search
loops are the same code now.

`uv run judge run searchsorted_f64 --unlocked` on the kernel of attempt 6
(check: pass, 45,915 cases), Apple M1 Pro, NumPy 2.5.5, 30 samples per point,
with other agents timing on the machine:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 22.3 ns | 107.3 ns | 4.81x [4.77 – 4.83] | 637.0 ns | 28.6x [28.34 – 28.70] |
| 64 | 89.3 ns | 738.9 ns | 8.28x [8.25 – 8.29] | 2.2 us | 24.7x [24.55 – 24.95] |
| 256 | 491.6 ns | 4.3 us | 8.76x [8.75 – 8.78] | 10.2 us | 20.7x [20.57 – 21.57] |
| 1,024 | 2.6 us | 23.8 us | 9.20x [9.18 – 9.22] | 50.2 us | 19.4x [19.34 – 19.55] |
| 4,096 | 12.9 us | 125.1 us | 9.69x [9.67 – 9.70] | 257.7 us | 20.0x [19.90 – 20.07] |
| 16,384 | 68.4 us | 649.9 us | 9.50x [9.47 – 9.52] | 1.29 ms | 18.9x [18.81 – 18.98] |
| 65,536 | 533.6 us | 4.31 ms | 8.07x [8.06 – 8.35] | 6.08 ms | 11.4x [11.10 – 11.41] |
| 262,144 | 2.70 ms | 22.50 ms | 8.32x [8.28 – 8.33] | 27.69 ms | 10.2x [10.18 – 10.27] |
| 1,048,576 | 15.79 ms | 128.23 ms | 8.12x [7.99 – 8.54] | 135.73 ms | 8.59x [8.55 – 8.81] |
| 4,194,304 | 143.16 ms | 1518.79 ms | 10.6x [10.55 – 10.70] | 733.24 ms | 5.12x [5.10 – 5.17] |
| 16,777,216 | 850.63 ms | 10698.03 ms | 12.6x [12.52 – 12.61] | 3564.78 ms | 4.19x [4.17 – 4.20] |
| overall | | | 8.70x [8.68 – 8.75] | | 13.3x [13.26 – 13.36] |

NumPy is timed from Python and includes its per-call overhead.

Not tried, and why:

- A step that classifies against three pivots held in registers in place of
  the first two steps: estimated at about 10% fewer instructions at 1,024,
  but attempt 10 shows the path is not limited by instruction count.
- 128 searches per group with 16-bit bases (n <= 65,536 only): attempt 10
  puts the sensitivity to the number of searches at 5% to 17% per doubling.
- For arrays far beyond the cache, grouping the needles by the part of `a`
  they fall in, so that a block of needles probes only cache-resident
  elements. It needs a per-needle link stored in `out` and a table of bucket
  heads that would have to live in registers (a kernel has no memory besides
  its buffers), and walking the links is itself one cache miss per needle.
  Too large a change to attempt in this run; it is the only idea here that
  could reduce the number of misses at 4M and 16M instead of overlapping them.

## The right side

`searchsorted_right_f64` (`side='right'`) is a copy of this kernel after
attempt 11 with every `a[i] < key` turned into `a[i] <= key`; see
[its log](../searchsorted_right_f64/attempts.md). An improvement kept here is
worth trying there, and the other way round.

That log also records a gap in what the judge checked, which held for this
kernel too: no generated input above 5,000 elements had a needle equal to an
element, so a wrong comparison on the path for arrays of 1,048,576 elements or
more passed `judge check`. `uv run delivery check` has the cases that catch it.

### The gap closed: needles drawn from the array

[Issue 48](https://github.com/sofeikov/kernel-arena/issues/48) gave the judge a
value class drawn from another argument. The spec declares
`In("v", ..., draws_from="a")`, and the judge searches, at every size and in
every scalar set, for needles drawn from `a`, with `a` holding repeated values
(`ties`) and, above 65,536 elements, random. The two mutations of the
long-array path were made on a copy of this kernel, each comparison turned to
the right side's, and checked with `uv run judge check searchsorted_f64
--unlocked`, once with the spec on `main` and once with the new one. The kernel
was restored after each.

| Changed | Spec on `main` | With needles drawn from `a` |
|---|---|---|
| `Lbig_init` (`fcmgt` to `fcmge`, two instructions) | Passes, 141,872 cases | **Fails**: `n=2097151 arr=start seed=1 scalars=k128 a=ties v=from_a` |
| `Lbig_probe` (`fcmgt` to `fcmge`, two instructions) | Passes, 141,872 cases | **Fails**: `n=1048577 arr=mid1 seed=1 scalars=k128 a=ties v=from_a` |

The first step needs a longer array on this side than on the right: taking the
upper window by mistake only shows when elements equal to the needle lie before
`n - p`, which at 1,048,577 elements is index 1. At 2,097,151 the windows meet
in the middle of the array, inside a run of `ties`. The unmutated kernel passes
145,599 cases.

## Ten-run means, locked

`uv run judge run searchsorted_f64`, locked, ten times, one run at a time, on a machine that was
not fully quiet (one-minute load average between 2.2 and 9.1 when a run started,
about 3 for most; a run the judge flagged as noisy was repeated). Check: pass,
141,872 cases. Mean of the ten, with the lowest and highest run in brackets. These
replace the indicative and single-pass figures above wherever they are quoted.

| Case | vs naive C, all sizes | at 1,048,576 | vs NumPy, all sizes | at 1,048,576 |
|---|---|---|---|---|
| 64 needles | 9.06x [8.89 – 9.27] | 10.2x [9.73 – 10.5] | 5.20x [5.13 – 5.30] | 2.34x [2.28 – 2.38] |
| 1,000 needles | 9.08x [8.98 – 9.14] | 8.83x [8.42 – 9.15] | 9.31x [9.17 – 9.74] | 6.40x [5.94 – 6.78] |

## A run on fewer needles than a group ([issue 39](https://github.com/sofeikov/hotkernels/issues/39))

Locked, on `main` at 2d48c69, Apple M1 Pro, with the self-hosted CI runner and
the desktop on the same machine (one-minute load average 2 to 4).

### What `compare` times

`compare` times the scalar sets `timed_scalar_sets` names: the default (64
needles) and `many` (1,000), across the eleven sizes from 16 to 16,777,216.
Neither reaches the path this run is about. With 64 needles every size from
16 takes a group path (sixteen below 64 elements, `Lv64` from 64, and `Lv64`
again from 1,048,576 through `Lfew`); with 1,000 the leftovers are covered by
redoing a full group. `Lscalar` runs only for a NaN needle, which the timed
data never holds. The sets with 1, 15, 17 and 63 needles are checked but not
timed. So `compare` cannot see a change to the few-needle path, and no attempt
here can satisfy the one rule: a correct change reports exactly 1.000x. That is
a gap in the spec, which is locked;
[issue 65](https://github.com/sofeikov/hotkernels/issues/65) asks for a timed set.

The local timings below come from a C harness outside the judge: the kernel
at `HEAD` and the attempt, each assembled under its own symbol, called in
turn on the same data (uniform `a` sorted, uniform needles, a fresh set of
needles for every call from a pool of 16,384 so the branch predictor cannot
learn them), median of 31 samples of 2,000 calls. They are indicative only.

## 12. Eight needles at once, a short group padded with its last needle

- **Idea:** replace the one-at-a-time branchy search of `Lsmall` with the
  branchless lower bound of attempt 2 on eight needles at once: bases in
  x5-x12, keys in d0-d7, `fcmp` + `csel` per probe, the step size in a
  register and one loop over the steps. From eight needles, groups of eight
  with the leftovers covered by redoing the last eight; below eight, the
  missing lanes load the last needle again (`csel` on the index) and only the
  real results are stored, so no buffer is needed. A group holding a NaN, and
  an empty array, send the whole call to `Lscalar`. It serves every call that
  took `Lsmall`: fewer than 64 needles from 64 elements, fewer than 16 from 16,
  and every call below 16 elements.
- **check:** pass, 145,599 cases
- **compare:** no significant difference, 1.000x [1.000 – 1.000]: no timed call
  takes the path (see above)
- **Kept:** no
- **Notes:** locally, against `HEAD`, nanoseconds per call:

  | n | 1 needle | 4 | 8 | 15 | 32 | 63 |
  |---|---|---|---|---|---|---|
  | 64 | 26 / 23, 1.11x | 106 / 24, 4.5x | 209 / 24, 8.9x | 387 / 47, 8.3x | 822 / 93, 8.8x | 1,613 / 189, 8.6x |
  | 4,096 | 63 / 45, 1.41x | 254 / 45, 5.6x | 489 / 45, 10.9x | 908 / 91, 10.0x | 1,911 / 176, 10.8x | 3,755 / 352, 10.7x |
  | 65,536 | 105 / 95, 1.11x | 431 / 106, 4.1x | 853 / 114, 7.5x | 1,590 / 217, 7.3x | 3,376 / 440, 7.7x | 6,647 / 871, 7.6x |
  | 1,048,576 | 167 / 177, 0.94x | 670 / 193, 3.5x | 1,305 / 209, 6.3x | 2,413 / 398, 6.1x | 5,124 / 800, 6.4x | 10,100 / 1,579, 6.4x |
  | 4,194,304 | 354 / 501, 0.71x | 1,433 / 671, 2.1x | 2,841 / 747, 3.8x | 5,490 / 1,482, 3.7x | 11,705 / 2,960, 4.0x | 22,386 / 5,798, 3.9x |
  | 16,777,216 | 580 / 1,059, 0.55x | 2,317 / 1,250, 1.9x | 4,614 / 1,315, 3.5x | 8,637 / 2,505, 3.4x | 18,982 / 4,945, 3.8x | 36,090 / 9,644, 3.7x |

  Padding costs nothing: one to eight needles take the same time. The one
  loss is a single needle against an array far larger than the cache, where
  the branchy search wins: the processor runs ahead down the predicted half
  and overlaps the cache misses, while the branchless search waits for each
  probe. Attempt 14 deals with it. An earlier harness that cycled through 64
  needle sets made the branchy search look about 3x faster on one needle
  than it is: the predictor learned the cycle. Fresh needles per call fixed it.

## 13. Prefetch both possible next probes (rejected on local timing)

- **Idea:** in attempt 12's step loop, prefetch `a[base + h/2 - 1]` and
  `a[base + h + h/2 - 1]` for every needle before its compare resolves, so
  the next probe is in the cache whichever way the compare goes. Two `prfm`
  per probe; the group pointer is parked in d20 to free a register.
- **check:** not run through the judge (local harness: same results as attempt 12)
- **compare:** not run through the judge; locally against attempt 12, 0.78x
  to 0.86x up to 16,384 elements, 0.81x to 0.98x from 65,536 to 1,048,576,
  and only one needle gains at the largest sizes (1.21x at 4,194,304, 1.25x
  at 16,777,216; 0.95x to 1.10x for four or more)
- **Kept:** no
- **Notes:** in the cache the prefetches are pure cost; with eight searches in
  flight they add little the eight lanes do not already overlap. The single
  needle is better served by the branchy search (attempt 14).

## 14. One needle against a large array goes to the branchy search

- **Idea:** attempt 12, but a call with exactly one needle and an array of
  1,048,576 or more goes to `Lscalar` (three instructions on entry to the
  path). Locally, two needles at 16,777,216 are already even (1,161 / 1,169 ns)
  and three are 1.42x, so only one needle is sent back.
- **check:** pass, 145,599 cases
- **compare:** no significant difference, 1.000x [1.000 – 1.000]: no timed call
  takes the path
- **Kept:** no. The kernel is unchanged on this run's branch. Attempts 12 and
  14 together are on the branch `claude/searchsorted_f64-few-needles-attempt`
  so they can be judged once the spec times few needles.
- **Notes:** locally, against `HEAD`: one needle 0.98x to 1.44x (1.00x and
  0.99x at 4,194,304 and 16,777,216, the same code); four needles 1.9x to
  5.8x; eight to 63 needles 3.5x to 11.8x, the most from 1,024 to 16,384
  elements. At 16 elements with 4 to 15 needles, 3.9x to 7.9x.

The same change applies to `searchsorted_right_f64` with `mi` turned to `ls`
in each `csel` (`a[i] <= key`), and its spec has the same two timed sets.
