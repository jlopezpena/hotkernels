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
