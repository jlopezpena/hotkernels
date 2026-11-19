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

## 3. Blocks of 16 that are flat, rising or general

- **Idea:** the idea of the x86-64 kernel. With M the maximum so far, a block
  of 16 elements is *flat* if no element is above M (eight `cmgt` against M
  in both lanes, an `orr` tree, `umaxv`, `fmov`, `cbnz`; the block is M
  sixteen times, four `stp` of q registers), *rising* if no element is below
  the one before it and the element before the block is M (the same test
  against eight overlapping `ldur` one element back; the block is a copy),
  and otherwise done the general way of attempt 2, four groups of four. The
  flat and rising loops run until a block fails their test. After a general
  block the next is tried flat if M did not change, rising if the block's
  last element is M, and done the general way untested otherwise; after a
  block that failed the flat test the flat test is tried again at once. Fewer
  than 16 elements: groups of four, then single elements.
- **check:** pass
- **compare:** significantly faster, 2.33x [2.31 – 2.35]
- **Kept:** yes
- **Notes:** `random`: 1.11x at 16 (one general block either way), 1.24x at
  64, 2.4x to 3.8x from 256 up; the flat loop runs at 0.104 ns per element at
  4,096 (5.3 cycles per block, about what eight 16-byte stores cost on this
  core by `cumsum_i64`'s log). `rising`: 2.3x to 3.0x from 64 up; the rising
  loop runs at 0.135 ns per element at 4,096 (6.9 cycles per block). 2.0x to
  2.3x at 4,194,304 and 16,777,216, about 0.19 ns per element, which is the
  speed of a copy from memory.

## 4. Rising test: half the shifted vectors from `ext`

- **Idea:** the rising loop loads 12 vectors per block (eight for the block,
  eight shifted back one element, as four `ldp` and eight `ldur`). The
  shifted vector of the second register of each `ldp` pair is the `ext` of
  the pair, so build those four with `ext` and load only the other four.
- **check:** pass
- **compare:** significantly faster, 1.03x [1.03 – 1.04]
- **Kept:** yes
- **Notes:** `rising` 1.14x to 1.16x from 256 to 16,384 (0.118 ns per
  element at 4,096, 6.0 cycles per block against 6.9), 1.08x at 64, nothing
  from 65,536 up; `random` unchanged.

## 5. Rising test: seven of the eight shifted vectors from `ext`

- **Idea:** attempt 4 with `ext` for every shifted vector but the first, so
  one `ldur` per block and seven `ext`.
- **check:** pass
- **compare:** significantly slower, 0.98x [0.98 – 0.98]
- **Kept:** no
- **Notes:** `rising` 0.86x to 0.93x from 64 to 16,384: 23 vector operations
  per block on four vector pipes is more than the loads it saves. Four loaded
  and four built (attempt 4) is the better balance.

## 6. Every general block in the vector registers

- **Idea:** a general block as a vector prefix maximum: each of the eight
  vectors against the input one element back (overlapping `ldur`; for lane 0
  that reaches into the previous block, which M already covers), then against
  the vector 1, 2 and 4 back, in place from the last vector down, then each
  against M in both lanes; `cmgt` and `bit` per maximum, 67 vector
  operations per block against 56 `cmp`/`csel` in the integer registers. M
  goes in with a `dup` and comes out with a `mov` from lane 1.
- **check:** pass
- **compare:** significantly faster, 1.014x [1.011 – 1.016], but not kept
  (see attempt 7)
- **Kept:** no
- **Notes:** 1.36x at 16 on both distributions, 1.11x at 64 `random`; but
  0.88x at 256, 0.93x at 1,024 and 0.97x at 4,096 `random`, 0.96x to 0.97x
  from 64 to 16,384 `rising`. A block takes some 30 cycles from M in to M out
  against about 10 in the integer registers, and that is what a mispredicted
  exit from the flat loop waits on. The gain at 16 is the first block, which
  attempt 7 takes alone.

## 7. The first block alone in the vector registers

- **Idea:** keep attempt 3's integer general path everywhere but at the start
  of `x`. There M is `x[0]`, so the first block's prefix maximum needs no
  step against M: lane 0 of the first shifted vector is `x[0]` itself
  (`ld1r`), then the steps 1, 2 and 4 vectors back; 50 vector operations, M
  out with a `mov` from lane 1.
- **check:** pass
- **compare:** significantly faster, 1.053x [1.052 – 1.053]
- **Kept:** yes
- **Notes:** 1.69x at 16 on both (8.9 ns to 5.3), 1.12x at 64 `random`,
  1.06x at 64 `rising`. 0.97x to 0.98x on `rising` from 1,024 to 16,384,
  which runs the same rising loop as before: probably where the loop now
  sits in memory.

## 8. Align the flat and rising loops to 64 bytes

- **Idea:** see whether attempt 7's 0.97x on `rising` from 1,024 to 16,384
  was where the loops sit.
- **check:** pass
- **compare:** significantly slower, 0.997x [0.996 – 0.998]
- **Kept:** no
- **Notes:** no size faster; 0.97x at 64 `rising` (the padding is executed
  on the way in).

## 9. First block: Sklansky steps instead of Kogge-Stone

- **Idea:** after the first step each lane holds the maximum of a run of
  elements ending at its own, so a maximum of two vectors whose runs meet is
  again a run. The odd vectors against the one before, then 2 and 3 against
  1 and 6 and 7 against 5, then 4 to 7 against 3: 12 maxima after the first
  step instead of 17, no `dup` needed, the same depth.
- **check:** pass
- **compare:** significantly faster, 1.019x [1.018 – 1.020]
- **Kept:** yes
- **Notes:** 1.13x at 16 on both (5.3 ns to 4.7), 1.07x at 64 and 1.04x at
  256 `rising`, 1.01x at 64 `random`.

## 10. Every general block in the vector registers, M brought in late

- **Idea:** attempt 6 again with the cost of its chain taken out. The block's
  prefix maximum uses the Sklansky steps of attempt 9; M (kept in v31 in
  both lanes, not moved over from x3) is brought into vector 3 before the
  last step, which carries it on to 4 to 7, and into vectors 0 to 2 one by
  one. So the chain from block to block is two maxima and a `dup` from lane
  1 of vector 7, and a block is 24 maxima (48 vector operations) and a `dup`
  against 56 `cmp`/`csel`. x3 is updated from lane 1 off the chain, for the
  branches and the tail.
- **check:** pass
- **compare:** significantly faster, 1.020x [1.018 – 1.021]
- **Kept:** yes
- **Notes:** `random`: 1.27x at 64 (21.1 ns to 16.6), 1.09x at 256, 1.07x at
  1,024, 0.99x at 16,384. `rising`: 1.03x at 4,096, nothing elsewhere.

## 11. Enter the rising loop after a general block only if the block rose

- **Idea:** outside the judge (a loop timer over one array, see the end of
  this log), data with a new maximum at a random half of the elements runs at
  0.33 ns per element at 4,096 but 0.69 at 2,000,000, slower than the naive
  C's 0.63: every block is general, its last element is M half the time, so
  the branch into the rising loop goes either way at random once the array is
  too long for the predictor to learn, and the rising test then fails. The
  first step of the general block already compares every element with the
  one before it; `orr` those eight masks, `umaxv`, `fmov` to w10, and try the
  rising loop only if none was true (the first block keeps the old rule).
- **check:** pass
- **compare:** significantly slower, 0.996x [0.994 – 0.998]
- **Kept:** no
- **Notes:** outside the judge the random-half data at 2,000,000 went from
  0.69 to 0.33 ns per element, but 1 0 2 1 3 2 ... (every block general,
  never rising) went from 0.28 to 0.33 for the nine extra vector operations.
  In the judge 0.92x at 64 `random` and 0.97x at 256; neither distribution
  has data that this helps. Not kept under the one rule; the case it fixes
  is listed under "Where it stands".

## 12. Try the flat loop after every general block that is not rising

- **Idea:** after a general block in which M changed and whose last element
  is not M, try the flat test rather than another general block, since in
  random data a new maximum is usually alone.
- **check:** pass
- **compare:** significantly faster, 1.009x [1.006 – 1.011]
- **Kept:** no
- **Notes:** 1.13x at 64 and 1.05x at 256 `random`, nothing else. But it
  makes every block of data that is neither flat nor rising pay a flat test
  that fails: outside the judge 1 0 2 1 3 2 ... went from 0.28 to 0.37 ns
  per element and the random-half data from 0.34 to 0.38 at 16,384 and 0.69
  to 0.75 at 2,000,000. The gain is all in the first blocks, which attempt
  13 takes without that cost.

## 13. After the first block, the flat loop unless the block ends on M

- **Idea:** attempt 12 for the first block only. After it, the rising loop
  if its last element is M, the flat loop otherwise; the general loop keeps
  its rule (flat only if M did not change), so data that is neither flat nor
  rising pays one failed flat test, not one per block.
- **check:** pass
- **compare:** significantly faster, 1.018x [1.017 – 1.019]
- **Kept:** yes
- **Notes:** `random` 1.07x at 16, 1.12x at 64, 1.04x at 256; `rising` 1.07x
  at 16, 1.05x at 64, 1.03x at 256, 1.02x at 1,024. At 16 the first block
  no longer saves M for a comparison afterwards.

## 14. `stnp` in the flat and rising loops

- **Idea:** store the flat and rising blocks with `stnp` in place of `stp`
  (the general blocks keep `stp`). `cumsum_i64` found no difference; tried
  here because the flat loop is bound by its stores more than by its
  arithmetic (outside the judge, at 1,024 to 4,096 constant elements,
  dropping half its compares gained under 5% and dropping half its stores
  15%).
- **check:** pass
- **compare:** significantly faster, 1.007x [1.003 – 1.011]; rerun 1.007x
  [1.004 – 1.010]
- **Kept:** yes
- **Notes:** the whole gain is 16,384 `random`: 1.14x [1.04 – 1.25] and
  1.17x [1.11 – 1.23]; every other size 1.00x. Outside the judge the same:
  16,384 random elements 0.126 to 0.099 ns per element, rising 0.126 to
  0.119, constant 0.126 to 0.123, and nothing from 65,536 up (0.17 ns per
  element in the second-level cache, 0.23 at 4,000,000).

## 15. Flat test over two blocks at once

- **Idea:** while 32 or more elements are left, the flat loop tests 32
  elements with one reduction and one branch and stores 32; if the test
  fails, the first of the two blocks goes through the 16-element test as
  before. Halves the `umaxv`, `fmov`, `cbnz` and pointer updates per element.
- **check:** pass
- **compare:** significantly slower, 0.949x [0.943 – 0.956]
- **Kept:** no
- **Notes:** `random` 0.79x to 0.84x from 64 to 4,096 and 0.75x at
  1,048,576: every new maximum now costs a failed 32-element test, a reload
  and the 16-element test; nothing faster anywhere (the loop is held by its
  stores, not by its overhead).

## 16. `addp` in place of `umaxv` to reduce the test masks

- **Idea:** the masks are 0 or all ones per 64-bit lane, so `addp d, v.2d`
  is zero only when both lanes are; it has a shorter latency than `umaxv`,
  which would let a mispredicted test resolve sooner.
- **check:** pass
- **compare:** no significant difference, 1.000x [0.999 – 1.002]
- **Kept:** no

## 17. Rising test: every shifted vector from an `ldp` one element back

- **Idea:** `ldp` of q registers cannot take an offset of -8, but from a
  second pointer `x0 - 8` four `ldp` give all eight shifted vectors, so the
  rising loop has no `ext` (16 vector operations per block against 20) and
  eight load instructions instead of eight plus four.
- **check:** pass
- **compare:** significantly slower, 0.985x [0.977 – 0.992]
- **Kept:** no
- **Notes:** `rising` 0.91x at 64, 0.91x to 0.94x at 256 to 16,384 (wide
  intervals); half of these loads straddle a cache line, as in attempt 3's
  eight `ldur`.

## Where it stands

Final `judge run` (another kernel run timing in another worktree): 3.07x
[3.02 – 3.08] over naive C on `random`, 3.32x [3.27 – 3.33] on `rising`,
3.19x [3.15 – 3.20] over both; 18.3x [18.09 – 18.37] over NumPy.

- 16: 1.47x on both (4.3 ns): the first block as a vector prefix maximum.
- 64 to 16,384: 2.1x to 6.1x. The flat loop runs at 0.10 ns per element in
  the first-level cache, about 5.3 cycles per block of 16, held by its eight
  16-byte stores; the rising loop at 0.12, about 6 cycles.
- 65,536 to 16,777,216: 2.1x to 3.3x, at the speed of a copy in the
  second-level cache (0.17 ns per element outside the judge) and from
  memory (0.19 to 0.24). The judge's 65,536 and 262,144 moved by up to 1.7x
  between runs with the other run active (11 µs and 16 µs at 65,536).
- Output 16 bytes behind its input: no stall; the close placement is as fast
  as the apart one or faster at every size.

Outside the judge (a loop timer over one array timed again and again, best of
five; ns per element; the naive C is 0.62 to 0.63 on every kind of data):

| data | 16,384 | 2,000,000 |
|---|---|---|
| random | 0.099 | 0.240 |
| sorted ascending | 0.122 | 0.241 |
| constant | 0.119 | 0.245 |
| noise of 1,000 on a rising trend | 0.224 | 0.442 |
| 1 0 2 1 3 2 ... (every block general) | 0.293 | 0.298 |
| a new maximum at a random half of the elements | 0.346 | 0.690 |

The last row is the one case found slower than the naive C: at 2,000,000
elements, too many for the branch predictor to learn, every block is general
and the choice of trying the rising loop after it (last element equal to M)
goes either way at random, and the rising test then fails. Attempt 11 fixes
it (0.33 ns per element) for a cost the judge sees at 64 and 256 `random`.

Not tried:

- **Stores past the caches with `dc zva`:** `cumsum_i64` measured it at 70
  times slower per line.
- **Software prefetch:** no gain anywhere for `cumsum_i64`; the hardware
  prefetcher follows both streams.
- **The general path in the integer registers alongside the vector one**
  (two independent halves of a block on separate pipes): the chain between
  the two halves would cross between register files twice per block.
