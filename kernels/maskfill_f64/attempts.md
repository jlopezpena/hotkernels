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

## 2. NEON blend, 16 elements per iteration

- **Idea:** load 16 mask bytes, `cmtst` to 0x00/0xFF, spread each pair of bytes over a 2x64-bit vector with one `tbl` per vector (eight index constants), `bsl` between the broadcast value and the loaded data, store back. Leftover handled by one overlapping final block of 16; n < 16 stays scalar.
- **check:** pass
- **compare:** significantly faster, 2.66x [2.64 – 2.69] (half_selected 2.73x, few_selected 2.58x)
- **Kept:** yes
- **Notes:** about 0.107 ns per element in L1 (5.5 cycles per 16 elements), the same for both distributions. 3.7x at 256, falling to 1.6–2.2x at 16M where memory speed takes over.

## 3. Skip blocks of 16 whose mask bytes are all zero

- **Idea:** load the 16 mask bytes as two 64-bit words first (`ldp x4, x5`), `orr` them, and `cbz` past the whole blend when nothing is selected. Saves eight loads and eight stores for 44% of the blocks at few_selected.
- **check:** pass
- **compare:** significantly faster overall, 1.04x [1.04 – 1.04], but not kept. half_selected 0.97x [0.96 – 0.97]; few_selected 1.12x [1.11 – 1.12], made of 1.12x–2.19x from 16 to 65,536 and 0.68x [0.67 – 0.69] at every size from 262,144 up.
- **Kept:** no
- **Notes:** rejected on judgement although the overall number passes the rule. The whole gain comes from sizes where the benchmark input is small enough for the branch predictor to memorise which blocks are empty (the same inputs are timed over and over). Where the pattern cannot be memorised (262,144 and larger) the skip branch mispredicts on about 44% of blocks and the kernel is 32% slower, even though it touches less memory; half_selected pays 3% for the test at every size. On masks seen for the first time the large-size numbers are the representative ones. The skip would be a real win for masks that are much sparser than 5% or clustered; the benchmark has no such distribution. To restore it: the four instructions above at the top of the loop plus a label before the pointer advance.

## 4. Unroll to 32 elements per iteration

- **Idea:** two blocks of 16 per iteration to halve the loop overhead (pointer advances, counter, branch); 16 to 31 elements handled by two overlapping blocks of 16.
- **check:** pass
- **compare:** significantly slower, 0.95x [0.94 – 0.95]
- **Kept:** no
- **Notes:** every size from 64 up is within 1% of the 16-per-iteration loop (0.98x–1.01x), so loop overhead is not what limits it. n=16 dropped to 0.53x because it now took the two-overlapping-blocks path, whose second block reloads data the first has just stored.

## 5. Software prefetch in the blend loop

- **Idea:** `prfm` two cache lines of `x` ahead of the loads, to help the sizes that do not fit in L1.
- **check:** not run (see notes)
- **compare:** not run through the judge. Timed in a private harness (min of 15, n = 1,024 / 65,536 / 4,194,304) for `pldl1keep`, `pstl1keep` and `pldl2keep` at distances 256, 512, 1,024 and 2,048 bytes: no variant beat the loop without prefetch at any size (0.105 / 0.155 / 0.179 ns per element without; 0.106 / 0.159–0.170 / 0.18–0.27 with).
- **Kept:** no
- **Notes:** the hardware prefetcher already follows the stream; the extra instructions only cost.

## 6. Where the blend loop's time goes; other orders of loads and stores

- **Idea:** find the limit of the blend loop by removing parts of it (private harness, n = 1,024, not judge runs), then try to schedule around it.
- **check:** not run (these variants are deliberately wrong or equivalent reorderings that were not kept)
- **compare:** not run through the judge. Per 1,024 elements: full loop 107 ns; without `tbl` 108; without `bsl` 108; without the stores 88; loads and stores only (no mask work) 103; stores only 81. Reorderings: half block at a time (load, blend, store, twice) 109; single `ldr`/`str` instead of pairs 109; four-register `ld1`/`st1` 145.
- **Kept:** no
- **Notes:** the loop is limited by memory operations, not by the mask work: eight 16-byte loads and eight 16-byte stores per 16 elements run at about three per cycle, and the stores alone at two per cycle. A blend that loads and stores every element cannot go below about 0.10 ns per element in L1 on this core. Going faster needs fewer loads and stores, which is attempt 7.

## 7. Sparse blocks: one store per 8 elements, no load, chosen per block

- **Idea:** read the 16 mask bytes as two 64-bit words. If each word has at most one byte set (`w & (w - 1) == 0` for both), do exactly two 8-byte stores of the value: `clz` gives the selected element's address, and a half with nothing selected repeats the store to the most recently selected element (a `csel` on the address, no branch). Nothing is loaded from `x` and unselected elements are not written. A block that fails the test starts a run of blended blocks that is not tested: 1 block, then 4, 16 and 64 while each run is followed directly by another failing block. After a run, blocks with no mask byte set are skipped until one is found to restart the address chain. The first block is always blended.
- **check:** pass (plus 3,000,000 random cases in a private fuzzer over densities 0 to 1, clustered masks and misaligned buffers: no failure)
- **compare:** significantly faster, 1.17x [1.16 – 1.19]. few_selected 1.38x [1.38 – 1.40]: 1.26x at 64, 1.45x–1.56x from 256 to 4,096, 2.55x at 16,384, 1.79x at 65,536, 1.41x at 262,144, 1.13x–1.14x at 1M and 4M, 1.05x at 16M; 0.97x at 16. half_selected 0.98x [0.98 – 1.01], not significant: 0.93x at 64, 0.97x at 16 and 256, within 2% above that.
- **Kept:** yes
- **Notes:** unlike attempt 3 this is faster at the large sizes too, where the per-block branch (taken for about 11% of blocks at 5% density) cannot be memorised. The sparse path is about 3.3 cycles per 16 elements against 5.5 for the blend. For the caller: on sparse blocks this path writes only selected elements, like the C loop; the store-back caveat at the top still applies to every blended block. It relies on mask bytes being exactly 0 or 1 (a byte such as 2 would give a wrong address in the sparse path), which is what the spec promises and what clang's code for the C loop also assumes.

## 8. Sparse path: two address chains, test with two branches, lone dense block blended in place

- **Idea:** three changes to the sparse-mode loop, measured together. (a) Give each half of the block its own "most recently selected" address (one for first halves, one for second halves, each kept with the bias its store uses), which removes the second pointer and shortens the carried dependency to one `csel`. (b) Test each half with `tst` + `b.ne` instead of two `and`, an `orr` and a `cbnz`. (c) When a block fails the test and does not directly follow a run, blend it in place and go straight back to testing, instead of going through the run set-up, the zero-block scan and the chain restart.
- **check:** pass (private fuzzer: 3,000,000 cases, no failure)
- **compare:** significantly faster, 1.04x [1.03 – 1.05]. few_selected 1.08x [1.07 – 1.08]: 1.04x–1.08x from 256 to 65,536, 1.27x at 262,144 and 1M, 1.09x at 4M, 0.97x at 64. half_selected 1.00x, no size significantly different.
- **Kept:** yes
- **Notes:** in a private harness (a) and (b) alone changed nothing measurable at 5% density (69.3 against 68.8 ns per 1,024); the gain is from (c). At 0.5% density the sparse loop runs at about 2.1 cycles per 16 elements; at 5% the 11% of blocks that fail the test cost far more than their share, which is why making them cheaper pays most where the branch mispredicts (262,144 and up).

## 9. Second sparse tier: up to two selected per half, four stores

- **Idea:** a block that fails the one-per-half test but has at most two selected in each half (`w & (w - 1)` has at most one bit left) gets four stores, to the highest (`clz`) and lowest (`rbit` + `clz`) selected element of each half, still without loading `x`. Only blocks with three or more in a half are blended. At 5% density that leaves 1.1% of blocks for the blend instead of 11%. A run of untested blended blocks now starts at the third block in a row that failed the first test (it was the second), so that two-per-half blocks do not immediately switch the loop to blending.
- **check:** pass (private fuzzer: 3,000,000 cases, no failure)
- **compare:** significantly faster, 1.01x [1.00 – 1.02]; repeated: 1.01x [1.00 – 1.02], also significant. few_selected 1.03x [1.03 – 1.04]: 1.02x–1.09x from 1,024 to 4M, but 0.96x at 256 and 0.98x at 64. half_selected 0.99x, not significant overall, 0.96x at 64.
- **Kept:** yes
- **Notes:** marginal. Two variants were measured in a private harness before this one, at n = 262,144 where the branches cannot be memorised, against the blend-only loop's 0.16 ns per element at every density. (a) Four-store tier with runs never started by two-per-half blocks: 0.077 at 5%, but 0.178–0.186 at 12–16% density, 12–16% worse than blending everything. (b) Runs started by the second failing block in a row, as before: no worse than the blend at any density, but only 0.081 at 5%. The kept rule (third in a row) gives 0.077 at 5%, 0.103 at 8%, 0.147 at 12% and 0.159–0.162 from 15% up, so it is never slower than the blend-only loop by more than noise. Attempt 8 measured the same way: 0.086 at 5%, 0.124–0.129 at 8%.
