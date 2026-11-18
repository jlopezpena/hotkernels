# maskfill_f64 attempts

This log holds two architectures. ARM64 (Apple M1 Pro) comes first. x86-64 (Intel Core i5-14600KF) follows under [x86-64](#x86-64), with its own numbering.

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

## 10. Software prefetch in the sparse loop

- **Idea:** the sparse path stores to only about 40% of the cache lines of `x`, in order but with gaps; prefetch two lines of `x` per block so the stores at 1M elements and more find their lines present.
- **check:** not run (see notes)
- **compare:** not run through the judge. Private harness at 5% density, ns per element at 4,096 / 262,144 / 1M / 4M: no prefetch 0.065 / 0.079 / 0.108 / 0.144; `pstl1keep` at 512 bytes 0.068 / 0.093 / 0.115 / 0.162; at 1,024 and 2,048 bytes, `pldl1keep` at 1,024 and `pstl2keep` at 2,048 and 4,096 all between those and 0.20 at 4M.
- **Kept:** no
- **Notes:** slower at every size and distance. Fetching every line costs more than the misses it hides, since most lines are never needed.

## 11. Test the first block before blending it

- **Idea:** the first block is always blended, so n = 16 never uses the sparse path and a dense n = 64 pays for a run of one block, a test and a second run. Test the first block's two mask words on entry instead: dense starts a run of four blocks at once, sparse goes straight to the tested loop.
- **check:** pass (private fuzzer: no failure)
- **compare:** significantly slower, 0.98x [0.97 – 0.99]. half_selected 0.96x [0.95 – 0.98]: 0.84x at 16, 1.04x at 64. few_selected 1.00x, not significant: 0.96x at 16 and at 64, 1.04x at 1,024.
- **Kept:** no
- **Notes:** at n = 16 the extra mask load, test and run set-up in front of the only block cost more than anything they save, and the entry branch is one more thing to mispredict when the judge cycles through 64 different small inputs.

## 12. Prefetch only the elements that will be stored, a few blocks ahead

- **Idea:** in the sparse loop, read the mask words of the block D blocks ahead, compute the address of each half's selected element the same way, and `prfum pstl1keep` it (an empty half prefetches its previous element, already in cache), so that only the lines that will be written are fetched early.
- **check:** not run (see notes)
- **compare:** not run through the judge: the experiment reads the mask past its end, so it was only timed in a private harness with padded buffers, at 5% density. ns per element at 4,096 / 262,144 / 1M / 4M / 16M: without 0.064 / 0.077 / 0.105 / 0.161 / 0.174; D = 4: 0.096 / 0.112 / 0.132 / 0.160 / 0.185; D = 8, 16 and 32 no better (0.095–0.098 / 0.106–0.110 / 0.129–0.137 / 0.166–0.206 / 0.182–0.202).
- **Kept:** no
- **Notes:** the 13 extra instructions per block cost 40–50% in cache and buy nothing at 4M and 16M. The large sizes are not waiting on late prefetches; a version gated on n would have had nothing to gain.

## 13. Align the two loop heads to 16 bytes

- **Idea:** `.p2align 4` before `Lrun` and `Lloop`.
- **check:** pass
- **compare:** no significant difference, 1.00x [0.99 – 1.02]
- **Kept:** no
- **Notes:** none.

## Not tried, and why

- **Wider or fewer stores in the blend.** The blend is at the core's limit of about three memory operations per cycle (attempt 6); NEON has no store wider than 16 bytes, and attempt 4 showed loop overhead is not the limit.
- **Testing blocks at a finer grain than 16 elements** (four stores per block chosen per 4 elements): about 40 instructions per block, no faster than the blend.
- **A scratch location for unwanted stores** (on the stack) instead of repeating the previous store: a kernel may not write outside its buffers.

## Final result

`uv run judge run maskfill_f64 --unlocked`, check PASS (30,482 cases): 4.61x [4.57 – 4.65] over naive C across both distributions, 17.1x [17.07 – 17.15] over NumPy. half_selected 7.35x [7.24 – 7.45] over naive C (2.90x–3.62x up to 16,384, where naive C's branches are memorised; 16.6x–20.3x from 65,536 up, where they are not). few_selected 2.90x [2.88 – 2.91] (1.60x at 16, 2.36x–2.83x from 64 to 65,536, 3.13x–4.72x from 262,144 up).

# x86-64

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, locked, pinned to one performance core by the judge.

What the compiler does with `naive.c` (`clang -O3 -march=native`): a branch per element. With half selected at random it is mispredicted half the time, 2.4 ns per element at 65,536; with 5% selected it is 0.12 ns per element. NumPy takes 3.0 ns per element at 1,048,576 with half selected.

## x86-64 1. Baseline: widen four mask bytes, blend and store back

- **Idea:** the ARM64 kernel's scheme, four elements per step. `vpmovsxbq` widens four mask bytes to four 64-bit lanes, `vpcmpeqq` against zero makes the lane mask, and the four elements are loaded, combined with the value by `vpand`, `vpandn` and `vpor`, and stored back. An element that is not selected is rewritten with its own bits. The value's bits are only ever copied. A mask byte is true when it is not zero. The last 1 to 3 elements take a scalar loop with a branch.
- **check:** pass
- **compare:** not run (first version). `run`: 2.80x over naive C overall (half_selected 5.84x, few_selected 1.34x), 11.6x over NumPy.
- **Kept:** yes
- **Notes:** 0.10 ns per element from 1,024 to 65,536, whatever the mask. `vblendvpd` was not used for the blend: measured on this core it runs once per cycle, and three logical instructions run three per cycle.

## x86-64 2. Blocks with at most four selected elements are stored one by one, with no load

- **Idea:** the mask bits of 32 elements come from one `vpcmpeqb` and one `vpmovmskb`. A block with at most four set bits is not blended: `tzcnt` and `blsr` take the positions out, and the value is stored to each, a fixed four stores. `tzcnt` sets the carry flag when no bit is left, and a `cmovnc` then leaves the store address where it was, so the extra stores write the value to a selected element a second time, or, in a block with no bit at all, to eight bytes below the stack pointer. No element is loaded, and a cache line with no selected element is not touched.
- **check:** pass
- **compare:** significantly faster, 1.366x [1.349 – 1.384]
- **Kept:** yes
- **Notes:** few_selected 1.92x (2.05x at 64, 2.4x at 1,024, 2.0x at 65,536, 1.6x at 1,048,576, 1.44x at 16,777,216). half_selected 0.97x: 0.93x to 0.96x up to 1,024 for the count and the branch each block now pays, nothing from 65,536. In sparse blocks no unselected element is written, but dense blocks still rewrite them, so the caller's constraint is unchanged.

## x86-64 3. The design's candidate: a masked store in place of the blend (`vmaskmovpd`)

- **Idea:** `vpmovzxbq`, a subtraction from zero to put the selection in the sign bit, and `vmaskmovpd`. Three instructions per four elements, no load of x, and unselected elements are not written at all.
- **check:** pass
- **compare:** significantly faster, 1.062x [1.058 – 1.065]
- **Kept:** no. Not mine to keep: it removes a caveat the ARM64 kernel has (decision 1 in `docs/kernel-findings.md`, the rewrite of unselected elements) and would make the binding's constraint differ by architecture. Reported for a person to decide.
- **Notes:** the two halves of the result point opposite ways. half_selected: 1.55x at 64, 1.64x at 256, 1.63x at 1,024, 1.58x at 4,096, then **0.86x at 65,536, 0.87x at 1,048,576** and 0.94x at 16,777,216. few_selected: 1.00x to 1.03x, since sparse blocks do not reach this code. Without the load of x nothing announces to the processor which lines the stores are about to need, so beyond the first-level cache each masked store waits for its line. A prefetch for the store's lines was not tried, the variant not being a candidate to keep. Masked stores are also reported slow on some AMD processors (open issue 4 of the design), untested here.

## x86-64 4. Lane masks of a dense block from its byte mask, stored below the stack pointer

- **Idea:** the block's `vpcmpeqb` result is 0xff where an element stays. Stored below the stack pointer and read back four bytes at a time with `vpmovsxbq`, it gives the lane mask in one vector instruction and not two.
- **check:** pass
- **compare:** no significant difference, 1.001x [0.988 – 1.013]
- **Kept:** no
- **Notes:** half_selected 0.91x at 4,096 and 0.93x at 65,536, 1.22x at 16.

## x86-64 5. Lane masks of a dense block from a 16-entry table

- **Idea:** the dense block's mask bits are already in a general register. Four bits index a table of 16 lane masks (512 bytes), so the mask is one load with no vector instruction, and a group is `vpandn` with the elements, `vpand` with the value, `vpor` and the store.
- **check:** pass (after one failure: the table's base register was also the sparse path's store address)
- **compare:** significantly faster, 1.021x [1.020 – 1.023]
- **Kept:** yes
- **Notes:** half_selected 1.05x to 1.08x up to 1,024, 1.04x at 65,536, nothing at 4,096 or from 1,048,576. No size slower.

## x86-64 6. Bring the blocks to a 32-byte boundary

- **Idea:** every load and every store of the blend straddles two cache lines at every second group when x starts 16 bytes into a line, which it usually does. A micro-benchmark outside the judge puts that at 15% (0.094 against 0.111 ns per element at 1,024 and at 65,536). Up to three elements are done one at a time with `cmovne` first.
- **check:** pass
- **compare:** no significant difference, 1.007x [0.998 – 1.016]
- **Kept:** no
- **Notes:** half_selected 1.05x (1.21x at 65,536, 1.06x at 1,024, nothing at 4,096 or 1,048,576), few_selected 0.97x (0.82x at 64, 0.90x at 256): the sparse path has no load to align and pays for the head.

## x86-64: where it stands

Final `judge run`:

`half_selected`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 3.4 ns | 6.8 ns | 2.01x [1.99 – 2.02] | 313.7 ns | 92.3x [91.7 – 93.0] |
| 64 | 7.3 ns | 22.5 ns | 3.10x [3.09 – 3.11] | 358.5 ns | 49.4x [49.3 – 49.4] |
| 256 | 26.2 ns | 84.1 ns | 3.21x [3.20 – 3.21] | 540.6 ns | 20.6x [20.6 – 20.6] |
| 1,024 | 100.9 ns | 333.4 ns | 3.30x [3.30 – 3.31] | 1.3 us | 13.3x [13.3 – 13.4] |
| 4,096 | 369.1 ns | 1.3 us | 3.55x [3.55 – 3.55] | 4.3 us | 11.6x [11.6 – 11.7] |
| 16,384 | 2.0 us | 5.9 us | 2.97x [2.96 – 2.99] | 40.5 us | 20.4x [20.4 – 20.4] |
| 65,536 | 6.3 us | 156.9 us | 24.7x [24.7 – 24.8] | 195.3 us | 30.8x [30.7 – 30.8] |
| 262,144 | 34.9 us | 657.5 us | 18.8x [18.8 – 18.9] | 798.8 us | 22.9x [22.9 – 23.0] |
| 1,048,576 | 174.5 us | 2.66 ms | 15.2x [15.0 – 15.5] | 3.21 ms | 18.4x [18.2 – 18.7] |
| 4,194,304 | 1.26 ms | 10.66 ms | 8.48x [8.41 – 8.55] | 12.89 ms | 10.2x [10.2 – 10.3] |
| 16,777,216 | 6.31 ms | 42.64 ms | 6.76x [6.74 – 6.79] | 51.89 ms | 8.23x [8.20 – 8.26] |
| overall |  |  | 5.87x [5.86 – 5.88] |  | 20.9x [20.8 – 20.9] |

`few_selected`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 3.3 ns | 3.7 ns | 1.10x [1.08 – 1.12] | 259.0 ns | 77.5x [77.2 – 77.7] |
| 64 | 3.6 ns | 9.4 ns | 2.61x [2.60 – 2.62] | 305.2 ns | 84.8x [84.8 – 84.9] |
| 256 | 11.1 ns | 30.6 ns | 2.74x [2.72 – 2.75] | 352.3 ns | 31.6x [31.6 – 31.6] |
| 1,024 | 37.7 ns | 122.5 ns | 3.25x [3.25 – 3.26] | 593.3 ns | 15.7x [15.7 – 15.8] |
| 4,096 | 216.1 ns | 469.1 ns | 2.17x [2.17 – 2.17] | 1.4 us | 6.60x [6.58 – 6.61] |
| 16,384 | 808.8 ns | 2.0 us | 2.41x [2.41 – 2.42] | 4.6 us | 5.73x [5.73 – 5.73] |
| 65,536 | 3.0 us | 7.9 us | 2.67x [2.65 – 2.69] | 32.5 us | 11.0x [10.9 – 11.1] |
| 262,144 | 12.6 us | 99.7 us | 7.88x [7.84 – 7.93] | 183.3 us | 14.5x [14.4 – 14.5] |
| 1,048,576 | 103.0 us | 451.6 us | 4.38x [4.38 – 4.39] | 752.8 us | 7.31x [7.29 – 7.32] |
| 4,194,304 | 739.1 us | 1.82 ms | 2.46x [2.45 – 2.47] | 3.06 ms | 4.13x [4.12 – 4.15] |
| 16,777,216 | 4.42 ms | 7.25 ms | 1.64x [1.64 – 1.64] | 12.30 ms | 2.78x [2.78 – 2.79] |
| overall |  |  | 2.67x [2.67 – 2.68] |  | 12.9x [12.9 – 12.9] |

All distributions: 3.96x [3.95 – 3.97] over naive C, 16.4x [16.4 – 16.4] over NumPy

The kernel writes no buffer other than x, so the judge has no close-placement table for it.

What holds it:

- **half_selected: 0.09 to 0.10 ns per element up to 65,536,** about two cycles per four elements, and 0.13 to 0.17 at 262,144 and 1,048,576. Knock-out variants in a micro-benchmark at 1,024 elements (no store, no load of x, no widening of the mask) each ran at 1.6 to 2.0 cycles, so no single part is the limit. The naive C is so far behind here (a mispredicted branch every second element) that the speedup over it is 15x to 25x from 16,384 to 1,048,576.
- **few_selected:** the sparse block, 0.035 to 0.05 ns per element up to 262,144, 2x to 8x over the naive C from 64 elements up.
- **16,777,216:** memory. Every line of x is read and written back with half selected.

The candidates the design lists for this kernel (`docs/x86-64-design.md`):

| Candidate | Outcome |
|---|---|
| `vmaskmovpd` stores only the selected elements | Measured, not kept (attempt 3): 1.6x faster up to 4,096 elements with half selected, 0.86x at 65,536 and 1,048,576. It removes the rewrite caveat, so the decision is a person's |
| The same load, blend and store | Kept (attempts 1 and 5), with the lane mask from a table |
| Masked loads and stores for leftover elements | Not tried: open issue 3 of the design. The last 1 to 3 elements take a scalar loop |
| Non-temporal stores | Not tried: the kernel stores to the buffer it has just read |
| Alignment of data | Rejected (attempt 6) |
| Alignment of code | The block loop starts on a 64-byte boundary. Not varied |
| An output just behind the input | Does not apply: one buffer |
| Instructions slow on other x86-64 processors | None in the kept kernel. Attempt 3's `vmaskmovpd` is one |

Left untried: a prefetch of the store's lines for attempt 3; a second sparse tier for blocks with five to eight selected elements (the judge times 5% and 50%, so `compare` could not see it).

Found here and worth trying on ARM64: nothing new; the store-only sparse block is the ARM64 kernel's own attempt 7.
