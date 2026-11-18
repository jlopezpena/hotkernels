# compress_f32 attempts (arm64)

This log holds two architectures. ARM64 (Apple M1 Pro) comes first. x86-64 (Intel Core i5-14600KF) follows under [x86-64](#x86-64), with its own numbering.

All judge commands in this run were passed `--unlocked`, because the spec is not on `main` yet. Other agents were timing kernels on the same machine, so intervals are wider than usual.

## 1. Scalar baseline

- **Idea:** the naive loop in assembly: store every element at `out[k]`, advance `k` by the mask byte.
- **check:** pass
- **compare:** not run (first version, nothing to compare against)
- **Kept:** yes
- **Notes:** starting point only; expected to match naive C.

## 2. Eight elements per step with a nibble shuffle table

- **Idea:** load 8 mask bytes as a u64; one multiply packs them into an 8-bit mask, two more give the selected counts (scaled to bytes) for all eight and for the low four. Each nibble indexes a 16-entry table of `tbl` patterns; each 4-lane half is compacted and stored whole at the write pointer. The write-pointer chain is one add per 8 elements. Scalar tail for the last 0 to 7.
- **check:** pass
- **compare:** significantly faster, 2.50x [2.48 – 2.50]
- **Kept:** yes
- **Notes:** half_selected 2.45x, few_selected 2.55x; the code is branchless so the two distributions run the same. About 0.125 ns per element in cache (roughly 3.2 cycles per 8 elements).

## 3. Cheaper mask and count extraction

- **Idea:** take both counts from one multiply (`m * 0x0404…04` leaves 4x the running count in every byte: `ubfx` byte 3 and `lsr` byte 7) and take the two nibbles straight from the mask product (`ubfx #56,#4` and `lsr #60`). Two instructions fewer per step, one multiply fewer.
- **check:** pass
- **compare:** significantly faster, 1.09x [1.09 – 1.09]
- **Kept:** yes
- **Notes:** the loop is limited by instruction count, not by a dependency chain: removing 2 of about 22 instructions gave 9%.

## 4. Sixteen elements per loop iteration

- **Idea:** two 8-element blocks per iteration, so the pointer updates and the loop branch are shared (`ldp` for 16 mask bytes, two `ldp q` and one `add` for x). One leftover 8-block, then the scalar tail.
- **check:** pass
- **compare:** significantly faster, 1.06x [1.06 – 1.07]
- **Kept:** yes
- **Notes:** n=16 is 3% slower (one extra branch on the way in); every other size gains 5 to 10%.

## 5. One 256-entry table indexed by the whole 8-bit mask

- **Idea:** multiply by 0x8040201008040201 instead, which puts the mask bits (reversed) in bits 56-63 and leaves bit 55 clear, so a single `lsr #55` gives twice the index of a 32-byte entry holding both halves' shuffle patterns. Replaces `ubfx` + `lsr` with one `lsr` per block. The table grows from 256 bytes to 8 KB.
- **check:** pass
- **compare:** significantly faster, 1.02x [1.02 – 1.02]
- **Kept:** yes
- **Notes:** +3 to 5% from n=64 to 16,384; 65,536 and 262,144 are 1 to 2% slower in half_selected (the table competes for L1); the largest sizes are +1%.

## 6. Register-offset mask loads with a single up-counting index

- **Idea:** replace `ldp x6, x15, [x1], #16` plus `subs`/`b.hs` with two `ldr [base, x4]` loads and `adds x4, #16`/`b.ne`, hoping to drop the pointer writeback.
- **check:** pass
- **compare:** significantly slower, 0.975x [0.972 – 0.977]
- **Kept:** no
- **Notes:** the post-indexed `ldp` of two general registers is cheaper than two separate loads; sizes above L1 lost 4%.

## 7. Sparse path: 64-bit bitmask and unconditional `clz` extractions, tiers of 6 and 12

- **Idea:** same scheme as compress_f64 attempt 3 and nonzero_bool attempts 3 to 5. Per 64 elements, build a bitmask in NEON (`ushl` by 0..7, three `addp`, `fmov`, `rbit`) and count it. At most 6 selected: find 6 positions with `clz` / `lsr` / `bic` and copy `x[pos]` with scalar `ldr s` / `stp s`, no branches. 7 to 12: a second copy with 12 extractions. More than 12: the dense table code (attempt 5) for those 64 elements. After a dense stretch, back to sparse if at most 1 in 8 was selected, else a longer dense stretch (up to 512 elements). The sparse loop needs 65 elements left (an exhausted `clz` copies `x[64]` as rubbish); exactly 64 left use a copy that masks the position with `and #63`.
- **check:** pass
- **compare:** significantly faster, 1.30x [1.30 – 1.31] (first run 1.30x [1.30 – 1.33] with a NOISY MACHINE warning; rerun clean)
- **Kept:** yes
- **Notes:** few_selected 1.74x (2.2 to 2.4x from 256 to 16,384, 1.9x to 1M, 1.17x at 4M and 16M where scattered reads of x miss the cache). half_selected 0.98x: n=64 and n=256 are 0.91x (the failed sparse attempt on the first block), 1,024 to 16,384 0.97 to 1.0x.

## 8. Longer dense stretch between density checks (4,096 elements instead of 512)

- **Idea:** the dense loop stops every 32 iterations to decide whether to go back to sparse; raise that to 256 so the bookkeeping and the loop-exit misprediction are spread over more elements.
- **check:** pass
- **compare:** no significant difference, 0.996x [0.990 – 1.004]
- **Kept:** no
- **Notes:** a local microbenchmark showed 2 to 4% at 16,384 half_selected, the judge showed 0.99x there. The bookkeeping is already negligible at 512.

## 9. Choose the starting mode from the first 16 mask bytes (local timing only)

- **Idea:** count the first 16 mask bytes with one `ldp`/`add`/`mul`/`lsr`; start dense if more than 3 are true, to avoid building a 64-byte bitmask that half_selected then throws away.
- **check:** not run through the judge
- **compare:** not run; local microbenchmark only: n=64 half_selected 0.146 to 0.143 ns per element (the version without a sparse path is 0.131), n=64 few_selected 0.070 to 0.075.
- **Kept:** no
- **Notes:** the probe costs about as much as it saves; a 3% gain on one size against a 6% loss on another.

## 10. `ld4` + `sli` bitmask in the sparse path (local timing only)

- **Idea:** build the 64-bit bitmask with `ld4` (de-interleaving load), three `sli`, `usra`, `xtn` instead of two `ldp q`, four `ushl` and four `addp`.
- **check:** not run through the judge
- **compare:** not run; local microbenchmark at 5% density: 0.0423 to 0.0463 ns per element at n=1,024, 0.0441 to 0.0516 at n=256.
- **Kept:** no
- **Notes:** `ld4` costs more than the six instructions it saves.

## 11. Dense stretch loop: 32 elements per iteration

- **Idea:** two 16-element bodies per iteration of the dense stretch loop, sharing the counter and the branch.
- **check:** pass
- **compare:** significantly faster, 1.006x [1.004 – 1.007]; rerun 1.004x [1.004 – 1.006]
- **Kept:** yes
- **Notes:** half_selected 1.01x (2 to 3% from n=256 to 16,384, nothing above), few_selected 1.00x. Small but repeatable.

## 12. Dense: compact each 8-element block as one 32-byte unit with two-register `tbl`

- **Idea:** instead of compacting each 4-lane half on its own and storing the second half at `write pointer + 4 * (count in the first half)`, look both output vectors up in the 32-byte pair with `tbl {v0, v1}` and store them together with one `stp`. The table entry becomes a 32-byte pattern over all eight lanes (still 8 KB). Removes the per-block `ubfx` for the low count and the register-offset store.
- **check:** pass
- **compare:** significantly faster, 1.015x [1.012 – 1.018] (third run, clean; the first two had NOISY MACHINE warnings and read 1.009x [1.008 – 1.010] and about the same)
- **Kept:** yes
- **Notes:** half_selected 1.02x: 4 to 6% from n=64 to 16,384, 0 to 1% above. n=65,536 read 0.92x and 0.95x in the two noisy runs and 1.02x in the clean one. few_selected unchanged.

## 13. `ldp` for the two halves of a table entry (local timing only)

- **Idea:** one `add` to form the entry address and one `ldp q` instead of two register-offset `ldr q`.
- **check:** not run through the judge
- **compare:** not run; local microbenchmark, half density: 0.0915 to 0.0914 ns per element at n=1,024, 0.0889 to 0.1031 at n=16,384.
- **Kept:** no
- **Notes:** no gain.

## 14. Sizes of the two sparse tiers, measured on masks the predictor has not seen (local timing only)

- **Idea:** the benchmark repeats one input per size, so the branch predictor can memorise which 64-element blocks need the second tier. A local benchmark with 1,024 different masks of 4,096 elements (data in L1) measures the tiers without that help.
- **check:** not run through the judge
- **compare:** not run; ns per element at 5% density for first/second tier sizes: 4/8 0.072, 5/10 0.053, 6/12 0.047, 6/10 0.046 to 0.047, 7/14 0.052, 8/16 0.055. The version without a sparse path is 0.097.
- **Kept:** no (6/12 stays)
- **Notes:** on the memorised benchmark smaller first tiers look better (see nonzero_bool attempt 5); they were not adopted. Known weakness, not visible to the judge: at 10% density the count sits on the boundary between the tiers, the tier branch is mispredicted about half the time, and this kernel runs at 0.126 ns per element against 0.097 for the version without a sparse path (30% slower). At 2%, 5%, 20% and 50% it is 0.042, 0.047, 0.094 and 0.092 against 0.097. A fix would be to stay in the 12-extraction code while its use is frequent; it cannot show as a gain on the two benchmark distributions, so under the keep rule it was not attempted.

## Final result (`uv run judge run compress_f32 --unlocked`, after attempt 14)

- **check:** pass (30,482 cases)
- **vs naive C:** 3.76x [3.76 – 3.77] overall; half_selected 2.75x [2.74 – 2.77], few_selected 5.14x [5.13 – 5.16]
- **vs NumPy:** 30.5x [30.46 – 30.71] overall; half_selected 42.6x [42.42 – 43.08], few_selected 21.9x [21.82 – 21.96]
- **Smallest per-size speedups vs naive C:** n=16 1.98x (half) and 1.95x (few); 16M 2.43x (half) and 3.69x (few).
- **Stopped because:** no untried idea left that is expected to help; see attempt 14 for the known weakness at about 10% density.

# x86-64

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, locked, pinned to one performance core by the judge.

What the compiler does with `naive.c` (`clang -O3 -march=native`): the scalar loop, one load, one store and one add of the mask byte per element, about 0.19 ns per element at every size up to 1,048,576. NumPy takes 3.4 ns per element for `x[mask]` at 1,048,576 with half selected.

## x86-64 1. Baseline: `vpermps` by a 256-entry table, 32 mask bytes per `vpmovmskb`

- **Idea:** the candidates from the design. One `vpcmpeqb` against zero and one `vpmovmskb` turn 32 mask bytes into 32 bits. Each byte of that word indexes a table of 256 permutations (8 KB, eight 32-bit lanes per entry); `vpermps` with the eight elements as its memory operand moves the selected ones to the front; all eight lanes are stored at the write position and `popcnt` advances it. A mask byte is true when it is not zero. 8 to 31 leftover elements take one table step per 8, the last 1 to 7 the scalar loop of the reference.
- **check:** pass (after one failure: the last group of a block was not masked, so five bits of the group before it reached the table offset and `vmovdqa` faulted on an unaligned entry)
- **compare:** not run (first version). `run`: 2.37x over naive C overall (half_selected 2.17x, few_selected 2.60x), 21.5x over NumPy.
- **Kept:** yes
- **Notes:** branch-free, so both distributions cost the same inside the first-level cache: 0.05 ns per element.

## x86-64 2. Take each group's table offset with `rorx`

- **Idea:** the mask word is shifted left by 5 once per block. A group's offset is then `rorx` by 8, 16 or 24 and one `and`, in place of a `mov`, an `and` and a destructive `shr` per group. One instruction slot fewer per group.
- **check:** pass
- **compare:** significantly faster, 1.008x [1.000 – 1.017]
- **Kept:** yes
- **Notes:** small, and at the edge of what the judge resolves. No size got slower.

## x86-64 3. Count output elements, not bytes

- **Idea:** the one chain that runs from group to group is the write position. Kept as a byte address it advanced through `lea r9, [r9 + r8*4]`, and a `lea` with a scaled index takes two cycles on this processor. Kept as an element count it advances through `add`, one cycle, and the store scales it in its address.
- **check:** pass
- **compare:** significantly faster, 1.018x [1.010 – 1.025]
- **Kept:** yes
- **Notes:** 1.06x at 1,024 with few selected, nothing elsewhere. The chain was not what held the loop. A micro-benchmark outside the judge, 1,024 elements in the first-level cache, with one thing removed at a time: 41.0 ns as it is, 41.1 ns without the store, 37.3 ns without the permute, 40.0 ns without `popcnt`. That is 1.70 cycles per group of eight whatever is taken out, so the loop is held by its instruction count: 35 instruction slots per block of 32 on a core that renames six per cycle.

## x86-64 4. Align the loads of x to 32 bytes

- **Idea:** what gave `sum_u8` 1.5x beyond the first-level cache. One extra table step over the first eight elements, restricted to those in front of the next 32-byte boundary, then blocks whose loads never straddle two cache lines.
- **check:** pass
- **compare:** significantly slower, 0.960x [0.952 – 0.968]
- **Kept:** no
- **Notes:** 1.02x to 1.05x from 16,384 to 1,048,576, and 0.71x at 64 and 0.85x at 256, where the head step is a large share of the call. The loads of x are not what limits this kernel beyond the first-level cache; see the next attempt.

## x86-64 5. Prefetch x and the mask

- **Idea:** `prefetcht0` 1,024 bytes ahead of x (two lines) and 256 ahead of the mask, once per block.
- **check:** pass
- **compare:** significantly slower, 0.980x [0.973 – 0.988]
- **Kept:** no
- **Notes:** 1.02x to 1.06x from 16,384 up, 0.81x to 0.96x below. A loop of its own for large arrays would keep the 2% to 6%; not pursued, because the micro-benchmark shows where the time goes there and it is not the reads. At 65,536 elements (data in the second-level cache), cycles per group of eight: 3.35 with half selected and 2.3 with 5% selected as the kernel is, 1.73 for both with the store removed, which is the speed inside the first-level cache. Beyond the first-level cache the kernel is held by the cache lines of the output: each has to be fetched before it is written and written back afterwards.

## x86-64 6. Blocks with at most four selected elements copied one by one

- **Idea:** the sparse path that paid on `compress_f64` and `nonzero_bool` (their attempt 2): a block of 32 with at most four selected elements skips the table, and `tzcnt` and `blsr` take four positions out of the mask word, each element copied with a scalar load and store.
- **check:** pass
- **compare:** significantly faster, 1.005x [1.001 – 1.009]
- **Kept:** no
- **Notes:** the verdict passes the rule and the change is still not worth having: few_selected 1.04x (1.07x to 1.09x from 65,536 to 1,048,576, but 0.81x at 1,024), half_selected 0.97x at every size, because every block pays the count and the branch. Four table steps for 32 `float32` cost 28 instruction slots and the four single copies 20, so there is little to win here; on `float64` the table path costs twice as much per block and the same change gave 1.24x.

## x86-64: where it stands

Final `judge run`:

`half_selected`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 2.4 ns | 4.1 ns | 1.72x [1.72 – 1.72] | 269.7 ns | 114x [114 – 114] |
| 64 | 4.5 ns | 14.4 ns | 3.17x [3.17 – 3.17] | 331.7 ns | 73.1x [73.0 – 73.2] |
| 256 | 15.4 ns | 47.6 ns | 3.09x [3.09 – 3.11] | 582.0 ns | 37.8x [37.8 – 37.9] |
| 1,024 | 60.9 ns | 192.5 ns | 3.16x [3.15 – 3.17] | 1.7 us | 28.2x [28.1 – 28.3] |
| 4,096 | 175.5 ns | 776.0 ns | 4.42x [4.42 – 4.42] | 5.9 us | 33.9x [33.8 – 34.0] |
| 16,384 | 1.3 us | 3.1 us | 2.37x [2.37 – 2.37] | 45.4 us | 34.8x [34.7 – 34.8] |
| 65,536 | 5.5 us | 13.1 us | 2.39x [2.39 – 2.40] | 212.6 us | 38.7x [38.7 – 38.8] |
| 262,144 | 23.5 us | 52.3 us | 2.22x [2.22 – 2.22] | 893.3 us | 38.0x [37.9 – 38.0] |
| 1,048,576 | 109.2 us | 207.2 us | 1.90x [1.89 – 1.91] | 3.57 ms | 32.6x [32.5 – 32.8] |
| 4,194,304 | 698.4 us | 1.05 ms | 1.50x [1.48 – 1.52] | 14.45 ms | 20.7x [20.5 – 20.9] |
| 16,777,216 | 3.68 ms | 4.72 ms | 1.28x [1.28 – 1.30] | 59.33 ms | 16.1x [16.1 – 16.2] |
| overall |  |  | 2.33x [2.32 – 2.33] |  | 36.8x [36.8 – 36.9] |

`few_selected`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 2.4 ns | 4.0 ns | 1.71x [1.70 – 1.71] | 219.1 ns | 92.8x [92.8 – 92.9] |
| 64 | 4.8 ns | 13.7 ns | 2.86x [2.85 – 2.86] | 264.4 ns | 55.3x [55.3 – 55.4] |
| 256 | 16.7 ns | 47.1 ns | 2.82x [2.82 – 2.83] | 323.8 ns | 19.4x [19.4 – 19.4] |
| 1,024 | 44.9 ns | 192.6 ns | 4.29x [4.27 – 4.30] | 595.0 ns | 13.3x [13.2 – 13.3] |
| 4,096 | 302.4 ns | 777.0 ns | 2.57x [2.57 – 2.57] | 1.6 us | 5.41x [5.40 – 5.42] |
| 16,384 | 902.9 ns | 3.1 us | 3.46x [3.46 – 3.46] | 5.4 us | 5.97x [5.96 – 5.98] |
| 65,536 | 3.7 us | 12.4 us | 3.34x [3.34 – 3.34] | 45.1 us | 12.1x [12.1 – 12.1] |
| 262,144 | 14.9 us | 50.0 us | 3.36x [3.36 – 3.36] | 212.8 us | 14.3x [14.3 – 14.3] |
| 1,048,576 | 68.7 us | 200.6 us | 2.92x [2.91 – 2.92] | 863.3 us | 12.6x [12.5 – 12.6] |
| 4,194,304 | 379.8 us | 864.7 us | 2.28x [2.24 – 2.31] | 3.48 ms | 9.17x [9.05 – 9.27] |
| 16,777,216 | 2.61 ms | 4.13 ms | 1.58x [1.57 – 1.59] | 14.34 ms | 5.49x [5.46 – 5.50] |
| overall |  |  | 2.73x [2.72 – 2.73] |  | 14.2x [14.2 – 14.2] |

All distributions: 2.52x [2.52 – 2.52] over naive C, 22.8x [22.8 – 22.9] over NumPy

With the output buffer 16 bytes behind its input (`half_selected`; "apart" is the same kernel with the output clear of the input):

`half_selected`:

| n | kernel | apart | speedup [95%] | naive C | speedup [95%] |
|---|---|---|---|---|---|
| 4,096 | 210.6 ns | 202.1 ns | 0.96x [0.96 – 0.96] | 778.0 ns | 3.69x [3.69 – 3.69] |
| 16,384 | 1.4 us | 1.3 us | 0.96x [0.96 – 0.96] | 3.5 us | 2.58x [2.57 – 2.58] |
| 65,536 | 5.4 us | 5.5 us | 1.01x [1.01 – 1.02] | 12.9 us | 2.37x [2.37 – 2.37] |
| 262,144 | 22.6 us | 22.3 us | 0.98x [0.98 – 0.99] | 50.6 us | 2.24x [2.23 – 2.24] |
| 1,048,576 | 108.7 us | 106.7 us | 0.98x [0.98 – 0.98] | 208.9 us | 1.92x [1.92 – 1.93] |
| 4,194,304 | 750.7 us | 890.5 us | 1.19x [1.18 – 1.20] | 1.09 ms | 1.45x [1.44 – 1.45] |
| 16,777,216 | 3.61 ms | 3.64 ms | 1.01x [1.00 – 1.02] | 4.76 ms | 1.32x [1.31 – 1.33] |
| overall |  |  | 1.01x [1.01 – 1.01] |  | 2.11x [2.11 – 2.11] |

No stall: the write position falls behind the read position as elements are dropped, so the stores are only briefly at the addresses the loads are about to read.

What holds it at each size:

- **16 to 1,024:** the instruction count. 1.70 cycles per eight elements inside the first-level cache, 0.04 to 0.05 ns per element.
- **4,096 and up:** the cache-line traffic of the output (attempt 5). The judge's timing loop cycles through several inputs at small sizes, so 4,096 elements are already outside the first-level cache there.
- **4,194,304 and 16,777,216:** memory. 0.17 to 0.22 ns per element with half selected, 1.3x to 1.5x over the naive C.

The candidates the design lists for this family (`docs/x86-64-design.md`):

| Candidate | Outcome |
|---|---|
| 32 mask bytes to a 32-bit mask in one `vpmovmskb` | Kept (attempt 1) |
| `vpermd`/`vpermps` moves the selected elements to the front by index | Kept (attempt 1), with the elements as the instruction's memory operand |
| The index from a table | Kept: 256 entries of 32 bytes, so the load needs no widening. Entries of 8 bytes widened with `vpmovzxbd` were not tried: the widening and the permute both need port 5, which would hold the loop to two cycles per group |
| The index from `pext` with no table | Not tried. It needs a `pdep`, a multiply, a `pext`, a move into a vector register and a widening in place of one load, and the loop is held by its instruction count |
| `popcnt` advances the write pointer | Kept, as an element count (attempt 3) |
| Sparse masks: `tzcnt` and `blsr` per set bit | Rejected here (attempt 6), kept in `compress_f64` and `nonzero_bool` |
| A block all zeros or all ones recognised with one compare | Not tried. On the random masks the judge times, a block of 32 is all zeros 19% of the time at 5% selected and never at 50%, so the branch would be mispredicted about as often as it saves four steps. It would pay on masks with long runs, which the judge does not time |
| The switch by density can go | It has gone: one path for every density, and no weak spot near 10% |
| Masked loads and stores for leftover elements | Not tried: whether they satisfy the memory rule is open issue 3 of the design. The last 1 to 7 elements take the scalar loop |
| Non-temporal stores | Not tried. Every store is overwritten in part by the next one, so they would need a staging buffer and a branch per filled cache line |
| An output just behind the input | No cost in the judge's close table |
| Alignment of data | Rejected (attempt 4) |
| Alignment of code | The block loop starts on a 64-byte boundary from attempt 1. Not varied |
| Instructions slow on other x86-64 processors | None used: no `pext`, `pdep`, gather or masked store |

Left untried, with what each would be worth: the mask word stored below the stack pointer and read back a byte at a time, and the count taken from the table entry's cache line instead of `popcnt` (together about 7 of 35 instruction slots, inside the first-level cache only); a prefetching loop for large arrays (2% to 6%, attempt 5).

Found here and worth knowing on ARM64: nothing that carries over. The ARM64 kernel's density switch exists because its dense path costs more per element than this one.
