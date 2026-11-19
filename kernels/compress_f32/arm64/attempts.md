# compress_f32 attempts (arm64)

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
