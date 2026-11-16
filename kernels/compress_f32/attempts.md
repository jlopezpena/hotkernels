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
