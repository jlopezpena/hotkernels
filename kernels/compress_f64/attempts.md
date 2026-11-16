# compress_f64 attempts (arm64)

All judge commands in this run were passed `--unlocked`, because the spec is not on `main` yet. Other agents were timing kernels on the same machine, so intervals are wider than usual.

## 1. Scalar baseline

- **Idea:** the naive loop in assembly: store every element at `out[k]`, advance `k` by the mask byte.
- **check:** pass
- **compare:** not run (first version, nothing to compare against)
- **Kept:** yes
- **Notes:** starting point only; expected to match naive C.

## 2. Eight elements per step, compaction by load address

- **Idea:** a pair needs one 16-byte store: `{x[p], x[p+1]}` if `mask[p]`, else `{x[p+1], x[p+2]}`. So load each pair from `x + 8*(p + 1 - mask[p])` and store it whole; no shuffle and no table. Mask bytes loaded as a u64 `m`; `C - (m << 3)` gives the four load offsets in 16-bit fields, `m * 0x0808…08` gives the store offsets (running counts times 8) in bytes. Write pointer advances once per 8 elements. Main loop needs 9 elements left (the pair load reads one past the pair); exactly 8 left go through a block whose last pair uses scalar stores; scalar tail below that.
- **check:** pass
- **compare:** significantly faster, 1.94x [1.93 – 1.96]
- **Kept:** yes
- **Notes:** 2.4x in L1 (about 0.139 ns per element), 2.0x for few_selected beyond L1, 1.2 to 1.8x for half_selected beyond L1. n=16,384 is an outlier (1.5 to 1.6x): x and out are each 128 KB there, the size of L1.

## 3. Sparse path: 64-bit bitmask and unconditional `clz` extractions, tiers of 6 and 12

- **Idea:** same scheme as nonzero_bool attempts 3 to 5. Per 64 elements, build a bitmask in NEON (`ushl` by 0..7, three `addp`, `fmov`, `rbit`) and count it. If at most 6 are selected, find 6 positions with `clz` / `lsr` / `bic` and copy `x[pos]` with scalar `ldr d` / `stp d`, no branches; 7 to 12 selected use a second copy with 12 extractions; more than 12 falls back to the dense code (attempt 2) for those 64 elements. After a dense stretch, go back to sparse if at most 1 in 8 was selected, else run a longer dense stretch (up to 512 elements). An exhausted `clz` gives 64 and copies `x[64]` as rubbish, so the sparse loop needs 65 elements left; exactly 64 left use a copy that masks the position with `and #63`.
- **check:** pass
- **compare:** significantly faster, 1.43x [1.43 – 1.44]
- **Kept:** yes
- **Notes:** few_selected 2.07x: 2.2x to 4.4x up to 16,384, 2.0 to 2.1x up to 1M, only 1.05x at 4M and 16M. half_selected 0.99x (n=64 0.94x and n=256 0.97x from the failed sparse attempt on the first block). At 4M and 16M the sparse path reads x at scattered positions, the hardware prefetcher does not follow, and the time is set by how many cache misses fit in the out-of-order window (about 31 cycles per 64 elements).

## 4. Software prefetch of x in the dense loop (local timing only)

- **Idea:** n=16,384 half_selected runs at 0.20 ns per element against 0.134 in L1; add `prfm pldl1keep, [x0, #256 / #512 / #1024]` per 8 elements.
- **check:** not run through the judge
- **compare:** not run; local microbenchmark at n=16,384 half density: 0.192 without, 0.220 / 0.201 / 0.210 with the three distances.
- **Kept:** no
- **Notes:** timing-only variants showed the slowdown follows the x read stream (fixing the read pointer gives 0.139, fixing the write pointer changes nothing). Above L1 the dense loop moves about one cache line per 3 cycles, which looks like the L2 transfer rate, not missing prefetch.

## 5. Sparse path that also prefetches 16 or 32 steps ahead (local timing only)

- **Idea:** at 4M and 16M few_selected the sparse path is only 1.05x faster than dense. Build the bitmask of the block 16 or 32 steps ahead as well and `prfm` the 6 elements it selects, so the scattered reads of x start earlier than the out-of-order window allows.
- **check:** not run through the judge
- **compare:** not run; local microbenchmark at n=4M, 5% density: 0.153 ns per element without, 0.161 with either distance; n=16,384: 0.047 without, 0.070 with.
- **Kept:** no
- **Notes:** no gain where it was meant to help, so that size is not limited by how early the misses start. It moves about 4.4 scattered cache lines per 64 elements (mask, selected x lines and the rubbish read of `x[64]`) at roughly half the streaming rate.

## 6. Dense stretch loop: 16 elements per iteration

- **Idea:** two 8-element blocks per iteration in the dense stretch loop (`ldp` for the two mask words, one loop branch per 16). The loop for the last 64 or fewer elements keeps 8 per iteration.
- **check:** pass
- **compare:** significantly faster, 1.017x [1.016 – 1.018]
- **Kept:** yes
- **Notes:** half_selected 1.03x: 8 to 10% from n=256 to 4,096, 2 to 4% from 65,536 to 1M, nothing at 16,384, 4M and 16M (memory-bound). few_selected unchanged.

## 7. Densities between the two benchmark distributions (local measurement only)

- **Idea:** check the sparse/dense choice on masks the predictor has not seen (1,024 different masks of 4,096 elements, data in L1), at densities the judge does not time.
- **check:** not run through the judge
- **compare:** not run; ns per element at 2% / 5% / 10% / 20% / 50%: 0.042 / 0.048 / 0.129 / 0.123 / 0.124, against 0.135 at every density for the dense-only version (attempt 2).
- **Kept:** nothing changed
- **Notes:** 10% density sits on the boundary between the two sparse tiers and the tier branch is mispredicted about half the time; for this kernel that still comes out slightly ahead of dense-only. See compress_f32 attempt 14 for the tier-size measurements.
