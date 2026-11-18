# compress_f64 attempts (arm64)

This log holds two architectures. ARM64 (Apple M1 Pro) comes first. x86-64 (Intel Core i5-14600KF) follows under [x86-64](#x86-64), with its own numbering.

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

## Final result (`uv run judge run compress_f64 --unlocked`, after attempt 7)

- **check:** pass (30,482 cases)
- **vs naive C:** 2.73x [2.72 – 2.74] overall; half_selected 1.84x [1.83 – 1.85], few_selected 4.03x [4.01 – 4.07]
- **vs NumPy:** 20.2x [20.15 – 20.36] overall; half_selected 27.2x [27.06 – 27.39], few_selected 15.1x [14.95 – 15.21]
- **Smallest per-size speedups vs naive C:** half_selected n=16,384 1.43x, 16M 1.46x, 4M 1.54x; few_selected n=16 1.70x, 16M 1.87x, 4M 1.89x.
- **Stopped because:** no untried idea left that is expected to help. half_selected is memory-bound from n=16,384 up; few_selected at 4M and 16M is limited by scattered reads of x.

# x86-64

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, locked, pinned to one performance core by the judge. Read with the `compress_f32` log, where the shared scheme was worked out.

What the compiler does with `naive.c` (`clang -O3 -march=native`): the scalar loop, about 0.19 ns per element up to 262,144. NumPy takes 3.4 ns per element at 1,048,576 with half selected.

## x86-64 1. Baseline: `vpermd` by a 16-entry table, 32 mask bytes per `vpmovmskb`

- **Idea:** the `compress_f32` kernel as it stood after its attempt 3, four elements per step. AVX2 has no permute of 64-bit lanes by a variable index, so the table (16 entries, 512 bytes) holds pairs of 32-bit lane numbers for `vpermd`. One mask word serves eight steps. The write position is an element count from the start.
- **check:** pass
- **compare:** not run (first version). `run`: 1.50x over naive C overall (half_selected 1.35x, few_selected 1.67x), 12.0x over NumPy.
- **Kept:** yes
- **Notes:** twice the steps of `compress_f32` per mask byte, each moving the same 32 bytes, so the output's cache-line traffic weighs twice as much. With half selected, 1,024 elements already take 0.15 ns each in the judge's timing, which cycles through several inputs and so leaves the first-level cache.

## x86-64 2. Blocks with at most four selected elements copied one by one

- **Idea:** a block of 32 whose mask word has at most four bits set skips the table. `tzcnt` and `blsr` take a fixed four positions out of the word, and each element is copied with one 8-byte load and store. Bit 31 is set before every `tzcnt`, so once the set bits are used up the position is 31, inside the block, and the copy lands beyond the write position's advance. Blocks with more go through the eight table steps as before. One compare of the block's count chooses.
- **check:** pass
- **compare:** significantly faster, 1.239x [1.233 – 1.245]
- **Kept:** yes
- **Notes:** few_selected 1.53x (1.60x at 1,024, 1.98x at 65,536, 1.37x at 1,048,576), half_selected no significant difference. The sparse block costs about 30 instruction slots against 63 for eight table steps, and makes four 8-byte stores where the table path makes eight of 32 bytes. **No weak spot between the two densities the judge times.** Micro-benchmark outside the judge, 65,536 elements, random masks, ns per element at 1% / 5% / 10% / 12% / 15% / 20% / 30% / 50% selected: 0.055 / 0.056 / 0.058 / 0.067 / 0.085 / 0.111 / 0.130 / 0.144 with this change, 0.100 / 0.099 / 0.107 / 0.108 / 0.111 / 0.116 / 0.129 / 0.141 without. Near the threshold the branch is mispredicted often, and the kernel is still not slower than the table path alone, because that path is held by its stores and not by the front end.

## x86-64: where it stands

Final `judge run`:

`half_selected`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 3.1 ns | 4.2 ns | 1.33x [1.33 – 1.34] | 269.4 ns | 86.4x [86.4 – 86.5] |
| 64 | 10.1 ns | 14.1 ns | 1.39x [1.39 – 1.39] | 332.2 ns | 32.8x [32.8 – 32.8] |
| 256 | 36.0 ns | 50.7 ns | 1.41x [1.41 – 1.41] | 599.1 ns | 16.7x [16.6 – 16.7] |
| 1,024 | 158.4 ns | 222.5 ns | 1.40x [1.40 – 1.41] | 1.7 us | 10.9x [10.9 – 10.9] |
| 4,096 | 540.0 ns | 790.2 ns | 1.46x [1.46 – 1.46] | 5.8 us | 10.8x [10.8 – 10.8] |
| 16,384 | 2.5 us | 4.1 us | 1.66x [1.66 – 1.67] | 44.8 us | 18.0x [17.9 – 18.2] |
| 65,536 | 9.8 us | 14.1 us | 1.44x [1.44 – 1.44] | 209.3 us | 21.4x [21.4 – 21.4] |
| 262,144 | 47.7 us | 56.9 us | 1.19x [1.19 – 1.19] | 877.6 us | 18.4x [18.4 – 18.4] |
| 1,048,576 | 209.9 us | 229.5 us | 1.09x [1.08 – 1.12] | 3.51 ms | 16.7x [16.5 – 17.1] |
| 4,194,304 | 1.55 ms | 1.91 ms | 1.23x [1.22 – 1.24] | 14.19 ms | 9.13x [9.09 – 9.18] |
| 16,777,216 | 6.87 ms | 8.14 ms | 1.18x [1.18 – 1.19] | 60.40 ms | 8.79x [8.78 – 8.81] |
| overall |  |  | 1.34x [1.34 – 1.34] |  | 17.7x [17.7 – 17.7] |

`few_selected`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 3.2 ns | 4.0 ns | 1.22x [1.22 – 1.22] | 218.3 ns | 67.3x [67.2 – 67.3] |
| 64 | 3.8 ns | 12.8 ns | 3.33x [3.32 – 3.33] | 262.7 ns | 68.4x [68.3 – 68.4] |
| 256 | 14.3 ns | 48.8 ns | 3.42x [3.42 – 3.42] | 319.8 ns | 22.4x [22.4 – 22.4] |
| 1,024 | 56.6 ns | 193.0 ns | 3.41x [3.41 – 3.41] | 594.4 ns | 10.5x [10.5 – 10.5] |
| 4,096 | 226.2 ns | 778.4 ns | 3.44x [3.44 – 3.44] | 1.7 us | 7.44x [7.42 – 7.47] |
| 16,384 | 903.7 ns | 3.1 us | 3.47x [3.47 – 3.47] | 5.4 us | 5.97x [5.97 – 5.97] |
| 65,536 | 3.7 us | 12.8 us | 3.43x [3.43 – 3.43] | 45.4 us | 12.2x [12.2 – 12.2] |
| 262,144 | 15.8 us | 50.4 us | 3.18x [3.18 – 3.18] | 212.0 us | 13.4x [13.4 – 13.4] |
| 1,048,576 | 91.9 us | 205.7 us | 2.24x [2.24 – 2.24] | 860.7 us | 9.36x [9.35 – 9.38] |
| 4,194,304 | 938.5 us | 1.68 ms | 1.79x [1.77 – 1.80] | 3.73 ms | 3.97x [3.96 – 4.00] |
| 16,777,216 | 4.32 ms | 6.08 ms | 1.41x [1.40 – 1.41] | 14.83 ms | 3.44x [3.43 – 3.45] |
| overall |  |  | 2.59x [2.58 – 2.59] |  | 12.3x [12.3 – 12.3] |

All distributions: 1.86x [1.86 – 1.86] over naive C, 14.8x [14.8 – 14.8] over NumPy

With the output buffer 16 bytes behind its input (`half_selected`; "apart" is the same kernel with the output clear of the input):

`half_selected`:

| n | kernel | apart | speedup [95%] | naive C | speedup [95%] |
|---|---|---|---|---|---|
| 4,096 | 553.0 ns | 540.7 ns | 0.98x [0.98 – 0.98] | 792.1 ns | 1.43x [1.43 – 1.43] |
| 16,384 | 2.4 us | 2.4 us | 0.99x [0.99 – 0.99] | 3.3 us | 1.35x [1.35 – 1.35] |
| 65,536 | 9.7 us | 9.6 us | 0.99x [0.99 – 0.99] | 13.8 us | 1.42x [1.42 – 1.42] |
| 262,144 | 47.8 us | 47.5 us | 0.99x [0.99 – 0.99] | 56.9 us | 1.19x [1.19 – 1.19] |
| 1,048,576 | 233.6 us | 315.0 us | 1.35x [1.32 – 1.38] | 249.0 us | 1.07x [1.06 – 1.08] |
| 4,194,304 | 1.55 ms | 1.69 ms | 1.09x [1.08 – 1.09] | 1.87 ms | 1.21x [1.20 – 1.21] |
| 16,777,216 | 6.89 ms | 6.76 ms | 0.98x [0.98 – 0.98] | 8.11 ms | 1.18x [1.17 – 1.18] |
| overall |  |  | 1.05x [1.04 – 1.05] |  | 1.26x [1.25 – 1.26] |

No stall, as in `compress_f32`.

What holds it: the instruction count up to 256 elements; from there, with half selected, the cache lines of the output and then memory, where it is 1.1x to 1.2x over the naive C. With few selected it stays 3.2x to 3.5x over the naive C up to 262,144.

The candidates the design lists for this family: as in the `compress_f32` log, with one difference. **Sparse masks, `tzcnt` and `blsr` per set bit: kept here** (attempt 2), as a fixed four per block of 32 chosen by the block's own count. The design asked whether the switch by density and its weak spot near 10% could go. There is still a switch, per block and with no state, and the micro-benchmark above finds no density at which it is slower than the table path alone. No `pext`, `pdep`, gather or masked store is used.

Left untried: a second sparse tier for blocks with five to eight selected elements, which would move the gain up to about 25% selected. The judge times 5% and 50%, so it could not be kept on a `compare` verdict.
