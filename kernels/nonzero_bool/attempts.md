# nonzero_bool attempts (arm64)

This log holds two architectures. ARM64 (Apple M1 Pro) comes first. x86-64 (Intel Core i5-14600KF) follows under [x86-64](#x86-64), with its own numbering.

All judge commands in this run were passed `--unlocked`, because the spec is not on `main` yet. Other agents were timing kernels on the same machine, so intervals are wider than usual.

## 1. Scalar baseline

- **Idea:** the naive loop in assembly: store every index at `out[k]`, advance `k` by the mask byte.
- **check:** pass
- **compare:** not run (first version, nothing to compare against)
- **Kept:** yes
- **Notes:** starting point only; expected to match naive C.

## 2. Eight mask bytes per step with a 256-entry position table

- **Idea:** load 8 mask bytes as a u64; one multiply packs them into an 8-bit mask, another gives 8x the true count. The mask indexes a 2 KB table whose 8-byte entry lists the positions of the true bytes in order. Widen those to 64 bits (`ushll` x3), add the block's base index (`uaddw`/`uaddw2` x4) and store all 64 bytes at the write pointer (`stp q` x2). Write pointer advances once per 8 elements. Scalar tail.
- **check:** pass
- **compare:** significantly faster, 2.55x [2.55 – 2.56]
- **Kept:** yes
- **Notes:** half_selected 2.37x, few_selected 2.75x. About 0.119 ns per element in L1 (roughly 3 cycles per 8). half_selected falls to 1.5 to 2.2x above 1M elements, where the output stream is memory-bound.

## 3. Sparse path: 64 mask bytes to a bitmask, 8 unconditional extractions

- **Idea:** per 64 mask bytes, build a 64-bit bitmask in NEON (`ushl` by 0..7, three rounds of `addp`, `fmov`) and count it (`cnt`, `addv`). If at most 8 bytes are true, extract 8 positions with `rbit`/`clz`/`x & (x-1)`, no branches, and store all 8 (extras are rubbish in the scratch buffer). Otherwise run the dense table code (attempt 2) for those 64 bytes. After a dense stretch, go back to sparse if at most 1 in 8 bytes was true, else run a longer dense stretch (up to 512 bytes) before looking again. About 70 instructions per 64 bytes in the sparse path against about 170 dense.
- **check:** pass
- **compare:** significantly faster, 1.24x [1.24 – 1.24]
- **Kept:** yes
- **Notes:** few_selected 1.65x (1.9x from n=1,024 up). half_selected 0.93x: unchanged from n=1,024 to 1M, but n=16 is 0.67x (3.0 to 4.5 ns), n=64 0.85x, n=256 0.94x, from the extra setup and the failed sparse attempt on the first block. 16M half_selected read 0.88x in this run; other agents were timing at the same time and that size is memory-bound, so it needs a recheck. Next: cut the small-n overhead.

## 4. Separate loop for n < 64 and for the leftover after the main loops

- **Idea:** inputs under 64 bytes go straight to the plain 8-byte dense loop from attempt 2 (no bitmask constants, no stretch bookkeeping); the same loop finishes what is left after the sparse/dense main loops.
- **check:** pass
- **compare:** significantly faster, 1.03x [1.03 – 1.03]
- **Kept:** yes
- **Notes:** n=16 goes from 4.5 to 3.3 ns (1.40x); everything else within 1 to 2%. 16M half_selected read 0.96x, again while other agents were running.

## 5. Cheaper sparse extraction, two tiers of 6 and 12

- **Idea:** three changes to the sparse path, tried together after timing-only experiments showed it is limited by instruction count (about 6 per cycle), not by the dependency chain: (a) reverse the bitmask once (`rbit`) and extract with `clz` / `lsr` / `bic` instead of `rbit` + `clz` + `sub` + `and` per position; (b) extract only 6 positions when the count is at most 6, a second copy extracts 12 when it is 7 to 12, and the dense code takes more than 12; (c) loop on the mask pointer against an end pointer instead of keeping a separate remaining count.
- **check:** pass
- **compare:** significantly faster, 1.17x [1.16 – 1.17]
- **Kept:** yes
- **Notes:** few_selected 1.39x, half_selected 0.98x (4M and 16M read 0.90x and 0.92x; the dense loop itself is unchanged, see the next entry for a noise check). Local timing of the tier sizes at 5% density: 4/8 was fastest up to 262,144 elements but slowest at 4M (0.072 ns per element against 0.046 for 6/12). The benchmark repeats one input, so the predictor memorises which blocks take the second tier; 4M is too long to memorise. 6/12 was chosen as the best on unmemorised data, not the best on the benchmark.

## 6. Dense: 32-bit position table and a count table (local timing only)

- **Idea:** replace the 8-byte table entry plus three `ushll` with a 32-byte entry of 32-bit positions (two `ldr q`, 8 KB table, index from `lsr #55` of a reversed-bit multiply), and replace the count multiply with a 16-bit count table load. Three instructions fewer per 8 bytes.
- **check:** not run through the judge
- **compare:** not run; local microbenchmark, half density, ns per element: 0.115 before, 0.140 with both tables, 0.123 with the position table only (n=16,384); 0.120 / 0.135 / 0.119 at n=1,024.
- **Kept:** no
- **Notes:** each extra register-offset table load costs more than the two or three NEON instructions it replaces. Timing-only deletions showed about 0.14 cycles per NEON instruction and about 0.22 per store micro-op in this loop.

## 7. Dense stretch loop: 16 mask bytes per iteration

- **Idea:** two 8-byte blocks per iteration in the dense stretch loop (`ldp` for the mask words, two base-index vectors, one loop branch per 16 bytes). The small-n loop keeps 8 per iteration.
- **check:** pass
- **compare:** significantly faster, 1.016x [1.015 – 1.017]
- **Kept:** yes
- **Notes:** half_selected 1.03x (3 to 5% from n=64 to 4M; 16M read 0.97x), few_selected 1.00x.

## 8. `dc zva` on the output line ahead of the write pointer (local timing only)

- **Idea:** half_selected above 1M elements is limited by memory traffic, and part of it is reading output cache lines only to overwrite them. Zero the 64-byte line 64 bytes past the write pointer before each block's stores, so the line does not have to be fetched (the scratch output allows writing there).
- **check:** not run through the judge
- **compare:** not run; local microbenchmark, half density: 0.129 to 0.254 ns per element at n=4M, 0.110 to 0.244 at n=16,384.
- **Kept:** no
- **Notes:** twice as slow at every size, including the memory-bound ones. The timing variant did not bound the zeroing at the end of the buffer and was never a candidate as written.

## 9. Densities between the two benchmark distributions (local measurement only)

- **Idea:** check the sparse/dense choice on masks the predictor has not seen (1,024 different masks of 4,096 bytes), at densities the judge does not time.
- **check:** not run through the judge
- **compare:** not run; ns per element at 2% / 5% / 10% / 20% / 50%: 0.038 / 0.045 / 0.127 / 0.110 / 0.112, against 0.116 at every density for the dense-only version (attempt 2).
- **Kept:** nothing changed
- **Notes:** known weakness, not visible to the judge: at 10% density the count sits on the boundary between the two sparse tiers, the tier branch is mispredicted about half the time, and this kernel is 9% slower than dense-only. See compress_f32 attempt 14.

## Final result (`uv run judge run nonzero_bool --unlocked`, after attempt 9)

- **check:** pass (7,268 cases)
- **vs naive C:** 4.27x [4.26 – 4.28] overall; half_selected 2.67x [2.66 – 2.68], few_selected 6.83x [6.81 – 6.84]
- **vs NumPy:** 15.7x [15.60 – 15.68] overall; half_selected 10.7x [10.67 – 10.75], few_selected 22.9x [22.73 – 22.93]
- **Smallest per-size speedups vs naive C:** n=16 2.20x in both distributions; half_selected 16M 2.45x.
- **Stopped because:** no untried idea left that is expected to help; see attempt 9 for the known weakness at about 10% density.

# x86-64

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, locked, pinned to one performance core by the judge.

What the compiler does with `naive.c` (`clang -O3 -march=native`): the scalar loop, one store and one add of the mask byte per element, about 0.19 ns per element. NumPy takes 0.42 ns per element at 1,048,576 with half selected and 0.69 with 5%.

## x86-64 1. Baseline: positions from a 256-entry table, widened with `vpmovzxbq`

- **Idea:** one `vpcmpeqb` and one `vpmovmskb` turn 32 mask bytes into 32 bits. Each byte of the word indexes a table of 256 entries of eight bytes (2 KB): the positions of its set bits in ascending order. Two `vpmovzxbq` widen an entry to eight 64-bit lanes, the index of the group's first element is added, both registers are stored at the write position and `popcnt` advances it. The write position is an element count, so the chain from group to group is one `add` (see `compress_f32`, x86-64 attempt 3). A mask byte is true when it is not zero.
- **check:** pass
- **compare:** not run (first version). `run`: 2.00x over naive C overall (half_selected 1.85x, few_selected 2.16x), 9.17x over NumPy.
- **Kept:** yes
- **Notes:** about 2.8 cycles per eight mask bytes inside the first-level cache, more than the eleven instructions need. Each group makes two 32-byte stores to the same advancing place, 64 bytes for at most eight indices, and the stores are what hold it. A table of 64-byte entries (16 KB) that needs no widening was not tried: the widening is not the limit.

## x86-64 2. Blocks with at most four true elements take `tzcnt` and `blsr`

- **Idea:** a block of 32 whose mask word has at most four bits set skips the table. `tzcnt` and `blsr` take a fixed four positions out of the word, each stored with one 8-byte store. Past the last set bit `tzcnt` returns 32 and the store is an index the write position does not advance over. Blocks with more go through the four table groups. One compare of the block's count chooses.
- **check:** pass
- **compare:** significantly faster, 1.401x [1.382 – 1.421]
- **Kept:** yes
- **Notes:** few_selected 1.97x (1.5x at 64, 2.0x at 1,024, 2.4x to 2.7x from 16,384 to 4,194,304), half_selected no significant difference. Four 8-byte stores per block where the table path makes eight of 32 bytes. **No weak spot between the two densities the judge times.** Micro-benchmark outside the judge, 65,536 elements, random masks, ns per element at 1% / 5% / 10% / 12% / 15% / 20% / 30% / 50% true: 0.037 / 0.035 / 0.052 / 0.049 / 0.060 / 0.082 / 0.104 / 0.113 with this change, 0.076 / 0.081 / 0.084 / 0.086 / 0.087 / 0.095 / 0.106 / 0.114 without. The ARM64 kernel is 9% slower than its dense path at 10%; here the mispredicted branch near the threshold costs less than the stores it saves.

## x86-64: where it stands

Final `judge run`:

`half_selected`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 3.3 ns | 4.0 ns | 1.22x [1.22 – 1.22] | 322.5 ns | 98.0x [98.0 – 98.1] |
| 64 | 6.0 ns | 11.3 ns | 1.89x [1.88 – 1.89] | 338.1 ns | 56.3x [56.2 – 56.4] |
| 256 | 19.2 ns | 46.8 ns | 2.44x [2.44 – 2.45] | 456.0 ns | 23.8x [23.7 – 23.9] |
| 1,024 | 83.5 ns | 273.3 ns | 3.27x [3.27 – 3.29] | 856.6 ns | 10.3x [10.2 – 10.3] |
| 4,096 | 301.2 ns | 775.5 ns | 2.57x [2.57 – 2.58] | 2.1 us | 6.90x [6.90 – 6.91] |
| 16,384 | 1.8 us | 3.1 us | 1.69x [1.69 – 1.70] | 7.9 us | 4.35x [4.34 – 4.35] |
| 65,536 | 7.4 us | 13.1 us | 1.77x [1.77 – 1.77] | 27.3 us | 3.67x [3.67 – 3.67] |
| 262,144 | 30.4 us | 56.3 us | 1.85x [1.85 – 1.85] | 109.4 us | 3.60x [3.59 – 3.60] |
| 1,048,576 | 138.9 us | 218.8 us | 1.58x [1.57 – 1.58] | 438.4 us | 3.16x [3.15 – 3.16] |
| 4,194,304 | 701.7 us | 901.6 us | 1.28x [1.22 – 1.31] | 1.79 ms | 2.55x [2.41 – 2.60] |
| 16,777,216 | 3.99 ms | 4.58 ms | 1.15x [1.14 – 1.15] | 11.38 ms | 2.85x [2.83 – 2.86] |
| overall |  |  | 1.79x [1.78 – 1.80] |  | 8.26x [8.22 – 8.28] |

`few_selected`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 3.1 ns | 3.2 ns | 1.01x [1.00 – 1.01] | 319.0 ns | 102x [102 – 102] |
| 64 | 3.1 ns | 10.9 ns | 3.50x [3.50 – 3.50] | 328.3 ns | 105x [105 – 105] |
| 256 | 9.7 ns | 46.7 ns | 4.79x [4.79 – 4.79] | 364.0 ns | 37.3x [37.3 – 37.4] |
| 1,024 | 32.9 ns | 194.2 ns | 5.91x [5.90 – 5.91] | 543.1 ns | 16.5x [16.5 – 16.5] |
| 4,096 | 137.9 ns | 775.6 ns | 5.62x [5.62 – 5.64] | 1.1 us | 7.96x [7.93 – 8.03] |
| 16,384 | 533.8 ns | 3.1 us | 5.81x [5.81 – 5.81] | 3.0 us | 5.70x [5.70 – 5.71] |
| 65,536 | 2.1 us | 12.4 us | 5.83x [5.83 – 5.83] | 23.4 us | 11.0x [10.9 – 11.0] |
| 262,144 | 8.5 us | 50.8 us | 5.96x [5.96 – 5.97] | 170.0 us | 20.0x [19.9 – 20.0] |
| 1,048,576 | 37.0 us | 202.6 us | 5.47x [5.46 – 5.48] | 724.8 us | 19.6x [19.6 – 19.6] |
| 4,194,304 | 175.2 us | 815.0 us | 4.65x [4.56 – 4.71] | 2.90 ms | 16.6x [16.2 – 16.8] |
| 16,777,216 | 776.9 us | 3.28 ms | 4.22x [4.15 – 4.28] | 11.85 ms | 15.2x [15.0 – 15.4] |
| overall |  |  | 4.41x [4.40 – 4.41] |  | 20.9x [20.8 – 20.9] |

All distributions: 2.81x [2.80 – 2.81] over naive C, 13.1x [13.1 – 13.1] over NumPy

With the output buffer 16 bytes behind its input (`half_selected`; "apart" is the same kernel with the output clear of the input):

`half_selected`:

| n | kernel | apart | speedup [95%] | naive C | speedup [95%] |
|---|---|---|---|---|---|
| 4,096 | 324.7 ns | 261.1 ns | 0.80x [0.80 – 0.81] | 775.2 ns | 2.39x [2.38 – 2.39] |
| 16,384 | 1.9 us | 1.9 us | 1.00x [1.00 – 1.00] | 3.5 us | 1.86x [1.86 – 1.86] |
| 65,536 | 7.4 us | 7.5 us | 1.01x [1.01 – 1.01] | 13.3 us | 1.81x [1.81 – 1.81] |
| 262,144 | 29.8 us | 31.2 us | 1.05x [1.05 – 1.05] | 56.2 us | 1.89x [1.88 – 1.89] |
| 1,048,576 | 138.3 us | 138.4 us | 1.00x [1.00 – 1.00] n.s. | 229.0 us | 1.66x [1.65 – 1.66] |
| 4,194,304 | 780.8 us | 931.1 us | 1.19x [1.18 – 1.22] | 961.0 us | 1.23x [1.23 – 1.23] |
| 16,777,216 | 3.80 ms | 3.92 ms | 1.03x [1.03 – 1.04] | 4.44 ms | 1.17x [1.16 – 1.17] |
| overall |  |  | 1.01x [1.01 – 1.01] |  | 1.67x [1.67 – 1.67] |

The close placement costs nothing from 16,384 up. At 4,096 it costs a fifth: the output is eight times the size of the mask, so only at the smallest sizes are the stores still near the addresses the mask loads are about to read.

What holds it: with half true, the stores of the table path (two of 32 bytes per eight mask bytes), then from 16,384 the cache lines of an output that is 4 bytes per mask byte, then memory. With few true, the instruction count of the sparse block, 0.035 ns per mask byte up to 1,048,576.

The candidates the design lists for this family (`docs/x86-64-design.md`):

| Candidate | Outcome |
|---|---|
| 32 mask bytes to a 32-bit mask in one `vpmovmskb` | Kept (attempt 1) |
| The index from a table | Kept: positions as bytes, widened with `vpmovzxbq`. `vpermd` does not apply, since the output is indices and not elements of an input |
| The index from `pext` with no table | Not tried: more instructions than one table load, and the table path is held by its stores |
| `popcnt` advances the write pointer | Kept |
| Sparse masks: `tzcnt` and `blsr` per set bit | Kept (attempt 2), as a fixed four per block |
| A block all zeros or all ones recognised with one compare | Not tried as a path of its own. An all-zero block takes the sparse path, which costs about five cycles for it. A branch for it alone would be taken at random on the masks the judge times |
| The switch by density can go | A switch remains, per block and with no state. The micro-benchmark finds no density at which it is slower than the table path alone, so the weak spot near 10% is gone |
| Masked loads and stores for leftover elements | Not tried: open issue 3 of the design. The last 1 to 7 elements take the scalar loop |
| Non-temporal stores | Not tried: every store is overwritten in part by the next |
| An output just behind the input | 0.80x at 4,096, nothing from 16,384 up (table above) |
| Alignment of data | Not tried here; rejected on `compress_f32` (its attempt 4), where the input is four times larger per element |
| Alignment of code | The block loop starts on a 64-byte boundary. Not varied |
| Instructions slow on other x86-64 processors | None used: no `pext`, `pdep`, gather or masked store |

Left untried: a second sparse tier for blocks with five to eight true elements (the judge times 5% and 50%, so `compare` could not see it); narrower stores in the table path when a group has at most four true elements, which is a branch per group.
