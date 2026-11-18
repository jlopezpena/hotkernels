# cumsum_i64 attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, locked, pinned to one performance core by the judge.

What the compiler does with `naive.c` (`clang -O3 -march=native`): the scalar chain, one add and one store per element, one cycle each: 0.19 ns per element at every size up to 262,144. NumPy takes 1.4 ns per element at 1,048,576.

**What there is to win, measured before anything was kept.** A micro-benchmark outside the judge, 16,384 elements (128 KB in, 128 KB out, in the second-level cache): a plain copy loop of 32-byte loads and stores takes 0.115 ns per element when both buffers are aligned and 0.124 when they are not. The naive C takes 0.19. So beyond the first-level cache no running sum can be more than about 1.65x faster than the naive C on this machine; inside it, about 2x was reached (0.094 ns per element). The design expected this family to reach about half its ARM64 figures; the reason is this ceiling, and a scalar chain that costs one cycle here against three on the M1.

## 1. Baseline: window sums from overlapping loads, one add per vector on the chain

- **Idea:** the ARM64 kernel's regrouping in 32-byte vectors. `D` = the input vector plus the same load one element back. `W` = `D` plus `D` two elements back, which is the upper half of the previous `D` and the lower half of this one: one `vperm2i128`. The output vector is the previous one plus `W`, so the chain is one `vpaddq` per four elements. The design's worry was the extra permute for a prefix sum across the register (`vpslldq` works inside each 128-bit half); the overlapping load avoids all but one. The `D` of later vectors is loaded before the current output is stored, which keeps the kernel right when `out` is `x`. Before the array every value counts as zero.
- **check:** pass
- **compare:** not run (first version). `run`: 1.09x over naive C overall, 12.9x over NumPy.
- **Kept:** yes
- **Notes:** 0.165 ns per element from 1,024 to 65,536 in the judge. In the micro-benchmark the same code takes 0.095 with every buffer aligned inside the first-level cache, and at 16,384 elements 0.134 aligned, 0.161 with x 16 bytes off and 0.172 with both off: a 32-byte load that straddles two cache lines takes a cycle more and a store two, and the load one element back straddles at every second vector whatever the alignment. Knock-out variants (no store, no permute, no chain) all ran at 1.8 to 2 cycles per vector: four vector instructions per vector on three ports, and two 32-byte loads per cycle at most.

## 2. Align the stores to 32 bytes

- **Idea:** 0 to 3 elements one at a time until `out` is a multiple of 32, then the vectors, with the sum so far as the output before the first.
- **check:** pass
- **compare:** no significant difference, 0.999x [0.991 – 1.007]
- **Kept:** no
- **Notes:** the judge places the output 16 bytes after the input counted within a cache line, so aligning one buffer leaves the other 16 bytes off: with the stores aligned, two of every four loads straddle. 1.06x to 1.07x at 16,384 and 65,536, 0.84x at 16.

## 3. 16-byte vectors at every size

- **Idea:** two elements per vector: `D` = input plus the load one element back, output = previous output plus `D`. No permute, a chain of one add per two elements, and 16-byte accesses at 8-byte offsets rarely straddle a line. The loads run two vectors ahead of the stores, which also keeps a store 16 bytes behind the input (within a page) from holding up the loads after it.
- **check:** pass
- **compare:** significantly slower, 0.991x [0.988 – 0.995]
- **Kept:** no
- **Notes:** the sizes disagree: 1.05x at 1,024, 1.04x at 4,096, 1.13x at 16,384 and 65,536; 0.87x to 0.92x from 16 to 256, where its twice as many steps count; 0.96x to 0.97x at 4,194,304 and 16,777,216. In the micro-benchmark it takes 0.094 ns per element inside the first-level cache and 0.146 at 16,384, the same at every alignment.

## 4. 16-byte vectors from 512 to 131,071 elements

- **Idea:** attempts 1 and 3 by size. The wide version for short arrays and for arrays that do not fit the second-level cache together with their output; the narrow one in between.
- **check:** pass
- **compare:** significantly faster, 1.023x [1.018 – 1.028]
- **Kept:** yes
- **Notes:** 1.05x at 1,024, 1.03x at 4,096, 1.13x at 16,384, 1.15x at 65,536, nothing elsewhere except 16 elements, 0.91x: two more compares on the way in. With the wide path reached by fall-through; with a taken branch it was 0.88x there and 1.021x overall.

## 5. Stores past the caches from 24 MB of output

- **Idea:** the design's candidate. `vmovntdq` does not read the line it is about to overwrite, which removes a third of the memory traffic. It needs a 32-byte-aligned address, so the head of attempt 2 comes back for this path only, and the vectors that are left at the end use ordinary stores after an `sfence`. The threshold is the third-level cache: at a size no cache holds, it costs a caller nothing that the output is in no cache afterwards.
- **check:** pass (after one failure: the narrow path fell through into the new code). The judge's correctness cases stop far below 3,145,728 elements, so this path was tested outside it: every path, 43 sizes up to 3,200,003, eight alignments of x and five of out including one that is not a multiple of 8, against the scalar sum, and with `out` = `x`. No failure.
- **compare:** significantly faster, 1.044x [1.040 – 1.049]
- **Kept:** yes
- **Notes:** 1.26x at 4,194,304 and 1.25x at 16,777,216, nothing elsewhere. In the micro-benchmark the same stores are 1.3x faster already at 262,144 and 1,048,576 elements (0.175 and 0.185 ns per element against 0.227 and 0.243) and slower at 65,536. They are not used there: an output of 2 to 24 MB written with ordinary stores is left in the third-level cache for the caller, and one written this way is not.

## Where it stands

Final `judge run`:

`random`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 2.7 ns | 3.0 ns | 1.12x [1.11 – 1.13] | 720.6 ns | 271x [271 – 272] |
| 64 | 11.7 ns | 12.7 ns | 1.08x [1.08 – 1.09] | 764.3 ns | 65.1x [65.0 – 65.1] |
| 256 | 40.9 ns | 45.5 ns | 1.11x [1.11 – 1.11] | 1.1 us | 26.9x [26.8 – 26.9] |
| 1,024 | 164.2 ns | 191.4 ns | 1.17x [1.17 – 1.17] | 2.2 us | 13.4x [13.3 – 13.4] |
| 4,096 | 625.4 ns | 778.9 ns | 1.25x [1.25 – 1.25] | 6.6 us | 10.5x [10.5 – 10.5] |
| 16,384 | 2.5 us | 3.1 us | 1.23x [1.23 – 1.23] | 24.1 us | 9.58x [9.58 – 9.58] |
| 65,536 | 10.0 us | 12.4 us | 1.24x [1.24 – 1.24] | 94.3 us | 9.44x [9.44 – 9.46] |
| 262,144 | 60.8 us | 63.9 us | 1.05x [1.05 – 1.05] | 379.5 us | 6.24x [6.22 – 6.25] |
| 1,048,576 | 289.2 us | 268.1 us | 0.93x [0.92 – 0.94] | 1.50 ms | 5.19x [5.16 – 5.26] |
| 4,194,304 | 1.83 ms | 2.45 ms | 1.34x [1.34 – 1.36] | 7.83 ms | 4.28x [4.24 – 4.29] |
| 16,777,216 | 7.37 ms | 10.09 ms | 1.37x [1.37 – 1.38] | 30.80 ms | 4.18x [4.17 – 4.20] |
| overall |  |  | 1.17x [1.16 – 1.17] |  | 13.8x [13.8 – 13.8] |

With the output buffer 16 bytes behind its input ("apart" is the same kernel with the output clear of the input):

`random`:

| n | kernel | apart | speedup [95%] | naive C | speedup [95%] |
|---|---|---|---|---|---|
| 4,096 | 629.0 ns | 653.3 ns | 1.04x [1.04 – 1.04] | 779.3 ns | 1.24x [1.24 – 1.24] |
| 16,384 | 2.5 us | 2.5 us | 1.00x [1.00 – 1.01] | 3.1 us | 1.24x [1.24 – 1.24] |
| 65,536 | 10.0 us | 10.3 us | 1.03x [1.03 – 1.03] | 12.4 us | 1.23x [1.23 – 1.23] |
| 262,144 | 67.0 us | 65.0 us | 0.97x [0.97 – 0.97] | 450.0 us | 6.72x [6.71 – 6.73] |
| 1,048,576 | 393.3 us | 517.4 us | 1.32x [1.30 – 1.33] | 1.45 ms | 3.70x [3.69 – 3.71] |
| 4,194,304 | 1.70 ms | 1.84 ms | 1.08x [1.07 – 1.09] | 2.43 ms | 1.43x [1.42 – 1.44] |
| 16,777,216 | 7.20 ms | 7.38 ms | 1.02x [1.02 – 1.03] | 10.06 ms | 1.40x [1.39 – 1.40] |
| overall |  |  | 1.06x [1.06 – 1.06] |  | 1.91x [1.91 – 1.92] |

No stall.

What holds it at each size:

- **16 to 256:** the steps of a call and the cold start of each; 1.1x over the naive C.
- **1,024 to 65,536:** the second-level cache. 0.15 to 0.16 ns per element against 0.115 to 0.124 for a plain copy, 1.17x to 1.25x over the naive C.
- **262,144 and 1,048,576:** the third-level cache: 1.05x and 0.93x against the naive C.
- **4,194,304 and 16,777,216:** memory, with a third less traffic than the naive C: 1.34x and 1.37x.

The candidates the design lists for this family (`docs/x86-64-design.md`):

| Candidate | Outcome |
|---|---|
| A prefix sum across the register with an extra permute (`vpslldq` works inside each half) | Kept in another form (attempt 1): the shift by one element is an overlapping load, and one `vperm2i128` per vector does the rest. Between 512 and 131,071 elements the 16-byte version without any permute is faster (attempt 4) |
| Non-temporal stores above the third-level cache | Kept (attempt 5) |
| The overflow test as a compare | Does not apply to integers |
| Masked loads and stores for leftover elements | Not tried: open issue 3 of the design. The last elements are done one at a time |
| An output just behind the input | No cost in the judge's close table: the loads run ahead of the stores |
| Alignment of data | Rejected as a general step (attempt 2); used where the stores need it (attempt 5) |
| Alignment of code | The loops start on 64-byte boundaries. Not varied |
| Instructions slow on other x86-64 processors | None used |

**Unlike the float kernels, this one is right when `out` is `x`**, as the ARM64 kernel is; tested outside the judge, which has no such case. A partial overlap is not supported by any vector kernel.

Left untried: the cache-bypassing stores between 2 MB and 24 MB of output (1.3x in the micro-benchmark, at the cost named in attempt 5: a person's call); a prefetch.

Found here and worth trying on ARM64: nothing. The M1's scalar chain is three times slower relative to its vector loop, so its kernel is not near a copy's speed in cache.
