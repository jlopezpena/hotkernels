# interp_f64 attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core), Linux, pinned to one performance core by the judge.

**The attempt below was judged with `--unlocked`; the table under "Where it stands" is locked.** The reason was the reference, not the kernel: `naive.c` on `main` fused the multiply and the add, NumPy's x86-64 build does not, and the reference failed its own check on this architecture (issue 73). This kernel was written on top of the pull request that corrects `naive.c`, and with that file different from `main` the judge refuses a locked run. The correction has since been merged, `uv run judge run interp_f64 --record` was repeated locked on `main`, and its report is `results/interp_f64/x86_64-intel-r-core-tm-i5-14600kf.json`.

What the compiler does with the corrected `naive.c` (`clang -O3 -march=native`): one query at a time with a branching search. With 1,000 queries it takes 36 ns per query against 1,024 sample points and 139 ns against 1,048,576. NumPy takes 42 and 150.

## 1. The right-side search in step, then naive.c's cases without a branch

- **Idea:** two phases per chunk of queries, four queries per vector.
  - **The search** is `searchsorted_right_f64`'s as it stood after its x86-64 attempt 3: `half = (len + 1) / 2; if xp[lo + half - 1] <= v: lo += half; len -= half`, one `vgatherqpd` per four queries, up to 64 queries in step and 512 when `xp` does not fit the second-level cache, and `lo` kept in `out`. A NaN query compares false at every step and ends with 0.
  - **The values** follow naive.c's cases read backwards, each later one overriding the earlier by a blend: both of its expressions are computed with a real `vdivpd`, the second replacing the first where the first is a NaN, then `fp[j]` where that is still a NaN and `fp[j] == fp[j+1]`; then `fp[j]` where `v == xp[j]` or `v < xp[0]`; then `fp[n-1]` where `v >= xp[n-1]`; then `v` where `v` is a NaN. `j` is moved into `0 .. n-2` first so that the four gathers of `xp[j]`, `xp[j+1]`, `fp[j]` and `fp[j+1]` stay inside the arrays.
  - **The multiply and the add are two instructions,** each rounded, which is what NumPy does on x86-64 and the design said had to be established first. The ARM64 kernel fuses them, as NumPy does there.
  - A count that is not a multiple of four: the last four queries are done first and their values kept below the stack pointer, because the chunks overwrite `lo` with values. One to three queries take a scalar copy of naive.c. One sample point: every answer is `fp[0]`.
- **check:** pass, first time (157,062 cases, unlocked). The judge's correctness sweep stops at 65,537 sample points, below the size where the chunk grows to 512 queries, so that path was tested outside it, together with the two `searchsorted` kernels: 2,520 runs against the three reference C files, with 1 to 700,000 sample points on both sides of 262,144, 1 to 3,000 queries on both sides of 64 and 512, and queries that are random, sample points themselves, one unit in the last place either side of a sample point, NaN and the infinities; sample points with repeats and with infinities at the ends; infinite `fp` values, which make the first expression a NaN. Bit-identical in every run.
- **compare:** not run (first version). `run`, unlocked: 4.57x over naive C and 6.96x over NumPy overall; with 1,000 queries 8.16x and 9.89x.
- **Kept:** yes
- **Notes:** with 1,000 queries: 3.1 ns per query against 1,024 sample points, 14 ns against 1,048,576, 58 ns against 16,777,216. With 64 queries against 16,777,216 sample points the kernel is slower than NumPy (0.75x): the judge repeats the same 64 queries, so NumPy's branches are learnt and its probes are in the cache (see the `searchsorted_f64` log).

## Where it stands

Final `judge run`, locked, on `main` after the correction to the reference was merged (157,062 cases pass). The unlocked run it replaces read 4.57x over naive C and 6.96x over NumPy overall.

`random`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 130.0 ns | 697.0 ns | 5.36x [5.32 – 5.46] | 1.3 us | 9.68x [9.65 – 9.82] |
| 64 | 153.6 ns | 1.2 us | 8.04x [8.01 – 8.05] | 1.9 us | 12.7x [12.6 – 12.7] |
| 256 | 178.7 ns | 514.6 ns | 2.88x [2.81 – 2.93] | 1.9 us | 10.5x [10.5 – 10.6] |
| 1,024 | 208.7 ns | 428.7 ns | 2.05x [2.05 – 2.06] | 1.2 us | 5.62x [5.62 – 5.63] |
| 4,096 | 209.1 ns | 490.4 ns | 2.35x [2.34 – 2.35] | 1.3 us | 6.09x [6.08 – 6.10] |
| 16,384 | 316.6 ns | 630.4 ns | 1.99x [1.99 – 1.99] | 1.5 us | 4.80x [4.80 – 4.81] |
| 65,536 | 339.4 ns | 830.1 ns | 2.45x [2.44 – 2.45] | 1.8 us | 5.21x [5.21 – 5.22] |
| 262,144 | 419.2 ns | 1.0 us | 2.48x [2.48 – 2.49] | 2.0 us | 4.71x [4.71 – 4.72] |
| 1,048,576 | 643.0 ns | 1.1 us | 1.71x [1.71 – 1.71] | 2.0 us | 3.14x [3.14 – 3.15] |
| 4,194,304 | 1.0 us | 1.7 us | 1.68x [1.53 – 1.72] | 2.1 us | 2.13x [1.91 – 2.17] |
| 16,777,216 | 3.1 us | 3.2 us | 1.06x [1.06 – 1.06] | 2.3 us | 0.75x [0.75 – 0.75] |
| overall |  |  | 2.48x [2.46 – 2.49] |  | 4.72x [4.68 – 4.74] |

`random, scalars=many`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 2.0 us | 9.4 us | 4.76x [4.70 – 4.83] | 11.0 us | 5.53x [5.49 – 5.56] |
| 64 | 2.3 us | 16.5 us | 7.11x [7.09 – 7.12] | 21.7 us | 9.38x [9.34 – 9.42] |
| 256 | 2.7 us | 23.7 us | 8.84x [8.83 – 8.85] | 32.3 us | 12.1x [12.0 – 12.1] |
| 1,024 | 3.0 us | 36.3 us | 12.0x [11.9 – 12.0] | 42.1 us | 14.0x [13.9 – 14.0] |
| 4,096 | 3.6 us | 28.4 us | 7.91x [7.87 – 7.96] | 43.0 us | 12.0x [12.0 – 12.1] |
| 16,384 | 4.8 us | 42.9 us | 9.03x [9.02 – 9.06] | 58.7 us | 12.4x [12.3 – 12.4] |
| 65,536 | 6.1 us | 57.2 us | 9.36x [9.34 – 9.38] | 72.2 us | 11.8x [11.8 – 11.9] |
| 262,144 | 9.1 us | 76.1 us | 8.37x [8.35 – 8.39] | 93.8 us | 10.3x [10.3 – 10.4] |
| 1,048,576 | 14.0 us | 116.1 us | 8.31x [8.27 – 8.36] | 148.7 us | 10.6x [10.6 – 10.7] |
| 4,194,304 | 34.0 us | 234.6 us | 6.90x [6.79 – 7.00] | 238.1 us | 7.00x [6.77 – 7.14] |
| 16,777,216 | 58.1 us | 443.7 us | 7.64x [7.60 – 7.70] | 328.1 us | 5.65x [5.61 – 5.70] |
| overall |  |  | 8.02x [8.00 – 8.04] |  | 9.63x [9.61 – 9.66] |

All distributions: 4.46x [4.44 – 4.47] over naive C, 6.75x [6.71 – 6.76] over NumPy

With the output 16 bytes behind the queries (64 queries; the judge times this placement on that set only), against the same kernel with the output clear:

`random`:

| n | kernel | apart | speedup [95%] | naive C | speedup [95%] |
|---|---|---|---|---|---|
| 4,096 | 210.8 ns | 209.9 ns | 1.00x [0.99 – 1.00] | 490.1 ns | 2.32x [2.32 – 2.33] |
| 16,384 | 352.3 ns | 267.2 ns | 0.76x [0.76 – 0.76] | 642.3 ns | 1.82x [1.82 – 1.82] |
| 65,536 | 340.0 ns | 338.5 ns | 1.00x [0.99 – 1.00] | 825.2 ns | 2.43x [2.43 – 2.43] |
| 262,144 | 475.4 ns | 375.4 ns | 0.79x [0.79 – 0.79] | 1.0 us | 2.10x [2.10 – 2.11] |
| 1,048,576 | 663.4 ns | 664.6 ns | 1.00x [1.00 – 1.00] | 1.1 us | 1.67x [1.67 – 1.68] |
| 4,194,304 | 1.7 us | 1.5 us | 0.89x [0.89 – 0.89] | 1.8 us | 1.10x [1.10 – 1.10] |
| 16,777,216 | 3.0 us | 2.6 us | 0.88x [0.85 – 0.92] | 3.2 us | 1.08x [1.05 – 1.12] |
| overall |  |  | 0.90x [0.89 – 0.90] |  | 1.71x [1.70 – 1.72] |

The kernel is slower at that placement: 0.90x overall, 0.76x and 0.79x at 16,384 and 262,144 sample points and level at 4,096, 65,536 and 1,048,576. It keeps each search's position in `out` and reads the queries again for every step, so a store to `out` can be taken for a store to the queries about to be loaded. Nothing was tried against it. It stays 1.71x over the naive C there.

What holds it: the search, as in `searchsorted_f64`: a gather every two to three cycles inside the caches, the misses beyond. The values cost about five cycles per query on top: four gathers, and a `vdivpd` that takes eight cycles for four queries.

The candidates the design lists for this family (`docs/x86-64-design.md`):

| Candidate | Outcome |
|---|---|
| `vgatherqpd` loads four probes in one instruction | Kept, for the search and for the four values the interpolation needs |
| Scalar loads with `cmov`, sixteen or more interleaved | Kept for one to three queries only |
| `interp` divides four per vector with `vdivpd` | Kept |
| NumPy's x86-64 build may not fuse the multiply and the add, and the kernel has to do whatever NumPy does | It does not fuse (issue 73: 6,131 of 6,131 differing cases match the unfused form). The kernel does not either. The shared reference did, and was corrected in a pull request of its own, merged before the final run |
| Gathers are slow on older AMD processors and on Intel processors with the gather mitigation | Used in every path from four queries up. Untested elsewhere |
| An output just behind the input | Slower there by 0.90x overall with 64 queries (table above); no path was written for it |

**What differs from the ARM64 kernel for a caller.**

- **The rounding of the last step** is the platform's NumPy's: two roundings here, one on ARM64. Results on the two architectures can differ in the last bit, as NumPy's own do.
- **No narrower path for a few queries** (issue 39): one to three are done one at a time, four and more as a thousand are.
- **No limit of 2^32 sample points:** positions are 64-bit throughout.
- **`out` is working storage during the call,** and the sample points must hold no NaN, as on ARM64.

Left untried: the two `xp` values and the two `fp` values of a query loaded as pairs in place of four gathers; skipping the second expression, which is only needed when the first is a NaN.
