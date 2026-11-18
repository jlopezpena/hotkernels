# interp_f64 attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core), Linux, pinned to one performance core by the judge.

**Every judge command in this run was passed `--unlocked`,** and its results are untrusted until that is no longer needed. The reason is the reference, not the kernel: `naive.c` on `main` fuses the multiply and the add, NumPy's x86-64 build does not, and the reference fails its own check on this architecture (issue 73). This kernel sits on top of the pull request that corrects `naive.c`; with that file different from `main`, the judge refuses a locked run. Once the correction is on `main`, `uv run judge run interp_f64` has to be repeated locked and the table below replaced.

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

Final `judge run --unlocked`:

`random`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 129.8 ns | 735.7 ns | 5.67x [5.64 – 5.73] | 1.2 us | 9.58x [9.54 – 9.60] |
| 64 | 153.3 ns | 1.2 us | 8.02x [8.00 – 8.03] | 2.0 us | 12.8x [12.8 – 12.8] |
| 256 | 180.5 ns | 530.8 ns | 2.94x [2.89 – 3.03] | 1.9 us | 10.3x [10.3 – 10.4] |
| 1,024 | 194.7 ns | 430.1 ns | 2.21x [2.21 – 2.21] | 1.2 us | 6.00x [5.99 – 6.00] |
| 4,096 | 211.7 ns | 490.2 ns | 2.32x [2.31 – 2.32] | 1.3 us | 6.00x [5.99 – 6.01] |
| 16,384 | 269.6 ns | 638.8 ns | 2.37x [2.37 – 2.37] | 1.5 us | 5.72x [5.72 – 5.73] |
| 65,536 | 336.9 ns | 829.8 ns | 2.46x [2.46 – 2.46] | 1.8 us | 5.25x [5.25 – 5.26] |
| 262,144 | 415.9 ns | 1.0 us | 2.50x [2.50 – 2.50] | 2.0 us | 4.74x [4.73 – 4.74] |
| 1,048,576 | 635.9 ns | 1.1 us | 1.75x [1.74 – 1.75] | 2.0 us | 3.19x [3.17 – 3.20] |
| 4,194,304 | 843.3 ns | 1.4 us | 1.65x [1.63 – 1.68] | 2.1 us | 2.52x [2.48 – 2.55] |
| 16,777,216 | 3.0 us | 3.3 us | 1.08x [1.06 – 1.09] | 2.3 us | 0.75x [0.74 – 0.75] |
| overall |  |  | 2.56x [2.55 – 2.57] |  | 4.90x [4.89 – 4.91] |

`random, scalars=many`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 2.0 us | 9.2 us | 4.66x [4.57 – 4.81] | 11.0 us | 5.56x [5.53 – 5.62] |
| 64 | 2.3 us | 16.4 us | 7.09x [7.07 – 7.10] | 21.7 us | 9.37x [9.29 – 9.41] |
| 256 | 2.7 us | 23.7 us | 8.89x [8.88 – 8.91] | 32.2 us | 12.1x [12.1 – 12.1] |
| 1,024 | 3.1 us | 35.8 us | 11.6x [11.5 – 11.6] | 42.3 us | 13.7x [13.6 – 13.7] |
| 4,096 | 3.4 us | 28.6 us | 8.43x [8.40 – 8.46] | 43.5 us | 12.8x [12.8 – 12.8] |
| 16,384 | 4.8 us | 43.0 us | 8.89x [8.87 – 8.93] | 58.7 us | 12.1x [12.1 – 12.2] |
| 65,536 | 6.1 us | 57.7 us | 9.43x [9.42 – 9.45] | 72.1 us | 11.8x [11.7 – 11.8] |
| 262,144 | 9.1 us | 74.7 us | 8.25x [8.23 – 8.26] | 91.6 us | 10.1x [10.1 – 10.2] |
| 1,048,576 | 14.3 us | 139.3 us | 9.73x [9.65 – 9.81] | 149.8 us | 10.5x [10.4 – 10.5] |
| 4,194,304 | 33.1 us | 231.4 us | 6.99x [6.85 – 7.08] | 256.4 us | 7.74x [7.54 – 7.91] |
| 16,777,216 | 57.7 us | 454.2 us | 7.87x [7.83 – 7.90] | 395.8 us | 6.86x [6.80 – 6.95] |
| overall |  |  | 8.16x [8.13 – 8.18] |  | 9.89x [9.86 – 9.91] |

All distributions: 4.57x [4.56 – 4.58] over naive C, 6.96x [6.95 – 6.97] over NumPy

What holds it: the search, as in `searchsorted_f64`: a gather every two to three cycles inside the caches, the misses beyond. The values cost about five cycles per query on top: four gathers, and a `vdivpd` that takes eight cycles for four queries.

The candidates the design lists for this family (`docs/x86-64-design.md`):

| Candidate | Outcome |
|---|---|
| `vgatherqpd` loads four probes in one instruction | Kept, for the search and for the four values the interpolation needs |
| Scalar loads with `cmov`, sixteen or more interleaved | Kept for one to three queries only |
| `interp` divides four per vector with `vdivpd` | Kept |
| NumPy's x86-64 build may not fuse the multiply and the add, and the kernel has to do whatever NumPy does | It does not fuse (issue 73: 6,131 of 6,131 differing cases match the unfused form). The kernel does not either. The shared reference did, and is corrected in the pull request under this one |
| Gathers are slow on older AMD processors and on Intel processors with the gather mitigation | Used in every path from four queries up. Untested elsewhere |
| An output just behind the input | Not looked at: the unlocked run's close table is as untrusted as the rest |

**What differs from the ARM64 kernel for a caller.**

- **The rounding of the last step** is the platform's NumPy's: two roundings here, one on ARM64. Results on the two architectures can differ in the last bit, as NumPy's own do.
- **No narrower path for a few queries** (issue 39): one to three are done one at a time, four and more as a thousand are.
- **No limit of 2^32 sample points:** positions are 64-bit throughout.
- **`out` is working storage during the call,** and the sample points must hold no NaN, as on ARM64.

Left untried: the two `xp` values and the two `fp` values of a query loaded as pairs in place of four gathers; skipping the second expression, which is only needed when the first is a NaN.
