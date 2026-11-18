# argmin_f64 attempts

This log holds two architectures. ARM64 (Apple M1 Pro) comes first. x86-64 (Intel Core i5-14600KF) follows under [x86-64](#x86-64), with its own numbering.

## 1. Scalar baseline

- **Idea:** simplest correct version: one element per step, `fcmp` for NaN and for strictly smaller.
- **check:** pass
- **compare:** n/a (first version)
- **Kept:** yes
- **Notes:** starting point only; same shape as naive C.

## 2. Port of the tuned argmax design

- **Idea:** the argmax_f64 kernel with the comparison reversed (see `kernels/argmax_f32/attempts.md` for how the design was reached and `kernels/argmax_f64/attempts.md` for the two-lane port): `fmin`/`fminp` instead of `fmax`/`fmaxp`, and `csel ... mi` (strictly less) for the best-block pointer. Above 64 elements: 32-element blocks, 15 `fmin` and one scalar `fminp` per block, one `fcmp` driving `b.vs` (NaN in this block), `csel` and a scalar `fmin`; at the end only the best block is rescanned. Up to 64 elements: straight-line code over a front half and an overlapping back half; two-step search through 8-element "quads" above 16. Below 4 elements: scalar.
- **check:** pass
- **compare:** significantly faster, 7.43x [7.38 – 7.47]
- **Kept:** yes
- **Notes:** timings match argmax_f64 size for size. Lessons carried over from argmax_f32: software prefetch hurts (its attempt 5); a deeper two-step rescan after the block loop is slower because that phase is paid for by its latency (9); lane-wise block-number tracking is slower than scalar `csel` tracking (10).

## 3. Two-step search for 8 < n <= 16

- **Idea:** same change as argmax_f64 attempt 3: compare the four pair minima the tree already produces, then reload and compare only the two vectors of the first pair that holds the minimum (22 vector operations instead of 25).
- **check:** pass
- **compare:** significantly faster in three of five runs: 1.005x [1.004 – 1.005], 1.006x [1.004 – 1.007], 1.006x [1.001 – 1.009]. The other two were not significant, with wide intervals from noise at the large sizes: 1.004x [1.000 – 1.012] and 0.992x [0.978 – 1.029].
- **Kept:** yes
- **Notes:** n = 16 was 1.06x in all five runs (2.6 -> 2.4 ns) and no other size is touched by the change. The overall gain is small enough that noise at 4M and 16M can hide it; kept because the per-size effect is stable and the same change passed twice out of twice on argmax_f64.

## Final result (after attempt 3)

`uv run judge run argmin_f64 --unlocked`: check PASS (18,081 cases). Overall 12.2x [12.10 – 12.23] over naive C and 16.8x [16.74 – 16.92] over NumPy. The kernel is argmax_f64 with the comparison reversed, instruction for instruction.

| n | kernel | naive C | speedup | NumPy | speedup |
|---|---|---|---|---|---|
| 16 | 2.5 ns | 10.2 ns | 4.15x [4.11 – 4.19] | 600.4 ns | 245x [242 – 247] |
| 64 | 6.3 ns | 53.2 ns | 8.42x [8.41 – 8.48] | 651.5 ns | 103x [103 – 104] |
| 256 | 19.1 ns | 296.1 ns | 15.5x [15.44 – 15.80] | 784.8 ns | 41.1x [40.88 – 41.55] |
| 1,024 | 61.9 ns | 1.3 us | 21.3x [21.17 – 21.50] | 1.3 us | 21.8x [21.63 – 22.03] |
| 4,096 | 238.3 ns | 5.4 us | 22.7x [22.59 – 22.86] | 3.5 us | 14.9x [14.85 – 14.99] |
| 16,384 | 950.4 ns | 21.9 us | 23.0x [22.67 – 23.27] | 12.8 us | 13.5x [13.26 – 13.58] |
| 65,536 | 6.5 us | 86.6 us | 13.3x [13.05 – 13.46] | 47.8 us | 7.33x [7.20 – 7.43] |
| 262,144 | 26.0 us | 344.7 us | 13.3x [13.15 – 13.39] | 187.8 us | 7.23x [7.14 – 7.30] |
| 1,048,576 | 119.2 us | 1.38 ms | 11.6x [11.43 – 11.70] | 745.9 us | 6.26x [6.20 – 6.34] |
| 4,194,304 | 682.9 us | 5.50 ms | 8.06x [7.68 – 8.37] | 2.99 ms | 4.38x [4.17 – 4.53] |
| 16,777,216 | 2.48 ms | 21.63 ms | 8.72x [8.52 – 8.87] | 11.65 ms | 4.69x [4.58 – 4.79] |

Not measured by the judge, timed outside it (best of 7, one input): ascending, descending and all-equal inputs take the same time as random ones (about 890 to 920 ns at 16,384). A NaN as the first element returns in about 3 ns at any size; a NaN as the last element costs 25 ns at 256 and 907 ns at 16,384.

# x86-64

All runs on an Intel Core i5-14600KF, Linux, locked, pinned to one performance core by the judge. This kernel is generated from the same source as `argmax_f32`, whose log (under "x86-64") has the reasoning, the candidates and what differs from ARM64 for a caller. Here: `vminpd` and "M is larger than a lane", 32 elements per block, four per vector.

## x86-64 1. Block reduction with one rare branch, then the position from the remembered block

- **Idea:** as `argmax_f32`, x86-64 attempt 1.
- **check:** pass, first time
- **compare:** not run (first version)
- **Kept:** yes
- **Notes:** none beyond that log's.

## x86-64 2. An answer inside the array whatever happens to the data

- **Idea:** as `argmax_f32`, x86-64 attempt 2.
- **check:** pass
- **compare:** not run
- **Kept:** yes
- **Notes:** none.

## x86-64: where it stands

Final `judge run`:

`random`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 5.6 ns | 10.2 ns | 1.82x [1.82 – 1.83] | 526.4 ns | 94.4x [94.2 – 94.8] |
| 64 | 10.2 ns | 39.1 ns | 3.84x [3.83 – 3.84] | 563.8 ns | 55.3x [55.2 – 55.4] |
| 256 | 24.0 ns | 188.0 ns | 7.84x [7.83 – 7.84] | 570.0 ns | 23.8x [23.7 – 23.8] |
| 1,024 | 69.0 ns | 778.7 ns | 11.3x [11.3 – 11.3] | 702.7 ns | 10.2x [10.2 – 10.2] |
| 4,096 | 249.2 ns | 3.1 us | 12.6x [12.6 – 12.6] | 1.2 us | 4.85x [4.85 – 4.85] |
| 16,384 | 967.4 ns | 12.6 us | 13.0x [13.0 – 13.0] | 3.2 us | 3.29x [3.29 – 3.29] |
| 65,536 | 4.2 us | 50.4 us | 11.9x [11.9 – 11.9] | 11.3 us | 2.66x [2.66 – 2.66] |
| 262,144 | 19.6 us | 202.0 us | 10.3x [10.3 – 10.3] | 43.8 us | 2.24x [2.23 – 2.24] |
| 1,048,576 | 91.0 us | 811.9 us | 8.92x [8.91 – 8.93] | 172.9 us | 1.90x [1.90 – 1.90] |
| 4,194,304 | 844.8 us | 3.35 ms | 3.96x [3.95 – 3.98] | 1.19 ms | 1.41x [1.41 – 1.42] |
| 16,777,216 | 4.17 ms | 13.39 ms | 3.21x [3.20 – 3.22] | 5.38 ms | 1.29x [1.29 – 1.30] |
| overall |  |  | 6.79x [6.79 – 6.79] |  | 6.07x [6.07 – 6.08] |
