# argmin_f32 attempts

This log holds two architectures. ARM64 (Apple M1 Pro) comes first. x86-64 (Intel Core i5-14600KF) follows under [x86-64](#x86-64), with its own numbering.

## 1. Scalar baseline

- **Idea:** simplest correct version: one element per step, `fcmp` for NaN and for strictly smaller.
- **check:** pass
- **compare:** n/a (first version)
- **Kept:** yes
- **Notes:** starting point only; same shape as naive C.

## 2. Port of the tuned argmax_f32 design

- **Idea:** the final argmax_f32 kernel with the comparison reversed (see `kernels/argmax_f32/attempts.md` for how it was reached): `fmin`/`fminv` instead of `fmax`/`fmaxv`, `csel ... mi` (strictly less) for the best-block pointer, and the running minimum starts from `x[0]` instead of an infinity constant. Above 64 elements: 64-element blocks, 15 `fmin` and one `fminv` per block, one `fcmp` driving `b.vs` (NaN in this block), `csel` and a scalar `fmin`; at the end only the best block is rescanned. Up to 64 elements: straight-line code over a front half and an overlapping back half; two-step search through "quads" above 32. Below 8 elements: scalar.
- **check:** pass
- **compare:** significantly faster, 14.1x [14.04 – 14.13]
- **Kept:** yes
- **Notes:** timings match argmax_f32 size for size (0.027 ns per element in the first-level cache, 0.049 in the second level, about 0.07 at 16.7M). Lessons carried over from argmax_f32: `fmaxp` broadcast instead of `fmaxv` + `dup` makes no difference (its attempt 3); software prefetch hurts (5); a deeper two-step rescan after the block loop is slower because that phase is paid for by its latency (9); lane-wise block-number tracking is slower than scalar `csel` tracking (10).

## 3. Straight-line path for 128 < n <= 256 over sixteen quads

- **Idea:** same change as argmax_f32 attempt 15: build sixteen quad minima (eight from the start, eight ending at n), take the minimum, find the first quad that holds it from one 64-bit mask, then compare that quad's 16 elements, instead of four rounds of the block loop plus a rescan.
- **check:** pass
- **compare:** significantly faster, 1.003x [1.002 – 1.006], repeated: 1.005x [1.004 – 1.006]
- **Kept:** yes
- **Notes:** n = 256 goes from 13.2 to 12.7 ns (1.04x, 1.05x on the repeat); no other timed size changes. Other argmax_f32 attempts not repeated here because they failed there on code this kernel shares instruction for instruction: one integer operation fewer on the n <= 16 path (its attempt 7), pairwise reduction instead of `fminv` in the loop (11, 13), loop alignment (12), nibble masks with a `csel` tree at the end of the rescan (14).

## Final result (after attempt 3)

`uv run judge run argmin_f32 --unlocked`: check PASS (18,081 cases). Overall 22.8x [22.68 – 22.85] over naive C and 22.7x [22.62 – 22.81] over NumPy. The kernel is argmax_f32 with the comparison reversed, instruction for instruction.

| n | kernel | naive C | speedup | NumPy | speedup |
|---|---|---|---|---|---|
| 16 | 1.7 ns | 10.3 ns | 6.03x [6.00 – 6.07] | 609.0 ns | 357x [354 – 360] |
| 64 | 4.1 ns | 54.9 ns | 13.3x [13.18 – 13.36] | 668.0 ns | 162x [160 – 163] |
| 256 | 12.9 ns | 303.9 ns | 23.6x [23.50 – 23.74] | 730.9 ns | 56.8x [56.55 – 57.25] |
| 1,024 | 34.1 ns | 1.3 us | 38.9x [38.73 – 39.08] | 1.1 us | 31.5x [31.37 – 31.68] |
| 4,096 | 119.4 ns | 5.4 us | 45.6x [45.26 – 45.91] | 2.5 us | 20.5x [20.38 – 20.70] |
| 16,384 | 459.0 ns | 21.6 us | 47.1x [46.58 – 47.35] | 7.8 us | 17.1x [16.95 – 17.15] |
| 65,536 | 3.3 us | 86.9 us | 26.6x [26.44 – 26.78] | 29.6 us | 9.06x [9.00 – 9.14] |
| 262,144 | 13.0 us | 342.3 us | 26.4x [25.81 – 27.04] | 115.8 us | 8.92x [8.73 – 9.12] |
| 1,048,576 | 54.5 us | 1.37 ms | 25.1x [24.88 – 26.05] | 457.8 us | 8.40x [8.30 – 8.75] |
| 4,194,304 | 300.0 us | 5.37 ms | 17.9x [17.71 – 18.06] | 1.79 ms | 5.95x [5.89 – 6.02] |
| 16,777,216 | 1.27 ms | 21.52 ms | 17.0x [16.74 – 17.04] | 7.15 ms | 5.64x [5.57 – 5.67] |

Not measured by the judge, timed outside it (best of 7, one input): ascending, descending and all-equal inputs take the same time as random ones (about 440 ns at 16,384 in all four cases). A NaN as the first element returns in 3.5 to 10 ns at any size; a NaN as the last element costs 114 ns at 256 and 472 ns at 16,384.

# x86-64

All runs on an Intel Core i5-14600KF, Linux, locked, pinned to one performance core by the judge. This kernel is generated from the same source as `argmax_f32`, whose log (under "x86-64") has the reasoning, the candidates and what differs from ARM64 for a caller. Here: `vminps` and "M is larger than a lane", 64 elements per block.

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
| 16 | 4.3 ns | 10.4 ns | 2.39x [2.38 – 2.42] | 525.6 ns | 121x [121 – 123] |
| 64 | 7.2 ns | 39.8 ns | 5.53x [5.52 – 5.53] | 553.7 ns | 77.0x [76.9 – 77.1] |
| 256 | 16.1 ns | 188.0 ns | 11.7x [11.7 – 11.7] | 557.1 ns | 34.7x [34.6 – 34.7] |
| 1,024 | 39.2 ns | 778.7 ns | 19.9x [19.9 – 19.9] | 618.4 ns | 15.8x [15.8 – 15.8] |
| 4,096 | 133.2 ns | 3.1 us | 23.6x [23.6 – 23.6] | 855.7 ns | 6.43x [6.42 – 6.43] |
| 16,384 | 491.6 ns | 12.6 us | 25.6x [25.6 – 25.6] | 1.9 us | 3.83x [3.83 – 3.83] |
| 65,536 | 1.9 us | 50.4 us | 26.4x [26.4 – 26.4] | 5.8 us | 3.05x [3.05 – 3.05] |
| 262,144 | 9.4 us | 201.8 us | 21.5x [21.5 – 21.5] | 22.0 us | 2.34x [2.34 – 2.34] |
| 1,048,576 | 45.6 us | 812.1 us | 17.8x [17.7 – 17.8] | 87.3 us | 1.91x [1.91 – 1.92] |
| 4,194,304 | 188.5 us | 3.23 ms | 17.2x [17.0 – 17.2] | 361.3 us | 1.92x [1.90 – 1.93] |
| 16,777,216 | 1.98 ms | 12.83 ms | 6.47x [6.43 – 6.51] | 2.56 ms | 1.29x [1.28 – 1.30] |
| overall |  |  | 13.2x [13.2 – 13.2] |  | 7.50x [7.50 – 7.51] |
