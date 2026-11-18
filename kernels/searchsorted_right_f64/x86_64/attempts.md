# searchsorted_right_f64 attempts (x86_64)

All runs on an Intel Core i5-14600KF, Linux, locked, pinned to one performance core by the judge. This kernel is `searchsorted_f64` as it stood after its x86-64 attempt 3, with two changes; its log has the measurements that led there.

## 1. The left kernel with `<=` for `<`, and n for a NaN needle

- **Idea:** side='right' advances when `a[lo + half - 1] <= needle`: `vcmplepd` in the vector paths and `cmovae` in the path for one to three needles. A NaN in `a` still compares false and is never passed, which is right for every needle that is not a NaN. A NaN needle goes after everything, so its answer is n and needs no search: when a NaN needle was seen, n is written over the answers of the NaN needles.
- **check:** pass
- **compare:** not run (first version)
- **Kept:** yes
- **Notes:** the two kernels are generated from one source and differ in the compare, the conditional move and the fix-up for NaN needles.

## Where it stands

Final `judge run`:

`random`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 52.5 ns | 483.6 ns | 9.21x [9.00 – 9.94] | 1.3 us | 24.9x [24.8 – 25.8] |
| 64 | 74.4 ns | 1.1 us | 15.4x [15.4 – 15.5] | 2.0 us | 26.2x [26.1 – 26.3] |
| 256 | 98.1 ns | 358.3 ns | 3.65x [3.65 – 3.65] | 1.7 us | 17.4x [17.3 – 17.5] |
| 1,024 | 113.4 ns | 408.6 ns | 3.60x [3.60 – 3.61] | 1.0 us | 9.26x [9.24 – 9.27] |
| 4,096 | 134.3 ns | 493.2 ns | 3.67x [3.67 – 3.67] | 1.1 us | 8.21x [8.20 – 8.23] |
| 16,384 | 183.8 ns | 640.6 ns | 3.49x [3.48 – 3.49] | 1.2 us | 6.67x [6.66 – 6.68] |
| 65,536 | 248.6 ns | 840.4 ns | 3.38x [3.38 – 3.38] | 1.4 us | 5.59x [5.59 – 5.60] |
| 262,144 | 329.9 ns | 1.1 us | 3.30x [3.30 – 3.30] | 1.5 us | 4.58x [4.58 – 4.59] |
| 1,048,576 | 588.9 ns | 1.2 us | 2.03x [2.02 – 2.03] | 1.8 us | 2.98x [2.97 – 2.99] |
| 4,194,304 | 1.5 us | 1.9 us | 1.27x [1.27 – 1.27] | 2.7 us | 1.78x [1.77 – 1.78] |
| 16,777,216 | 2.8 us | 3.5 us | 1.25x [1.23 – 1.26] | 4.2 us | 1.51x [1.50 – 1.52] |
| overall |  |  | 3.46x [3.45 – 3.49] |  | 6.68x [6.67 – 6.70] |

`random, scalars=many`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 795.4 ns | 5.0 us | 6.35x [6.20 – 6.46] | 11.9 us | 14.9x [14.9 – 15.0] |
| 64 | 1.1 us | 15.8 us | 13.7x [13.7 – 13.8] | 21.6 us | 18.8x [18.7 – 18.9] |
| 256 | 1.5 us | 23.5 us | 15.8x [15.7 – 15.9] | 29.5 us | 19.8x [19.7 – 19.8] |
| 1,024 | 1.8 us | 33.1 us | 18.1x [18.0 – 18.1] | 38.2 us | 20.9x [20.9 – 20.9] |
| 4,096 | 2.1 us | 8.9 us | 4.20x [4.17 – 4.24] | 36.3 us | 17.1x [17.1 – 17.2] |
| 16,384 | 3.3 us | 20.5 us | 6.27x [6.09 – 6.37] | 49.2 us | 15.0x [15.0 – 15.1] |
| 65,536 | 4.6 us | 40.6 us | 8.87x [8.81 – 8.95] | 60.6 us | 13.3x [13.2 – 13.3] |
| 262,144 | 7.1 us | 56.4 us | 7.91x [7.83 – 7.95] | 72.6 us | 10.2x [10.2 – 10.2] |
| 1,048,576 | 12.6 us | 119.0 us | 9.46x [9.39 – 9.51] | 82.5 us | 6.56x [6.54 – 6.57] |
| 4,194,304 | 31.5 us | 214.2 us | 6.81x [6.72 – 6.91] | 103.6 us | 3.29x [3.27 – 3.31] |
| 16,777,216 | 55.4 us | 426.7 us | 7.70x [7.55 – 7.78] | 142.4 us | 2.57x [2.52 – 2.60] |
| overall |  |  | 8.74x [8.70 – 8.77] |  | 10.8x [10.7 – 10.8] |

All distributions: 5.50x [5.49 – 5.52] over naive C, 8.48x [8.47 – 8.49] over NumPy

With the output buffer 16 bytes behind its input (64 needles; "apart" is the same kernel with the output clear of the input):

`random`:

| n | kernel | apart | speedup [95%] | naive C | speedup [95%] |
|---|---|---|---|---|---|
| 4,096 | 134.2 ns | 134.1 ns | 1.00x [1.00 – 1.00] | 493.1 ns | 3.67x [3.67 – 3.67] |
| 16,384 | 183.7 ns | 184.3 ns | 1.00x [1.00 – 1.00] | 641.7 ns | 3.49x [3.49 – 3.49] |
| 65,536 | 253.7 ns | 258.8 ns | 1.02x [1.02 – 1.02] | 837.4 ns | 3.30x [3.30 – 3.30] |
| 262,144 | 326.6 ns | 327.5 ns | 1.00x [1.00 – 1.00] | 1.1 us | 3.33x [3.33 – 3.33] |
| 1,048,576 | 447.1 ns | 622.6 ns | 1.39x [1.39 – 1.39] | 1.2 us | 2.71x [2.71 – 2.71] |
| 4,194,304 | 1.5 us | 1.5 us | 0.96x [0.95 – 0.96] | 2.0 us | 1.30x [1.29 – 1.31] |
| 16,777,216 | 2.8 us | 2.7 us | 0.94x [0.94 – 0.95] | 3.4 us | 1.19x [1.18 – 1.20] |
| overall |  |  | 1.04x [1.04 – 1.04] |  | 2.49x [2.48 – 2.49] |

Everything in the `searchsorted_f64` log under "where it stands" applies: what holds it, the candidates, and what differs from the ARM64 kernel for a caller. `vgatherqpd` is used in every path from four needles up and is untested on other x86-64 processors.
