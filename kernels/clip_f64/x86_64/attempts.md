# clip_f64 attempts (x86_64)

All runs on an Intel Core i5-14600KF, Linux, locked, pinned to one performance core by the judge. This kernel is generated from the same source as `clip_f32`, whose log has the reasoning, the candidates, the test outside the judge (3,936 runs for this kernel too, no mismatch) and the question for a person about the caller's constraints. Here: `vmaxpd` and `vminpd`, four elements per vector.

## 1. `vmaxpd` then `vminpd` with the bound first; aligned stores; loads a block ahead

- **Idea:** as `clip_f32`, x86-64 attempt 1.
- **check:** pass, first time
- **compare:** not run (first version). `run`: 1.08x over naive C and 6.81x over NumPy overall.
- **Kept:** yes
- **Notes:** a `float64` is twice the bytes per element, so memory is the limit from smaller arrays on, and the kernel moves the same bytes as the naive C. 1.04x to 1.13x over it across the six tables.

## Where it stands

Final `judge run`:

`random`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 1.7 ns | 2.1 ns | 1.23x [1.23 – 1.23] | 1.1 us | 622x [621 – 623] |
| 64 | 9.8 ns | 10.1 ns | 1.04x [1.03 – 1.04] | 1.1 us | 112x [112 – 112] |
| 256 | 31.1 ns | 35.0 ns | 1.13x [1.12 – 1.13] | 1.2 us | 37.6x [37.6 – 37.7] |
| 1,024 | 120.2 ns | 129.2 ns | 1.07x [1.07 – 1.08] | 1.4 us | 11.3x [11.3 – 11.3] |
| 4,096 | 473.5 ns | 565.0 ns | 1.19x [1.19 – 1.19] | 1.9 us | 3.96x [3.96 – 3.97] |
| 16,384 | 1.9 us | 2.3 us | 1.22x [1.21 – 1.22] | 3.8 us | 2.01x [2.01 – 2.01] |
| 65,536 | 7.5 us | 7.6 us | 1.01x [1.01 – 1.01] | 12.7 us | 1.68x [1.68 – 1.69] |
| 262,144 | 61.1 us | 62.6 us | 1.02x [1.02 – 1.03] | 63.9 us | 1.05x [1.04 – 1.05] |
| 1,048,576 | 272.4 us | 258.8 us | 0.95x [0.95 – 0.96] | 337.9 us | 1.24x [1.23 – 1.25] |
| 4,194,304 | 1.68 ms | 2.23 ms | 1.32x [1.32 – 1.33] | 3.70 ms | 2.20x [2.19 – 2.21] |
| 16,777,216 | 6.76 ms | 9.01 ms | 1.33x [1.33 – 1.34] | 14.92 ms | 2.21x [2.20 – 2.21] |
| overall |  |  | 1.13x [1.13 – 1.13] |  | 7.15x [7.14 – 7.15] |

`random, scalars=zero_lo`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 1.7 ns | 2.1 ns | 1.22x [1.22 – 1.22] | 1.1 us | 620x [619 – 620] |
| 64 | 9.6 ns | 10.2 ns | 1.07x [1.06 – 1.07] | 1.1 us | 115x [115 – 115] |
| 256 | 31.0 ns | 33.5 ns | 1.08x [1.08 – 1.08] | 1.2 us | 37.8x [37.8 – 37.8] |
| 1,024 | 119.4 ns | 130.3 ns | 1.09x [1.09 – 1.09] | 1.3 us | 11.3x [11.3 – 11.3] |
| 4,096 | 471.0 ns | 472.1 ns | 1.00x [1.00 – 1.00] | 1.9 us | 4.01x [4.00 – 4.01] |
| 16,384 | 1.9 us | 2.2 us | 1.16x [1.16 – 1.16] | 3.8 us | 2.00x [2.00 – 2.01] |
| 65,536 | 7.5 us | 7.6 us | 1.01x [1.01 – 1.01] | 12.7 us | 1.69x [1.69 – 1.69] |
| 262,144 | 98.1 us | 64.1 us | 0.65x [0.65 – 0.65] | 63.0 us | 0.64x [0.64 – 0.64] |
| 1,048,576 | 378.6 us | 278.4 us | 0.74x [0.73 – 0.74] | 321.4 us | 0.85x [0.84 – 0.86] |
| 4,194,304 | 1.70 ms | 2.23 ms | 1.31x [1.31 – 1.32] | 3.67 ms | 2.15x [2.15 – 2.16] |
| 16,777,216 | 6.89 ms | 9.18 ms | 1.33x [1.33 – 1.34] | 14.89 ms | 2.16x [2.15 – 2.17] |
| overall |  |  | 1.04x [1.04 – 1.04] |  | 6.61x [6.60 – 6.61] |

`sparse_negative_zeros`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 1.7 ns | 2.1 ns | 1.22x [1.22 – 1.22] | 1.1 us | 625x [624 – 625] |
| 64 | 9.5 ns | 10.4 ns | 1.10x [1.10 – 1.10] | 1.1 us | 116x [116 – 117] |
| 256 | 30.5 ns | 34.3 ns | 1.12x [1.12 – 1.12] | 1.2 us | 39.1x [39.1 – 39.2] |
| 1,024 | 119.5 ns | 128.5 ns | 1.08x [1.08 – 1.08] | 1.4 us | 11.5x [11.5 – 11.5] |
| 4,096 | 474.4 ns | 567.6 ns | 1.20x [1.20 – 1.20] | 1.9 us | 4.00x [3.99 – 4.00] |
| 16,384 | 2.0 us | 1.9 us | 0.96x [0.96 – 0.96] | 3.8 us | 1.92x [1.92 – 1.92] |
| 65,536 | 7.5 us | 7.6 us | 1.01x [1.00 – 1.01] | 11.4 us | 1.52x [1.52 – 1.52] |
| 262,144 | 61.1 us | 63.1 us | 1.03x [1.03 – 1.03] | 64.3 us | 1.05x [1.05 – 1.05] |
| 1,048,576 | 384.7 us | 286.1 us | 0.74x [0.74 – 0.75] | 327.9 us | 0.85x [0.85 – 0.87] |
| 4,194,304 | 1.68 ms | 2.19 ms | 1.30x [1.30 – 1.31] | 3.74 ms | 2.22x [2.20 – 2.24] |
| 16,777,216 | 6.92 ms | 9.13 ms | 1.32x [1.31 – 1.33] | 14.89 ms | 2.15x [2.14 – 2.17] |
| overall |  |  | 1.09x [1.09 – 1.09] |  | 6.88x [6.88 – 6.90] |

`sparse_negative_zeros, scalars=zero_lo`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 1.8 ns | 2.1 ns | 1.21x [1.21 – 1.21] | 1.1 us | 615x [614 – 615] |
| 64 | 9.2 ns | 10.5 ns | 1.15x [1.15 – 1.15] | 1.1 us | 119x [119 – 119] |
| 256 | 31.7 ns | 34.7 ns | 1.09x [1.09 – 1.10] | 1.2 us | 37.5x [37.4 – 37.5] |
| 1,024 | 119.1 ns | 131.2 ns | 1.10x [1.10 – 1.10] | 1.4 us | 11.3x [11.3 – 11.4] |
| 4,096 | 472.1 ns | 476.4 ns | 1.01x [1.01 – 1.01] | 1.9 us | 4.00x [4.00 – 4.00] |
| 16,384 | 2.0 us | 1.9 us | 0.96x [0.96 – 0.96] | 3.8 us | 1.93x [1.93 – 1.93] |
| 65,536 | 7.5 us | 7.6 us | 1.01x [1.01 – 1.01] | 11.4 us | 1.51x [1.51 – 1.51] |
| 262,144 | 61.2 us | 63.3 us | 1.04x [1.03 – 1.04] | 64.2 us | 1.05x [1.05 – 1.05] |
| 1,048,576 | 387.0 us | 291.6 us | 0.75x [0.75 – 0.76] | 326.3 us | 0.84x [0.84 – 0.86] |
| 4,194,304 | 1.73 ms | 2.26 ms | 1.31x [1.30 – 1.32] | 3.70 ms | 2.14x [2.13 – 2.15] |
| 16,777,216 | 6.77 ms | 8.98 ms | 1.33x [1.32 – 1.33] | 14.87 ms | 2.19x [2.19 – 2.20] |
| overall |  |  | 1.07x [1.07 – 1.08] |  | 6.83x [6.82 – 6.84] |

`rounded`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 1.7 ns | 2.1 ns | 1.22x [1.22 – 1.22] | 1.1 us | 619x [618 – 619] |
| 64 | 9.1 ns | 10.5 ns | 1.16x [1.16 – 1.16] | 1.1 us | 121x [121 – 121] |
| 256 | 30.8 ns | 35.6 ns | 1.16x [1.16 – 1.16] | 1.2 us | 38.5x [38.4 – 38.5] |
| 1,024 | 125.0 ns | 138.7 ns | 1.11x [1.11 – 1.11] | 1.4 us | 10.8x [10.8 – 10.9] |
| 4,096 | 472.2 ns | 553.0 ns | 1.17x [1.17 – 1.17] | 1.9 us | 3.99x [3.99 – 4.00] |
| 16,384 | 2.0 us | 1.9 us | 0.96x [0.96 – 0.96] | 3.8 us | 1.92x [1.92 – 1.92] |
| 65,536 | 7.5 us | 8.7 us | 1.17x [1.17 – 1.17] | 11.4 us | 1.52x [1.52 – 1.52] |
| 262,144 | 72.0 us | 64.9 us | 0.90x [0.90 – 0.90] | 64.4 us | 0.90x [0.89 – 0.90] |
| 1,048,576 | 399.2 us | 260.7 us | 0.65x [0.65 – 0.65] | 305.2 us | 0.76x [0.76 – 0.77] |
| 4,194,304 | 1.69 ms | 2.20 ms | 1.30x [1.30 – 1.31] | 3.73 ms | 2.21x [2.20 – 2.22] |
| 16,777,216 | 6.81 ms | 8.99 ms | 1.32x [1.32 – 1.33] | 14.93 ms | 2.19x [2.19 – 2.20] |
| overall |  |  | 1.08x [1.08 – 1.08] |  | 6.70x [6.69 – 6.70] |

`rounded, scalars=zero_lo`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 1.7 ns | 2.1 ns | 1.23x [1.22 – 1.23] | 1.1 us | 625x [624 – 626] |
| 64 | 9.2 ns | 10.3 ns | 1.12x [1.12 – 1.12] | 1.1 us | 120x [120 – 120] |
| 256 | 31.4 ns | 34.3 ns | 1.09x [1.09 – 1.09] | 1.2 us | 38.3x [38.3 – 38.4] |
| 1,024 | 119.1 ns | 131.5 ns | 1.10x [1.10 – 1.10] | 1.4 us | 11.6x [11.5 – 11.6] |
| 4,096 | 473.5 ns | 567.1 ns | 1.20x [1.20 – 1.20] | 1.9 us | 3.98x [3.98 – 3.99] |
| 16,384 | 2.0 us | 1.9 us | 0.96x [0.96 – 0.96] | 3.8 us | 1.93x [1.93 – 1.93] |
| 65,536 | 7.5 us | 8.7 us | 1.17x [1.16 – 1.17] | 11.4 us | 1.52x [1.52 – 1.52] |
| 262,144 | 71.9 us | 64.9 us | 0.90x [0.90 – 0.90] | 64.4 us | 0.90x [0.89 – 0.90] |
| 1,048,576 | 396.6 us | 260.8 us | 0.66x [0.66 – 0.66] | 289.3 us | 0.73x [0.73 – 0.73] |
| 4,194,304 | 1.71 ms | 2.21 ms | 1.29x [1.28 – 1.31] | 3.74 ms | 2.19x [2.18 – 2.20] |
| 16,777,216 | 6.80 ms | 9.01 ms | 1.32x [1.32 – 1.33] | 14.90 ms | 2.19x [2.18 – 2.20] |
| overall |  |  | 1.08x [1.08 – 1.08] |  | 6.70x [6.69 – 6.70] |

All distributions: 1.08x [1.08 – 1.08] over naive C, 6.81x [6.81 – 6.81] over NumPy

With the output buffer 16 bytes behind its input ("apart" is the same kernel with the output clear of the input):

`random`:

| n | kernel | apart | speedup [95%] | naive C | speedup [95%] |
|---|---|---|---|---|---|
| 4,096 | 478.9 ns | 468.4 ns | 0.98x [0.98 – 0.98] | 564.2 ns | 1.18x [1.18 – 1.18] |
| 16,384 | 1.9 us | 2.0 us | 1.03x [1.03 – 1.03] | 2.2 us | 1.17x [1.17 – 1.17] |
| 65,536 | 7.7 us | 7.7 us | 1.00x [1.00 – 1.00] | 7.7 us | 1.00x [1.00 – 1.00] |
| 262,144 | 59.6 us | 98.0 us | 1.64x [1.64 – 1.65] | 111.9 us | 1.88x [1.88 – 1.88] |
| 1,048,576 | 313.5 us | 531.7 us | 1.70x [1.68 – 1.71] | 395.3 us | 1.26x [1.25 – 1.27] |
| 4,194,304 | 1.46 ms | 1.72 ms | 1.18x [1.17 – 1.19] | 2.14 ms | 1.46x [1.45 – 1.47] |
| 16,777,216 | 6.59 ms | 6.80 ms | 1.03x [1.03 – 1.04] | 9.33 ms | 1.42x [1.41 – 1.42] |
| overall |  |  | 1.19x [1.19 – 1.19] |  | 1.31x [1.31 – 1.32] |

**Verdict for a person to confirm.** 1.08x over the naive C is below the 10% the goal template asks of an assembly kernel. What the assembly has that the C does not is in the `clip_f32` log: no stall with the output just behind the input, cache-bypassing stores from 24 MB, and the same code as the `float32` kernel. Removing `x86_64/kernel.s` makes this a C kernel on x86-64; the reference C has none of the ARM64 caveats either.
