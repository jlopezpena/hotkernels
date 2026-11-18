# clip_f64 attempts

This log holds two architectures. ARM64 (Apple M1 Pro) comes first. x86-64 (Intel Core i5-14600KF) follows under [x86-64](#x86-64), with its own numbering.

The design was worked out on `clip_f32` first; see `kernels/clip_f32/attempts.md` for the experiments behind each step. The entries here are the same steps measured again for float64.

## 1. Compare and select, 8 elements per iteration, overlapping tail

- **Idea:** the same four operations per vector that clang emits for `naive.c` (two `fcmgt`, two `bit`), four vectors per iteration, with no entry checks and no scalar remainder: a leftover of 1 to 7 elements is one more block ending exactly at the end, 2 to 7 elements are a front piece and an overlapping back piece, and a single element is scalar `fcmp`/`fcsel`.
- **check:** pass
- **compare:** n/a (first version)
- **Kept:** yes
- **Notes:** starting point. Bit-exact for any bounds by construction: nothing is arithmetic, so NaN elements, NaN bounds, signed zeros and `lo > hi` all follow the C expression.

## 2. `fmax`/`fmin` fast path, guarded so it stays bit-exact

- **Idea:** as `clip_f32` attempt 2. `fmin(fmax(x, lo), hi)` is two vector operations per vector instead of four. It is not the C expression in four situations, and each is excluded: `lo > hi` or a NaN bound (`fcmp` at entry); `lo` is +0.0 or `hi` is -0.0 (bit test at entry); FPCR flush-to-zero or default-NaN mode (FPCR read at entry); a signalling NaN element, which `fmax` would return quieted (FPSR.IOC read after the loop; if set, the array is redone on the compare-and-select path from the untouched input). FPSR is also read at entry, and a flag the caller left set is cleared and set again afterwards.
- **check:** pass
- **compare:** significantly faster, 1.162x [1.157 – 1.165]
- **Kept:** yes
- **Notes:** 256: 1.43x, 1,024: 1.49x, 4,096: 1.50x, 16,384: 1.64x; nothing changes at 16 and 64 (below the threshold) or from 65,536 up (limited by moving the data). The equivalence argument is in the header of `kernel.s`. The guarded paths are not reachable with the judge's inputs, so they were verified outside it with the same harness as `clip_f32`, built for float64: 515,916 calls against the compiled `naive.c` over 19 x 19 bound pairs, quiet and signalling NaNs, zeros, denormals, FPSR clean and with flags set, seven FPCR modes; no difference, no flag lost, and removing any one guard makes it fail. Thresholds from local timing: the clean path breaks even at 80 elements (96 used); with the flag already set it breaks even at 512 (768 used).

## 3. Non-temporal loads (`ldnp`) in the fast loop

- **Idea:** as `clip_f32` attempt 3. From 65,536 elements up the limit is moving the data, and the system `memmove` moves it faster than an `ldp`/`stp` loop because it loads with `ldnp`. `ldnp` reads the same bytes as `ldp`; only the cache hint differs.
- **check:** pass
- **compare:** significantly faster, 1.100x [1.094 – 1.104]
- **Kept:** yes
- **Notes:** 65,536: 1.16x, 262,144: 1.30x, 1,048,576: 1.37x, 4,194,304: 1.32x, 16,777,216: 1.07x. n = 256 measured 0.97x [0.96 – 0.98] in this run; the other sizes up to 16,384 are unchanged.

## 4. A path for `lo = +0.0` that watches for -0.0 elements

- **Idea:** `np.clip(x, 0, 1)` took the compare-and-select path, because `fmax(-0.0, +0.0)` is +0.0 where C keeps -0.0. With `lo = +0.0` and `hi` not -0.0 that is the only element value `fmin(fmax(x, lo), hi)` gets wrong. So run the two operations and spend a third per vector on finding out whether a -0.0 was there: a signed minimum over the 32-bit halves of the bit patterns (`smin.4s`, four accumulators), since the high half of -0.0 is 0x80000000, the smallest signed value. The array is done in chunks; after each the minimum is tested, and a chunk that held a -0.0 is done again from the input by compare and select. A credit decides whether to carry on: a clean chunk earns its length, a chunk with a -0.0 costs four times its length, and a negative credit sends the rest of the array to compare and select. Chunk length was half the credit, between 8 and 64 blocks.
- **check:** pass
- **compare:** significantly faster, 1.078x [1.073 – 1.083] over both timed pairs of bounds. Bounds (0, 1): 1.16x [1.15 – 1.17], with 1.16x at 256, 1.30x at 1,024, 1.32x at 4,096, 1.33x at 16,384. Bounds (-0.5, 0.5): 1.00x, not significant, as it should be.
- **Kept:** yes
- **Notes:** I looked for a path that is exact in three operations with no fallback and did not find one. The lower clamp at +0.0 has to zero negative values, keep -0.0 and keep negative NaNs; every two-operation form gets one of those wrong (`fmax` loses -0.0, an unsigned compare with the sign bit zeroes negative NaNs, `fcmlt #0` is right but raises IOC for quiet NaNs, which takes away the signalling-NaN check). The exact forms need four operations, which is what compare and select already costs. The second pass uses `fcmgt`, which raises IOC for a quiet NaN too, so FPSR is read before it (set: a signalling NaN went through, start over) and after it (set: the chunk held a NaN, finish by compare and select).

## 5. Treat a chunk with a -0.0 as expensive: probe chunk, new credit rule, no zero path below 256 elements

- **Idea:** timing attempt 4 outside the judge on data that does contain -0.0 showed two problems the judge's random data cannot show. Arrays of 96 to 255 elements with a -0.0 in them ran at 0.65x to 0.84x of `naive.c`, because the first chunk of 64 elements was wasted. And an array with one -0.0 per 1,000 elements ran at 0.78x at 16,384 elements and beyond: redoing a chunk costs about 75 ns, not the 11 ns its 64 elements take, most of it branch misprediction that a repeated benchmark input hides at smaller sizes. So: the first chunk is 4 blocks (32 elements), later chunks are a quarter of the clean run so far (8 to 64 blocks), a chunk with a -0.0 costs three times its length plus 200 blocks of credit, and the path is not tried below 256 elements. Data in which -0.0 is not rare now gives up after the first short chunk.
- **check:** pass
- **compare:** on the judge's timing data as it was (random values only): no significant difference, 1.000x [0.996 – 1.003], with 0.97x at 256 and 0.98x at 1,024 for bounds (0, 1). The spec was then given two more kinds of timing data, with sparse and with dense negative zeros, so that the judge times what this attempt is about. With them: significantly faster, 1.002x [1.000 – 1.003]; the largest change is 1.09x at 256 elements on rounded data with bounds (0, 1).
- **Kept:** yes, on a margin the judge barely sees. The reason to keep it is the worst case, measured outside the judge.
- **Notes:** speed relative to `naive.c` for bounds (0, 1), measured outside the judge with the machine in use (best of 15 runs each):

| -0.0 in the data | 256 | 512 | 1,024 | 4,096 | 16,384 | 262,144 |
|---|---|---|---|---|---|---|
| none | 1.14x | 1.21x | 1.28x | 1.34x | 1.33x | 1.14x |
| one, in the middle | 0.88x | 0.99x | 1.03x | 1.03x | 1.22x | 1.00x |
| one per 1,000 elements | 1.14x | 1.06x | 1.03x | 1.01x | 1.01x | 1.00x |
| 1% | 0.88x | 0.93x | 0.99x | 1.00x | 1.01x | 1.01x |
| 19% (rounded data) | 0.91x | 0.95x | 0.99x | 1.00x | 1.02x | 1.00x |

Below 256 elements the path is not taken and the kernel runs as before. When this was written the judge had no array with a single -0.0 among ordinary values, so nothing in it reached the second pass followed by more fast chunks: a copy of the kernel whose second pass stores nothing passed the check. The judge has since been given the value class `neg_zero_at`, and that copy now fails at `n=2047 scalars=zero_lo x=neg_zero_at@2046` (and the float32 one at `n=4095`). Verification outside the judge: a differential test against the C expression, 1.1 billion elements over 500,000 calls on the final code (and 1.6 billion more on the versions before it), with `lo = +0.0` in 46% of the calls; -0.0 at densities from none to all and at chosen positions; quiet and signalling NaNs of both signs; negative denormals whose high half equals that of -0.0; values whose low half is 0x80000000; FPSR flags set before the call (none may be lost); sizes from 0 to 70,000 at eight alignments; a check that nothing outside the output is written. No difference. Each of five mutations of the new code (no branch on a found -0.0, no FPSR read before the second pass, no flag restore when giving up, one accumulator dropped, giving up from the cursor instead of the start of the chunk) makes it fail.

## Result

`uv run judge run clip_f64 --unlocked`, `kernel.s` as of commit 3101e7f, Apple M1 Pro, other work running on the machine at the time:

| n | kernel | naive C | speedup [95% interval] |
|---|---|---|---|
| 16 | 2.9 ns | 3.2 ns | 1.09x [1.09 – 1.10] |
| 64 | 11.0 ns | 11.0 ns | 1.00x [1.00 – 1.00] n.s. |
| 256 | 28.8 ns | 42.5 ns | 1.47x [1.46 – 1.49] |
| 1,024 | 109.9 ns | 171.3 ns | 1.56x [1.54 – 1.58] |
| 4,096 | 424.2 ns | 676.5 ns | 1.59x [1.54 – 1.62] |
| 16,384 | 1.6 us | 2.7 us | 1.67x [1.66 – 1.67] |
| 65,536 | 6.9 us | 10.8 us | 1.56x [1.55 – 1.57] |
| 262,144 | 27.7 us | 43.3 us | 1.56x [1.56 – 1.57] |
| 1,048,576 | 135.6 us | 176.7 us | 1.30x [1.30 – 1.31] |
| 4,194,304 | 705.5 us | 949.1 us | 1.35x [1.34 – 1.36] |
| 16,777,216 | 2.92 ms | 3.14 ms | 1.08x [1.07 – 1.08] |
| overall | | | 1.36x [1.36 – 1.37] |

Assembly is kept. The same limits apply as for `clip_f32` (see the end of its log), with the thresholds halved: the speedup in this table is for bounds with `lo <= hi`, `lo` not +0.0 and `hi` not -0.0; since attempts 4 and 5, `lo = +0.0` has its own path (about 1.3x in cache on data without -0.0, the speed of `naive.c` on data with it); other bounds run at the speed of `naive.c`; with FPSR.IOC already set on entry, 96 to 767 elements run at the speed of `naive.c` and larger arrays pay 33 ns more per call (locally 1.29x over `naive.c` at 1,024 and 1.57x at 4,096); an array containing a signalling NaN is processed twice; `x` and `out` must not overlap. The judge verifies the result for its own bounds; everything else was verified with the local harness described under attempt 2.

## Ten-run means, locked

`uv run judge run clip_f64`, locked, ten times, one run at a time, on a machine that was
not fully quiet (one-minute load average between 2.2 and 9.1 when a run started,
about 3 for most; a run the judge flagged as noisy was repeated). Check: pass,
53,489 cases. Mean of the ten, with the lowest and highest run in brackets. These
replace the indicative and single-pass figures above wherever they are quoted.

| Case | vs naive C, all sizes | at 1,048,576 | vs NumPy, all sizes | at 1,048,576 |
|---|---|---|---|---|
| bounds (-0.5, 0.5), random | 1.36x [1.31 – 1.38] | 1.35x [1.15 – 1.42] | 14.5x [14.1 – 14.6] | 5.06x [4.26 – 5.37] |
| bounds (0, 1), random | 1.22x [1.20 – 1.23] | 1.30x [1.21 – 1.33] | 13.0x [12.8 – 13.1] | 4.88x [4.50 – 5.03] |
| bounds (0, 1), sparse -0.0 | 1.02x [1.02 – 1.02] | 1.01x [1.00 – 1.01] | 10.8x [10.8 – 10.9] | 3.78x [3.72 – 3.84] |
| bounds (0, 1), rounded | 1.00x [0.99 – 1.00] | 1.01x [1.00 – 1.02] | 10.6x [10.3 – 10.8] | 3.77x [3.68 – 3.82] |
| all six tables | 1.20x [1.19 – 1.21] | | 12.8x [12.6 – 12.9] | |

# x86-64

All runs on an Intel Core i5-14600KF, Linux, locked, pinned to one performance core by the judge. This kernel is generated from the same source as `clip_f32`, whose log (under "x86-64") has the reasoning, the candidates, the test outside the judge (3,936 runs for this kernel too, no mismatch) and the question for a person about the caller's constraints. Here: `vmaxpd` and `vminpd`, four elements per vector.

## x86-64 1. `vmaxpd` then `vminpd` with the bound first; aligned stores; loads a block ahead

- **Idea:** as `clip_f32`, x86-64 attempt 1.
- **check:** pass, first time
- **compare:** not run (first version). `run`: 1.08x over naive C and 6.81x over NumPy overall.
- **Kept:** yes
- **Notes:** a `float64` is twice the bytes per element, so memory is the limit from smaller arrays on, and the kernel moves the same bytes as the naive C. 1.04x to 1.13x over it across the six tables.

## x86-64: where it stands

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
