# clip_f64 attempts

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
