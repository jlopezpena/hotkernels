# clip_f32 attempts

## 1. Compare and select, 16 elements per iteration, overlapping tail

- **Idea:** the same four operations per vector that clang emits for `naive.c` (two `fcmgt`, two `bit`), four vectors per iteration, but with no entry checks and no scalar remainder: a leftover of 1 to 15 elements is one more block ending exactly at the end, and 4 to 15 elements are a front piece and an overlapping back piece. Below 4 is a scalar `fcmp`/`fcsel` loop.
- **check:** pass
- **compare:** n/a (first version)
- **Kept:** yes
- **Notes:** starting point. Bit-exact for any bounds by construction: nothing is arithmetic, so NaN elements, NaN bounds, signed zeros and `lo > hi` all follow the C expression.

## 2. `fmax`/`fmin` fast path, guarded so it stays bit-exact

- **Idea:** compare and select needs four vector operations per vector, and that is the limit of the baseline and of clang's loop (16 operations per 16 elements on 4 vector pipes: 4.3 cycles). `fmin(fmax(x, lo), hi)` needs two, which leaves only the stores as the limit (2.6 cycles per 16 elements). It is not the C expression in four situations, so each is excluded: (1) `lo > hi` or a NaN bound, by `fcmp` at entry; (2) `lo` is +0.0 or `hi` is -0.0, the only bounds for which a signed-zero tie changes the result, by a bit test at entry; (3) FPCR flush-to-zero or default-NaN mode, by reading FPCR at entry; (4) a signalling NaN element, which `fmax` would return quieted: `fmax`/`fmin` raise FPSR.IOC for a signalling NaN and for nothing else, so FPSR is read after the loop and, if the flag is set, the array is redone on the exact path from the untouched input. FPSR is also read at entry; a flag the caller left set is cleared and set again afterwards. Anything excluded takes the compare-and-select path of attempt 1.
- **check:** pass
- **compare:** significantly faster, 1.166x [1.163 – 1.169] with the fast path from 192 elements. (First tried from 64 elements: 1.082x [1.080 – 1.085], with n = 64 at 0.48x, because the two FPSR reads cost about as much as 170 elements.)
- **Kept:** yes
- **Notes:** equivalence is proved case by case in the header of `kernel.s`, not assumed. 256: 1.33x, 1,024: 1.52x, 4,096 and 16,384: 1.58x. From 65,536 up nothing changes (1.00 to 1.03x): there both versions run at the same 0.27 cycles per element, so the limit is the caches, not the vector operations. The judge passes one pair of bounds, no signalling NaN and a default FPCR, so paths (1) to (4) were verified outside the judge: a harness comparing against the compiled `naive.c` bit for bit over 19 x 19 bound pairs (NaN, signalling NaN, infinities, both zeros, denormals, `lo > hi`), data with quiet and signalling NaNs, zeros, denormals and values equal and adjacent to the bounds, every n from 0 to 300 and sizes up to 70,000, FPSR clean and with flags already set, seven FPCR modes, buffers against inaccessible pages: 515,916 calls, no difference, no flag lost. Removing any one of the guards makes that harness fail. Thresholds from local timing: the clean path wins from about 170 elements (192 used); with the flag already set, clearing and restoring it costs about 33 ns, so that path is used from 1,536 elements.

## 3. Non-temporal loads (`ldnp`) in the fast loop

- **Idea:** from 65,536 elements up, attempt 2 ran no faster than compare and select. Local timing showed a plain `ldp`/`stp` copy loop with no vector operations at all runs at that same speed (0.26 cycles per element), so the limit there is moving the data, not clipping it; but the system `memmove` is faster at those sizes (0.20). Disassembling it shows it loads with `ldnp`. `ldnp` reads the same bytes as `ldp`; only the cache hint differs.
- **check:** pass
- **compare:** significantly faster, 1.091x [1.085 – 1.098]
- **Kept:** yes
- **Notes:** 262,144: 1.19x, 1,048,576: 1.29x, 4,194,304: 1.54x, 16,777,216: 1.09x; nothing changes up to 16,384. Locally the loop now matches `memmove` at every size from 65,536 up. Tried locally in the same harness and not worth a judge run, because none of them moved the timings: `stnp` instead of `stp` (no change at any size, with or without `ldnp`), `prfm` on the input 512 or 1,024 bytes ahead (no change), `prfm pstl1keep` on the output (10 to 20% slower in the second-level cache, 2 to 4 times slower beyond it), eight vectors per iteration (no change). `dc zva` on the output is far worse: 20 to 90 times slower when issued next to the stores, and still 1.4 times slower when a whole 64 KB of output is zeroed first and then filled.

## 4. Symmetric bounds on the short path: one absolute compare

- **Idea:** below 192 elements the kernel is still compare and select, four operations per vector. When `lo` is exactly `-hi` (as the judge's bounds are), `facgt` finds both sides at once and a clipped element keeps its own sign, so `facgt` + `bic` + `bit` is three operations and still exact for every element, NaN included, in any FPCR mode. Used for 16 elements and up on the exact path when a bit test of the bounds at entry says they are symmetric.
- **check:** pass
- **compare:** no significant difference, 1.007x [1.000 – 1.013]
- **Kept:** no
- **Notes:** n = 64 gained (5.6 ns to 4.8 ns, 1.16x) but n = 16 lost (1.9 ns to 2.0 ns, 0.92x) to the extra test of the bounds at entry. It would also have helped only one shape of bounds. Not tried for the same range: `fmax`/`fmin` plus a vector test for NaN in the results instead of the FPSR reads. That is three operations per vector for any bounds, but any NaN in a short array, quiet ones included, would then cost a second pass, about 1.7 times the plain compare and select.

## 5. The `lo = +0.0` path of `clip_f64`

- **Idea:** the path worked out on `clip_f64` (its attempts 4 and 5), with four floats per vector: `fmin(fmax(x, lo), hi)` plus one `smin.4s` per vector that finds a -0.0 (bit pattern 0x80000000, the smallest signed 32-bit value) in a chunk; a chunk that held one is done again by compare and select; a credit rule gives up early when -0.0 is not rare. The first chunk is 4 blocks (64 elements), later ones 8 to 64 blocks, and the path is not tried below 512 elements.
- **check:** pass
- **compare:** significantly faster, 1.026x [1.023 – 1.029] over the six timed tables (three kinds of data, two pairs of bounds). Bounds (0, 1) on random data: 1.18x [1.17 – 1.19], with 1.19x at 1,024, 1.32x at 4,096, 1.34x at 16,384. Bounds (0, 1) on data with negative zeros: 0.99x [0.99 – 1.00], with 0.95x to 0.97x at 1,024, which is the price of the first chunk. Bounds (-0.5, 0.5): unchanged.
- **Kept:** yes
- **Notes:** speed relative to `naive.c` for bounds (0, 1), measured outside the judge with the machine in use: without -0.0, 1.05x at 512, 1.14x at 1,024, 1.28x at 4,096 and 16,384; with -0.0 at 1% or 19%, 0.86x to 0.89x at 512, 0.95x at 1,024, 0.98x to 1.01x from 4,096. Verified outside the judge with the differential test described under `clip_f64` attempt 5, built for float32: 1.1 billion elements over 500,000 calls, no difference.

## Result

`uv run judge run clip_f32 --unlocked`, `kernel.s` as of commit 3101e7f, Apple M1 Pro, other work running on the machine at the time:

| n | kernel | naive C | speedup [95% interval] |
|---|---|---|---|
| 16 | 1.9 ns | 2.2 ns | 1.17x [1.16 – 1.17] |
| 64 | 5.9 ns | 6.2 ns | 1.04x [1.03 – 1.06] |
| 256 | 16.5 ns | 22.1 ns | 1.34x [1.30 – 1.35] |
| 1,024 | 58.8 ns | 84.1 ns | 1.43x [1.40 – 1.46] |
| 4,096 | 226.4 ns | 336.9 ns | 1.49x [1.43 – 1.51] |
| 16,384 | 829.8 ns | 1.4 us | 1.66x [1.65 – 1.67] |
| 65,536 | 3.5 us | 5.3 us | 1.54x [1.54 – 1.54] |
| 262,144 | 13.9 us | 21.5 us | 1.55x [1.55 – 1.55] |
| 1,048,576 | 68.2 us | 86.0 us | 1.26x [1.26 – 1.29] |
| 4,194,304 | 345.2 us | 536.5 us | 1.55x [1.55 – 1.57] |
| 16,777,216 | 1.45 ms | 1.58 ms | 1.09x [1.08 – 1.12] |
| overall | | | 1.36x [1.35 – 1.37] |

Assembly is kept. Where it stands and what the numbers do not show:

- From 4,096 elements up the fast loop runs at the speed of the system `memmove` for the same number of bytes (locally 0.16 cycles per element in the first-level cache, 0.20 in the second, 0.26 to 0.28 beyond), so there is nothing left for the clipping itself to give. Below 192 elements it is compare and select, as clang's loop is; the gain there (1.17x at 16, 1.04x at 64) is entry code only.
- The speedup is for bounds that qualify for the fast path: `lo <= hi`, `lo` not +0.0, `hi` not -0.0. The judge's bounds (-0.5, 0.5) do. Bounds such as (0, 1) do not, because `fmax(-0.0, +0.0)` is +0.0 where C and NumPy keep -0.0. Until attempt 5 they took the compare-and-select path at the speed of `naive.c`; since then `lo = +0.0` has its own path, which is about 1.3x in cache on data without -0.0 and runs at the speed of `naive.c` on data with it.
- It assumes FPSR.IOC is clear on entry, as it is in the judge. If the caller's thread has it set, which any earlier invalid operation does (0/0, or a vector comparison with a NaN such as `naive.c` or this kernel's own short path make on NaN data), then 192 to 1,535 elements take the compare-and-select path (speed of `naive.c`) and from 1,536 the fast path costs 33 ns more per call (locally 1.28x over `naive.c` at 2,048, 1.45x at 4,096, 1.6x at 16,384).
- An array containing a signalling NaN is processed twice (locally 1.6 to 1.8 times the time of `naive.c`). Quiet NaNs cost nothing extra.
- `x` and `out` must not overlap.
- Verified by the judge: the result for its bounds at every size, memory safety, calling convention. Verified only by the local harness described under attempt 2: every other pair of bounds, signalling NaNs, non-default FPCR, FPSR flags preserved, and the compare-and-select path beyond a few hundred elements.
