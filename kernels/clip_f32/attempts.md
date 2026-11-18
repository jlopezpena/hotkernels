# clip_f32 attempts

This log holds two architectures. ARM64 (Apple M1 Pro) comes first. x86-64 (Intel Core i5-14600KF) follows under [x86-64](#x86-64), with its own numbering.

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

## Ten-run means, locked

`uv run judge run clip_f32`, locked, ten times, one run at a time, on a machine that was
not fully quiet (one-minute load average between 2.2 and 9.1 when a run started,
about 3 for most; a run the judge flagged as noisy was repeated). Check: pass,
53,489 cases. Mean of the ten, with the lowest and highest run in brackets. These
replace the indicative and single-pass figures above wherever they are quoted.

| Case | vs naive C, all sizes | at 1,048,576 | vs NumPy, all sizes | at 1,048,576 |
|---|---|---|---|---|
| bounds (-0.5, 0.5), random | 1.38x [1.33 – 1.41] | 1.44x [1.11 – 1.52] | 28.2x [27.2 – 28.9] | 11.0x [8.47 – 11.7] |
| bounds (0, 1), random | 1.23x [1.21 – 1.24] | 1.31x [1.23 – 1.33] | 25.0x [24.1 – 25.3] | 10.0x [9.44 – 10.2] |
| bounds (0, 1), sparse -0.0 | 1.01x [1.00 – 1.02] | 1.00x [1.00 – 1.01] | 20.7x [20.4 – 20.8] | 7.62x [7.56 – 7.73] |
| bounds (0, 1), rounded | 1.01x [1.01 – 1.01] | 1.00x [0.99 – 1.01] | 20.6x [20.5 – 20.7] | 7.67x [7.59 – 7.72] |
| all six tables | 1.22x [1.21 – 1.23] | | 24.9x [24.7 – 25.1] | |

# x86-64

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, locked, pinned to one performance core by the judge. `clip_f32` and `clip_f64` are generated from one source and were written together; this log has the reasoning.

**The opponent and the baseline.** NumPy has an AVX2 loop for `clip` here: 0.096 ns per `float32` at 65,536. The compiler vectorises `naive.c` (`clang -O3 -march=native`), but not into the two instructions the design expected: it emits `vminps`, a `vcmpltps` and a `vblendvps` per vector, and `vblendvps` runs once per cycle on this processor. 0.044 ns per element at 1,024, 0.069 at 65,536.

## x86-64 1. `vmaxps` then `vminps` with the bound first; aligned stores; loads a block ahead

- **Idea:** the design's candidate. `vmaxps a, b` computes `a > b ? a : b`: it returns its second operand when the two are unordered and when they are equal, zeros of either sign included. With the bound first and the data second, `t = vmaxps lo, v` is the reference's `v < lo ? lo : v` and `out = vminps hi, t` is its `t > hi ? hi : t`, bit for bit, for every pair of bounds with `lo <= hi`: a NaN element stays, and a zero element keeps its sign against a zero bound. **One path, whatever the bounds and the data.** Around those two instructions: the stores are aligned to 32 bytes (an element can be clamped twice without harm, so alignment costs one overlapping vector and the leftover is a block that ends with the last element); the loads run a block of four vectors ahead of the stores, so that a store just behind the input does not hold up the loads after it; and from 24 MB of output the blocks are stored with `vmovntps`.
- **check:** pass, first time, both kernels (53,489 cases each: the spec's twelve pairs of bounds, with zeros of both signs as bounds and as data). Outside the judge: 3,936 runs per kernel against the reference expression, with the same twelve pairs of bounds, data that is one tenth each of +0.0, -0.0, NaN, both infinities and the bounds themselves, 36 sizes up to 6,300,001 so that the cache-bypassing stores are reached, nine alignments of `out`, and with `out` = `x`. No mismatch.
- **compare:** not run (first version). `run`: 1.25x over naive C and 12.7x over NumPy overall.
- **Kept:** yes
- **Notes:** 0.025 ns per element at 1,024, eight elements per cycle, which is what two ports for `vmaxps` and `vminps` allow. The six tables of the judge (three kinds of data, two pairs of bounds) agree to within a few percent: 1.22x to 1.32x over the naive C. The ARM64 kernel has to time them separately because its speed depends on them.

## x86-64: where it stands

Final `judge run`:

`random`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 1.7 ns | 3.5 ns | 2.02x [2.00 – 2.06] | 1.1 us | 608x [602 – 614] |
| 64 | 3.0 ns | 3.3 ns | 1.10x [1.10 – 1.11] | 1.1 us | 369x [368 – 369] |
| 256 | 7.0 ns | 11.1 ns | 1.58x [1.58 – 1.59] | 1.1 us | 160x [160 – 161] |
| 1,024 | 25.5 ns | 45.3 ns | 1.78x [1.78 – 1.78] | 1.3 us | 50.5x [50.5 – 50.6] |
| 4,096 | 100.3 ns | 174.6 ns | 1.74x [1.74 – 1.74] | 1.6 us | 15.6x [15.6 – 15.6] |
| 16,384 | 948.9 ns | 1.1 us | 1.20x [1.20 – 1.20] | 2.5 us | 2.66x [2.66 – 2.67] |
| 65,536 | 3.8 us | 4.5 us | 1.20x [1.20 – 1.20] | 6.3 us | 1.69x [1.69 – 1.69] |
| 262,144 | 21.2 us | 21.0 us | 0.99x [0.99 – 1.00] | 25.5 us | 1.20x [1.20 – 1.20] |
| 1,048,576 | 120.6 us | 121.0 us | 1.00x [1.00 – 1.00] | 126.6 us | 1.05x [1.05 – 1.05] |
| 4,194,304 | 961.2 us | 981.8 us | 1.02x [1.02 – 1.04] | 1.10 ms | 1.14x [1.13 – 1.15] |
| 16,777,216 | 3.83 ms | 4.95 ms | 1.29x [1.28 – 1.30] | 7.66 ms | 2.00x [1.98 – 2.02] |
| overall |  |  | 1.32x [1.32 – 1.32] |  | 11.3x [11.2 – 11.3] |

`random, scalars=zero_lo`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 1.9 ns | 3.6 ns | 1.89x [1.84 – 1.90] | 1.1 us | 567x [566 – 567] |
| 64 | 2.8 ns | 3.3 ns | 1.17x [1.16 – 1.17] | 1.1 us | 388x [387 – 390] |
| 256 | 7.0 ns | 11.1 ns | 1.58x [1.58 – 1.59] | 1.1 us | 161x [160 – 161] |
| 1,024 | 25.5 ns | 45.2 ns | 1.77x [1.77 – 1.77] | 1.3 us | 50.5x [50.4 – 50.5] |
| 4,096 | 100.5 ns | 175.3 ns | 1.74x [1.74 – 1.75] | 1.6 us | 15.7x [15.7 – 15.7] |
| 16,384 | 948.3 ns | 1.1 us | 1.20x [1.20 – 1.20] | 2.5 us | 2.67x [2.66 – 2.67] |
| 65,536 | 3.8 us | 4.5 us | 1.20x [1.20 – 1.20] | 6.4 us | 1.69x [1.69 – 1.69] |
| 262,144 | 41.4 us | 28.5 us | 0.69x [0.69 – 0.69] | 320.3 us | 7.74x [7.73 – 7.75] |
| 1,048,576 | 185.5 us | 131.8 us | 0.71x [0.71 – 0.71] | 705.5 us | 3.80x [3.80 – 3.81] |
| 4,194,304 | 950.7 us | 973.4 us | 1.02x [1.02 – 1.03] | 1.08 ms | 1.14x [1.13 – 1.15] |
| 16,777,216 | 3.92 ms | 5.10 ms | 1.30x [1.29 – 1.31] | 7.51 ms | 1.92x [1.90 – 1.94] |
| overall |  |  | 1.23x [1.23 – 1.24] |  | 14.9x [14.9 – 14.9] |

`sparse_negative_zeros`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 1.8 ns | 3.5 ns | 2.01x [1.97 – 2.04] | 1.1 us | 605x [599 – 610] |
| 64 | 2.8 ns | 3.3 ns | 1.17x [1.17 – 1.17] | 1.1 us | 393x [392 – 394] |
| 256 | 7.0 ns | 11.1 ns | 1.58x [1.58 – 1.59] | 1.1 us | 160x [160 – 160] |
| 1,024 | 25.5 ns | 45.5 ns | 1.78x [1.78 – 1.79] | 1.3 us | 50.3x [50.3 – 50.4] |
| 4,096 | 100.1 ns | 174.5 ns | 1.74x [1.74 – 1.74] | 1.6 us | 15.7x [15.6 – 15.7] |
| 16,384 | 948.9 ns | 1.1 us | 1.21x [1.21 – 1.21] | 2.5 us | 2.66x [2.66 – 2.66] |
| 65,536 | 3.8 us | 4.5 us | 1.20x [1.20 – 1.20] | 6.3 us | 1.69x [1.68 – 1.69] |
| 262,144 | 41.3 us | 28.5 us | 0.69x [0.69 – 0.69] | 320.3 us | 7.75x [7.74 – 7.75] |
| 1,048,576 | 195.7 us | 129.8 us | 0.66x [0.66 – 0.66] | 1.04 ms | 5.29x [5.28 – 5.29] |
| 4,194,304 | 947.2 us | 962.6 us | 1.02x [1.01 – 1.02] | 1.05 ms | 1.11x [1.10 – 1.12] |
| 16,777,216 | 3.91 ms | 5.01 ms | 1.28x [1.28 – 1.29] | 7.51 ms | 1.92x [1.92 – 1.93] |
| overall |  |  | 1.23x [1.23 – 1.23] |  | 15.4x [15.4 – 15.4] |

`sparse_negative_zeros, scalars=zero_lo`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 1.8 ns | 3.6 ns | 2.02x [2.00 – 2.05] | 1.1 us | 600x [593 – 607] |
| 64 | 2.9 ns | 3.5 ns | 1.18x [1.18 – 1.18] | 1.1 us | 373x [373 – 374] |
| 256 | 7.0 ns | 11.1 ns | 1.58x [1.58 – 1.59] | 1.1 us | 161x [161 – 161] |
| 1,024 | 25.6 ns | 45.6 ns | 1.78x [1.78 – 1.79] | 1.3 us | 50.3x [50.3 – 50.4] |
| 4,096 | 100.6 ns | 184.2 ns | 1.83x [1.83 – 1.83] | 1.6 us | 15.6x [15.6 – 15.7] |
| 16,384 | 949.3 ns | 1.1 us | 1.20x [1.20 – 1.20] | 2.5 us | 2.66x [2.66 – 2.67] |
| 65,536 | 3.8 us | 4.5 us | 1.20x [1.20 – 1.20] | 6.3 us | 1.69x [1.69 – 1.69] |
| 262,144 | 41.3 us | 28.5 us | 0.69x [0.69 – 0.69] | 320.3 us | 7.75x [7.74 – 7.75] |
| 1,048,576 | 195.3 us | 129.7 us | 0.66x [0.66 – 0.66] | 1.25 ms | 6.40x [6.39 – 6.40] |
| 4,194,304 | 950.2 us | 979.9 us | 1.03x [1.03 – 1.04] | 1.05 ms | 1.11x [1.10 – 1.12] |
| 16,777,216 | 3.98 ms | 5.18 ms | 1.30x [1.29 – 1.31] | 7.53 ms | 1.89x [1.88 – 1.91] |
| overall |  |  | 1.24x [1.24 – 1.25] |  | 15.6x [15.6 – 15.6] |

`rounded`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 1.8 ns | 3.5 ns | 1.97x [1.94 – 2.00] | 1.1 us | 599x [595 – 604] |
| 64 | 3.1 ns | 3.3 ns | 1.06x [1.05 – 1.06] | 1.1 us | 349x [349 – 350] |
| 256 | 7.0 ns | 11.1 ns | 1.58x [1.58 – 1.59] | 1.1 us | 160x [160 – 160] |
| 1,024 | 25.5 ns | 45.6 ns | 1.78x [1.78 – 1.79] | 1.3 us | 50.5x [50.4 – 50.5] |
| 4,096 | 100.7 ns | 175.3 ns | 1.74x [1.74 – 1.74] | 1.6 us | 15.7x [15.6 – 15.7] |
| 16,384 | 948.2 ns | 1.1 us | 1.20x [1.20 – 1.20] | 2.5 us | 2.67x [2.67 – 2.67] |
| 65,536 | 3.8 us | 4.5 us | 1.20x [1.20 – 1.20] | 6.4 us | 1.69x [1.69 – 1.69] |
| 262,144 | 41.1 us | 26.5 us | 0.64x [0.64 – 0.64] | 23.8 us | 0.58x [0.58 – 0.58] |
| 1,048,576 | 197.0 us | 128.7 us | 0.65x [0.65 – 0.65] | 124.4 us | 0.63x [0.63 – 0.63] |
| 4,194,304 | 965.5 us | 996.5 us | 1.03x [1.03 – 1.04] | 1.06 ms | 1.10x [1.10 – 1.11] |
| 16,777,216 | 3.46 ms | 4.58 ms | 1.32x [1.32 – 1.33] | 7.47 ms | 2.16x [2.14 – 2.17] |
| overall |  |  | 1.22x [1.21 – 1.22] |  | 10.0x [10.0 – 10.0] |

`rounded, scalars=zero_lo`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 1.8 ns | 3.6 ns | 2.02x [1.99 – 2.04] | 1.1 us | 602x [597 – 606] |
| 64 | 2.7 ns | 3.3 ns | 1.21x [1.21 – 1.21] | 1.1 us | 408x [407 – 409] |
| 256 | 7.0 ns | 11.1 ns | 1.58x [1.58 – 1.58] | 1.1 us | 162x [162 – 162] |
| 1,024 | 25.7 ns | 45.8 ns | 1.78x [1.78 – 1.79] | 1.3 us | 50.1x [50.0 – 50.1] |
| 4,096 | 101.0 ns | 175.2 ns | 1.73x [1.73 – 1.74] | 1.6 us | 15.6x [15.5 – 15.6] |
| 16,384 | 948.4 ns | 1.1 us | 1.21x [1.21 – 1.21] | 2.5 us | 2.68x [2.68 – 2.68] |
| 65,536 | 3.8 us | 4.5 us | 1.20x [1.20 – 1.20] | 6.3 us | 1.69x [1.69 – 1.69] |
| 262,144 | 41.2 us | 26.5 us | 0.64x [0.64 – 0.64] | 23.9 us | 0.58x [0.58 – 0.58] |
| 1,048,576 | 197.1 us | 128.7 us | 0.65x [0.65 – 0.65] | 124.3 us | 0.63x [0.63 – 0.63] |
| 4,194,304 | 962.4 us | 1.01 ms | 1.04x [1.04 – 1.06] | 1.05 ms | 1.09x [1.09 – 1.10] |
| 16,777,216 | 3.50 ms | 4.60 ms | 1.31x [1.31 – 1.32] | 7.59 ms | 2.17x [2.15 – 2.18] |
| overall |  |  | 1.23x [1.23 – 1.23] |  | 10.2x [10.2 – 10.2] |

All distributions: 1.25x [1.24 – 1.25] over naive C, 12.7x [12.7 – 12.7] over NumPy

With the output buffer 16 bytes behind its input ("apart" is the same kernel with the output clear of the input):

`random`:

| n | kernel | apart | speedup [95%] | naive C | speedup [95%] |
|---|---|---|---|---|---|
| 4,096 | 100.0 ns | 100.0 ns | 1.00x [1.00 – 1.00] | 175.1 ns | 1.75x [1.75 – 1.75] |
| 16,384 | 960.4 ns | 957.8 ns | 1.00x [1.00 – 1.00] | 1.1 us | 1.11x [1.11 – 1.11] |
| 65,536 | 3.8 us | 3.7 us | 0.98x [0.98 – 0.98] | 4.2 us | 1.11x [1.11 – 1.11] |
| 262,144 | 15.9 us | 46.8 us | 2.94x [2.93 – 2.95] | 51.8 us | 3.25x [3.25 – 3.26] |
| 1,048,576 | 126.4 us | 181.8 us | 1.44x [1.44 – 1.44] | 201.5 us | 1.59x [1.59 – 1.60] |
| 4,194,304 | 985.9 us | 1.21 ms | 1.23x [1.22 – 1.24] | 1.02 ms | 1.04x [1.04 – 1.05] |
| 16,777,216 | 3.25 ms | 3.79 ms | 1.17x [1.16 – 1.18] | 4.69 ms | 1.44x [1.44 – 1.45] |
| overall |  |  | 1.29x [1.29 – 1.29] |  | 1.50x [1.49 – 1.50] |

No stall. The survey found NumPy 8x slower in this placement and the naive C 1.5x; with the loads a block ahead, the kernel is not slower in it at any size.

What holds it:

- **Up to 1,024:** two instructions per vector on two ports: eight elements per cycle.
- **4,096 to 4,194,304:** the caches and memory. The kernel moves the same bytes as the naive C and NumPy, and is level with both at 1,048,576.
- **16,777,216:** the cache-bypassing stores: 1.3x over the naive C, 2.0x over NumPy.

The candidates the design lists for this family (`docs/x86-64-design.md`):

| Candidate | Outcome |
|---|---|
| `vmaxps` and `vminps` with the bound first and the data second compute the reference bit for bit for every ordered pair of bounds | Kept, and confirmed by the judge's twelve pairs of bounds and by the test above |
| All three `clip` caveats of the ARM64 kernel would not exist on x86-64 | **They do not exist in this kernel; whether the documents and the binding say so is a person's decision.** See below |
| The compiler very likely emits the same two instructions, so the kernel would ship as C | It does not: it emits a minimum, a compare and a blend. The assembly is 1.25x over it (`float32`) and 1.08x (`float64`) |
| Non-temporal stores above the third-level cache | Kept |
| An output just behind the input: a path that does not stall, perhaps the only place a kernel wins | No stall, by loading ahead; and it is not the only place it wins |
| Alignment of data | Stores aligned; loads as they come |
| Masked loads for leftover elements | Not needed: the leftover is a block that overlaps the one before |
| Instructions slow on other x86-64 processors | None used |

**For a person to decide: the caller's constraints on x86-64.** The table "What a caller must know" has, for the ARM64 `clip` kernels: `x` and `out` must not overlap; reads and writes FPSR; with `lo = +0.0`, data with a negative zero in its first few thousand elements runs at the speed of the C. None of the three is true of this kernel:

- **No path depends on the bounds or the data,** and negative zeros cost nothing.
- **No status register is read.** One flag is still written: `vmaxps` and `vminps` set the invalid flag in MXCSR whenever an operand is a NaN, quiet or signalling, on every path and at every size. So the binding's clearing of flags after `clip` is needed on x86-64 as on ARM64, and for every NaN, where the ARM64 kernel sets the flag for a quiet NaN only on its compare-and-select path.
- **`out` = `x` gives the right result,** because every element is computed from its own input and clamping twice changes nothing. Tested outside the judge at every size above. A partial overlap is still wrong, as for any vector kernel.

The kernel is not allowed to remove a caveat on its own authority, and nothing in the binding is changed here. The binding sends in-place `clip` to NumPy today (issue 7); on x86-64 it would not have to.

Left untried: a prefetch; the cache-bypassing stores between 2 MB and 24 MB of output (see `cumsum_i64`, x86-64 attempt 5).

Found here and worth knowing on ARM64: nothing that carries over; the difference is in what the two instruction sets' maximum and minimum do with zeros and NaN.
