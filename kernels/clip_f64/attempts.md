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
