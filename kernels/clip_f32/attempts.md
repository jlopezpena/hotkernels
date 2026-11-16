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
- **Notes:** 262,144: 1.19x, 1,048,576: 1.29x, 4,194,304: 1.54x, 16,777,216: 1.09x; nothing changes up to 16,384. Locally the loop now matches `memmove` at every size from 65,536 up. Tried locally in the same harness and not worth a judge run, because none of them moved the timings: `stnp` instead of `stp` (no change at any size, with or without `ldnp`), `prfm` on the input 512 or 1,024 bytes ahead (no change), `prfm pstl1keep` on the output (9% slower in the second-level cache, 2 to 3 times slower beyond it), eight vectors per iteration (no change). `dc zva` on the output is far worse: 20 to 90 times slower when issued next to the stores, and still 1.4 times slower when a whole 64 KB of output is zeroed first and then filled.
