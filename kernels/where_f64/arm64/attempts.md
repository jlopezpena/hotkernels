# where_f64 attempts (arm64)

All runs on an Apple M1 Pro (performance cores), macOS, locked (`spec.py` and `naive.c` as on `main`). Other kernel runs were timing other kernels on the same machine at the same time, so the sizes from about 100,000 elements up are noisier than usual.

Every element is 17 bytes read and 8 written. The naive C is left scalar by clang on this machine (about 0.36 ns per element in the first-level cache), and NumPy is level with it.

## 1. Baseline: blocks of 16, cmtst and tbl to lane masks, bsl, overlapping last block

- **Idea:** sixteen bytes of c are loaded once and made 0xff or 0x00 by `cmtst`; eight `tbl` spread each pair of bytes over the two 64-bit lanes of a vector (the indices from `maskfill_f64`), and `bsl` takes each lane from a or b. Loads and stores in pairs (`ldp`/`stp`), half a block at a time. The elements left after the whole blocks are done by one more block that ends at the end of the buffers (out never overlaps an input, so writing some elements twice is harmless). Under 16 elements, a scalar loop with `csel`. Nothing but loads, stores, `tbl` and `bsl` touches the elements.
- **check:** pass
- **compare:** n/a (first version). `run`: 1.66x [1.66 – 1.67] over naive C overall (half_true 1.66x, mostly_true 1.67x), 4.83x over NumPy.
- **Kept:** yes
- **Notes:** 1.92x at 16, 2.6x to 2.7x from 256 to 4,096 (0.13 ns per element), 1.35x to 1.38x from 16,384 to 262,144 (0.26 ns per element), 1.04x to 1.17x from 1,048,576 up. Close placement: 1.00x of the clear placement at every size but 1,048,576, where the clear timing was the slow one.

## 2. Non-temporal loads (`ldnp`) for a and b

- **Idea:** as in `clip_f64` attempt 3: `ldnp` reads the same bytes as `ldp`, and its hint let that loop run at memory-copy speed beyond the first-level cache.
- **check:** pass
- **compare:** significantly faster, 1.060x [1.036 – 1.086]
- **Kept:** yes
- **Notes:** nothing up to 4,096; 1.14x to 1.15x at 16,384, 1.11x to 1.13x at 262,144, 1.26x at 1,048,576, 1.05x at 4,194,304, 1.02x to 1.04x at 16,777,216.
