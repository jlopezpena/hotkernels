# compress_f64 attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, locked, pinned to one performance core by the judge. Read with the `compress_f32` log, where the shared scheme was worked out.

What the compiler does with `naive.c` (`clang -O3 -march=native`): the scalar loop, about 0.19 ns per element up to 262,144. NumPy takes 3.4 ns per element at 1,048,576 with half selected.

## 1. Baseline: `vpermd` by a 16-entry table, 32 mask bytes per `vpmovmskb`

- **Idea:** the `compress_f32` kernel as it stood after its attempt 3, four elements per step. AVX2 has no permute of 64-bit lanes by a variable index, so the table (16 entries, 512 bytes) holds pairs of 32-bit lane numbers for `vpermd`. One mask word serves eight steps. The write position is an element count from the start.
- **check:** pass
- **compare:** not run (first version). `run`: 1.50x over naive C overall (half_selected 1.35x, few_selected 1.67x), 12.0x over NumPy.
- **Kept:** yes
- **Notes:** twice the steps of `compress_f32` per mask byte, each moving the same 32 bytes, so the output's cache-line traffic weighs twice as much. With half selected, 1,024 elements already take 0.15 ns each in the judge's timing, which cycles through several inputs and so leaves the first-level cache.

## 2. Blocks with at most four selected elements copied one by one

- **Idea:** a block of 32 whose mask word has at most four bits set skips the table. `tzcnt` and `blsr` take a fixed four positions out of the word, and each element is copied with one 8-byte load and store. Bit 31 is set before every `tzcnt`, so once the set bits are used up the position is 31, inside the block, and the copy lands beyond the write position's advance. Blocks with more go through the eight table steps as before. One compare of the block's count chooses.
- **check:** pass
- **compare:** significantly faster, 1.239x [1.233 – 1.245]
- **Kept:** yes
- **Notes:** few_selected 1.53x (1.60x at 1,024, 1.98x at 65,536, 1.37x at 1,048,576), half_selected no significant difference. The sparse block costs about 30 instruction slots against 63 for eight table steps, and makes four 8-byte stores where the table path makes eight of 32 bytes. **No weak spot between the two densities the judge times.** Micro-benchmark outside the judge, 65,536 elements, random masks, ns per element at 1% / 5% / 10% / 12% / 15% / 20% / 30% / 50% selected: 0.055 / 0.056 / 0.058 / 0.067 / 0.085 / 0.111 / 0.130 / 0.144 with this change, 0.100 / 0.099 / 0.107 / 0.108 / 0.111 / 0.116 / 0.129 / 0.141 without. Near the threshold the branch is mispredicted often, and the kernel is still not slower than the table path alone, because that path is held by its stores and not by the front end.

## Where it stands

Final `judge run`:

`half_selected`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 3.1 ns | 4.2 ns | 1.33x [1.33 – 1.34] | 269.4 ns | 86.4x [86.4 – 86.5] |
| 64 | 10.1 ns | 14.1 ns | 1.39x [1.39 – 1.39] | 332.2 ns | 32.8x [32.8 – 32.8] |
| 256 | 36.0 ns | 50.7 ns | 1.41x [1.41 – 1.41] | 599.1 ns | 16.7x [16.6 – 16.7] |
| 1,024 | 158.4 ns | 222.5 ns | 1.40x [1.40 – 1.41] | 1.7 us | 10.9x [10.9 – 10.9] |
| 4,096 | 540.0 ns | 790.2 ns | 1.46x [1.46 – 1.46] | 5.8 us | 10.8x [10.8 – 10.8] |
| 16,384 | 2.5 us | 4.1 us | 1.66x [1.66 – 1.67] | 44.8 us | 18.0x [17.9 – 18.2] |
| 65,536 | 9.8 us | 14.1 us | 1.44x [1.44 – 1.44] | 209.3 us | 21.4x [21.4 – 21.4] |
| 262,144 | 47.7 us | 56.9 us | 1.19x [1.19 – 1.19] | 877.6 us | 18.4x [18.4 – 18.4] |
| 1,048,576 | 209.9 us | 229.5 us | 1.09x [1.08 – 1.12] | 3.51 ms | 16.7x [16.5 – 17.1] |
| 4,194,304 | 1.55 ms | 1.91 ms | 1.23x [1.22 – 1.24] | 14.19 ms | 9.13x [9.09 – 9.18] |
| 16,777,216 | 6.87 ms | 8.14 ms | 1.18x [1.18 – 1.19] | 60.40 ms | 8.79x [8.78 – 8.81] |
| overall |  |  | 1.34x [1.34 – 1.34] |  | 17.7x [17.7 – 17.7] |

`few_selected`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 3.2 ns | 4.0 ns | 1.22x [1.22 – 1.22] | 218.3 ns | 67.3x [67.2 – 67.3] |
| 64 | 3.8 ns | 12.8 ns | 3.33x [3.32 – 3.33] | 262.7 ns | 68.4x [68.3 – 68.4] |
| 256 | 14.3 ns | 48.8 ns | 3.42x [3.42 – 3.42] | 319.8 ns | 22.4x [22.4 – 22.4] |
| 1,024 | 56.6 ns | 193.0 ns | 3.41x [3.41 – 3.41] | 594.4 ns | 10.5x [10.5 – 10.5] |
| 4,096 | 226.2 ns | 778.4 ns | 3.44x [3.44 – 3.44] | 1.7 us | 7.44x [7.42 – 7.47] |
| 16,384 | 903.7 ns | 3.1 us | 3.47x [3.47 – 3.47] | 5.4 us | 5.97x [5.97 – 5.97] |
| 65,536 | 3.7 us | 12.8 us | 3.43x [3.43 – 3.43] | 45.4 us | 12.2x [12.2 – 12.2] |
| 262,144 | 15.8 us | 50.4 us | 3.18x [3.18 – 3.18] | 212.0 us | 13.4x [13.4 – 13.4] |
| 1,048,576 | 91.9 us | 205.7 us | 2.24x [2.24 – 2.24] | 860.7 us | 9.36x [9.35 – 9.38] |
| 4,194,304 | 938.5 us | 1.68 ms | 1.79x [1.77 – 1.80] | 3.73 ms | 3.97x [3.96 – 4.00] |
| 16,777,216 | 4.32 ms | 6.08 ms | 1.41x [1.40 – 1.41] | 14.83 ms | 3.44x [3.43 – 3.45] |
| overall |  |  | 2.59x [2.58 – 2.59] |  | 12.3x [12.3 – 12.3] |

All distributions: 1.86x [1.86 – 1.86] over naive C, 14.8x [14.8 – 14.8] over NumPy

With the output buffer 16 bytes behind its input (`half_selected`; "apart" is the same kernel with the output clear of the input):

`half_selected`:

| n | kernel | apart | speedup [95%] | naive C | speedup [95%] |
|---|---|---|---|---|---|
| 4,096 | 553.0 ns | 540.7 ns | 0.98x [0.98 – 0.98] | 792.1 ns | 1.43x [1.43 – 1.43] |
| 16,384 | 2.4 us | 2.4 us | 0.99x [0.99 – 0.99] | 3.3 us | 1.35x [1.35 – 1.35] |
| 65,536 | 9.7 us | 9.6 us | 0.99x [0.99 – 0.99] | 13.8 us | 1.42x [1.42 – 1.42] |
| 262,144 | 47.8 us | 47.5 us | 0.99x [0.99 – 0.99] | 56.9 us | 1.19x [1.19 – 1.19] |
| 1,048,576 | 233.6 us | 315.0 us | 1.35x [1.32 – 1.38] | 249.0 us | 1.07x [1.06 – 1.08] |
| 4,194,304 | 1.55 ms | 1.69 ms | 1.09x [1.08 – 1.09] | 1.87 ms | 1.21x [1.20 – 1.21] |
| 16,777,216 | 6.89 ms | 6.76 ms | 0.98x [0.98 – 0.98] | 8.11 ms | 1.18x [1.17 – 1.18] |
| overall |  |  | 1.05x [1.04 – 1.05] |  | 1.26x [1.25 – 1.26] |

No stall, as in `compress_f32`.

What holds it: the instruction count up to 256 elements; from there, with half selected, the cache lines of the output and then memory, where it is 1.1x to 1.2x over the naive C. With few selected it stays 3.2x to 3.5x over the naive C up to 262,144.

The candidates the design lists for this family: as in the `compress_f32` log, with one difference. **Sparse masks, `tzcnt` and `blsr` per set bit: kept here** (attempt 2), as a fixed four per block of 32 chosen by the block's own count. The design asked whether the switch by density and its weak spot near 10% could go. There is still a switch, per block and with no state, and the micro-benchmark above finds no density at which it is slower than the table path alone. No `pext`, `pdep`, gather or masked store is used.

Left untried: a second sparse tier for blocks with five to eight selected elements, which would move the gain up to about 25% selected. The judge times 5% and 50%, so it could not be kept on a `compare` verdict.
