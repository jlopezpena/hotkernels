# nonzero_bool attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, locked, pinned to one performance core by the judge.

What the compiler does with `naive.c` (`clang -O3 -march=native`): the scalar loop, one store and one add of the mask byte per element, about 0.19 ns per element. NumPy takes 0.42 ns per element at 1,048,576 with half selected and 0.69 with 5%.

## 1. Baseline: positions from a 256-entry table, widened with `vpmovzxbq`

- **Idea:** one `vpcmpeqb` and one `vpmovmskb` turn 32 mask bytes into 32 bits. Each byte of the word indexes a table of 256 entries of eight bytes (2 KB): the positions of its set bits in ascending order. Two `vpmovzxbq` widen an entry to eight 64-bit lanes, the index of the group's first element is added, both registers are stored at the write position and `popcnt` advances it. The write position is an element count, so the chain from group to group is one `add` (see `compress_f32`, x86-64 attempt 3). A mask byte is true when it is not zero.
- **check:** pass
- **compare:** not run (first version). `run`: 2.00x over naive C overall (half_selected 1.85x, few_selected 2.16x), 9.17x over NumPy.
- **Kept:** yes
- **Notes:** about 2.8 cycles per eight mask bytes inside the first-level cache, more than the eleven instructions need. Each group makes two 32-byte stores to the same advancing place, 64 bytes for at most eight indices, and the stores are what hold it. A table of 64-byte entries (16 KB) that needs no widening was not tried: the widening is not the limit.

## 2. Blocks with at most four true elements take `tzcnt` and `blsr`

- **Idea:** a block of 32 whose mask word has at most four bits set skips the table. `tzcnt` and `blsr` take a fixed four positions out of the word, each stored with one 8-byte store. Past the last set bit `tzcnt` returns 32 and the store is an index the write position does not advance over. Blocks with more go through the four table groups. One compare of the block's count chooses.
- **check:** pass
- **compare:** significantly faster, 1.401x [1.382 – 1.421]
- **Kept:** yes
- **Notes:** few_selected 1.97x (1.5x at 64, 2.0x at 1,024, 2.4x to 2.7x from 16,384 to 4,194,304), half_selected no significant difference. Four 8-byte stores per block where the table path makes eight of 32 bytes. **No weak spot between the two densities the judge times.** Micro-benchmark outside the judge, 65,536 elements, random masks, ns per element at 1% / 5% / 10% / 12% / 15% / 20% / 30% / 50% true: 0.037 / 0.035 / 0.052 / 0.049 / 0.060 / 0.082 / 0.104 / 0.113 with this change, 0.076 / 0.081 / 0.084 / 0.086 / 0.087 / 0.095 / 0.106 / 0.114 without. The ARM64 kernel is 9% slower than its dense path at 10%; here the mispredicted branch near the threshold costs less than the stores it saves.

## Where it stands

Final `judge run`:

`half_selected`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 3.3 ns | 4.0 ns | 1.22x [1.22 – 1.22] | 322.5 ns | 98.0x [98.0 – 98.1] |
| 64 | 6.0 ns | 11.3 ns | 1.89x [1.88 – 1.89] | 338.1 ns | 56.3x [56.2 – 56.4] |
| 256 | 19.2 ns | 46.8 ns | 2.44x [2.44 – 2.45] | 456.0 ns | 23.8x [23.7 – 23.9] |
| 1,024 | 83.5 ns | 273.3 ns | 3.27x [3.27 – 3.29] | 856.6 ns | 10.3x [10.2 – 10.3] |
| 4,096 | 301.2 ns | 775.5 ns | 2.57x [2.57 – 2.58] | 2.1 us | 6.90x [6.90 – 6.91] |
| 16,384 | 1.8 us | 3.1 us | 1.69x [1.69 – 1.70] | 7.9 us | 4.35x [4.34 – 4.35] |
| 65,536 | 7.4 us | 13.1 us | 1.77x [1.77 – 1.77] | 27.3 us | 3.67x [3.67 – 3.67] |
| 262,144 | 30.4 us | 56.3 us | 1.85x [1.85 – 1.85] | 109.4 us | 3.60x [3.59 – 3.60] |
| 1,048,576 | 138.9 us | 218.8 us | 1.58x [1.57 – 1.58] | 438.4 us | 3.16x [3.15 – 3.16] |
| 4,194,304 | 701.7 us | 901.6 us | 1.28x [1.22 – 1.31] | 1.79 ms | 2.55x [2.41 – 2.60] |
| 16,777,216 | 3.99 ms | 4.58 ms | 1.15x [1.14 – 1.15] | 11.38 ms | 2.85x [2.83 – 2.86] |
| overall |  |  | 1.79x [1.78 – 1.80] |  | 8.26x [8.22 – 8.28] |

`few_selected`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 3.1 ns | 3.2 ns | 1.01x [1.00 – 1.01] | 319.0 ns | 102x [102 – 102] |
| 64 | 3.1 ns | 10.9 ns | 3.50x [3.50 – 3.50] | 328.3 ns | 105x [105 – 105] |
| 256 | 9.7 ns | 46.7 ns | 4.79x [4.79 – 4.79] | 364.0 ns | 37.3x [37.3 – 37.4] |
| 1,024 | 32.9 ns | 194.2 ns | 5.91x [5.90 – 5.91] | 543.1 ns | 16.5x [16.5 – 16.5] |
| 4,096 | 137.9 ns | 775.6 ns | 5.62x [5.62 – 5.64] | 1.1 us | 7.96x [7.93 – 8.03] |
| 16,384 | 533.8 ns | 3.1 us | 5.81x [5.81 – 5.81] | 3.0 us | 5.70x [5.70 – 5.71] |
| 65,536 | 2.1 us | 12.4 us | 5.83x [5.83 – 5.83] | 23.4 us | 11.0x [10.9 – 11.0] |
| 262,144 | 8.5 us | 50.8 us | 5.96x [5.96 – 5.97] | 170.0 us | 20.0x [19.9 – 20.0] |
| 1,048,576 | 37.0 us | 202.6 us | 5.47x [5.46 – 5.48] | 724.8 us | 19.6x [19.6 – 19.6] |
| 4,194,304 | 175.2 us | 815.0 us | 4.65x [4.56 – 4.71] | 2.90 ms | 16.6x [16.2 – 16.8] |
| 16,777,216 | 776.9 us | 3.28 ms | 4.22x [4.15 – 4.28] | 11.85 ms | 15.2x [15.0 – 15.4] |
| overall |  |  | 4.41x [4.40 – 4.41] |  | 20.9x [20.8 – 20.9] |

All distributions: 2.81x [2.80 – 2.81] over naive C, 13.1x [13.1 – 13.1] over NumPy

With the output buffer 16 bytes behind its input (`half_selected`; "apart" is the same kernel with the output clear of the input):

`half_selected`:

| n | kernel | apart | speedup [95%] | naive C | speedup [95%] |
|---|---|---|---|---|---|
| 4,096 | 324.7 ns | 261.1 ns | 0.80x [0.80 – 0.81] | 775.2 ns | 2.39x [2.38 – 2.39] |
| 16,384 | 1.9 us | 1.9 us | 1.00x [1.00 – 1.00] | 3.5 us | 1.86x [1.86 – 1.86] |
| 65,536 | 7.4 us | 7.5 us | 1.01x [1.01 – 1.01] | 13.3 us | 1.81x [1.81 – 1.81] |
| 262,144 | 29.8 us | 31.2 us | 1.05x [1.05 – 1.05] | 56.2 us | 1.89x [1.88 – 1.89] |
| 1,048,576 | 138.3 us | 138.4 us | 1.00x [1.00 – 1.00] n.s. | 229.0 us | 1.66x [1.65 – 1.66] |
| 4,194,304 | 780.8 us | 931.1 us | 1.19x [1.18 – 1.22] | 961.0 us | 1.23x [1.23 – 1.23] |
| 16,777,216 | 3.80 ms | 3.92 ms | 1.03x [1.03 – 1.04] | 4.44 ms | 1.17x [1.16 – 1.17] |
| overall |  |  | 1.01x [1.01 – 1.01] |  | 1.67x [1.67 – 1.67] |

The close placement costs nothing from 16,384 up. At 4,096 it costs a fifth: the output is eight times the size of the mask, so only at the smallest sizes are the stores still near the addresses the mask loads are about to read.

What holds it: with half true, the stores of the table path (two of 32 bytes per eight mask bytes), then from 16,384 the cache lines of an output that is 4 bytes per mask byte, then memory. With few true, the instruction count of the sparse block, 0.035 ns per mask byte up to 1,048,576.

The candidates the design lists for this family (`docs/x86-64-design.md`):

| Candidate | Outcome |
|---|---|
| 32 mask bytes to a 32-bit mask in one `vpmovmskb` | Kept (attempt 1) |
| The index from a table | Kept: positions as bytes, widened with `vpmovzxbq`. `vpermd` does not apply, since the output is indices and not elements of an input |
| The index from `pext` with no table | Not tried: more instructions than one table load, and the table path is held by its stores |
| `popcnt` advances the write pointer | Kept |
| Sparse masks: `tzcnt` and `blsr` per set bit | Kept (attempt 2), as a fixed four per block |
| A block all zeros or all ones recognised with one compare | Not tried as a path of its own. An all-zero block takes the sparse path, which costs about five cycles for it. A branch for it alone would be taken at random on the masks the judge times |
| The switch by density can go | A switch remains, per block and with no state. The micro-benchmark finds no density at which it is slower than the table path alone, so the weak spot near 10% is gone |
| Masked loads and stores for leftover elements | Not tried: open issue 3 of the design. The last 1 to 7 elements take the scalar loop |
| Non-temporal stores | Not tried: every store is overwritten in part by the next |
| An output just behind the input | 0.80x at 4,096, nothing from 16,384 up (table above) |
| Alignment of data | Not tried here; rejected on `compress_f32` (its attempt 4), where the input is four times larger per element |
| Alignment of code | The block loop starts on a 64-byte boundary. Not varied |
| Instructions slow on other x86-64 processors | None used: no `pext`, `pdep`, gather or masked store |

Left untried: a second sparse tier for blocks with five to eight true elements (the judge times 5% and 50%, so `compare` could not see it); narrower stores in the table path when a group has at most four true elements, which is a branch per group.
