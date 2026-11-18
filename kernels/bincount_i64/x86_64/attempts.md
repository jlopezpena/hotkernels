# bincount_i64 attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, pinned to one performance core by the judge. The runs are `--unlocked`: `spec.py` and `naive.c` are new in the same pull request and not on `main` yet. Neither was changed after the first line of assembly was written. Other judges were running on other cores of the machine at times, so sizes from 1,048,576 elements up are noisier than usual.

This kernel was written straight after `bincount_i16`, whose log beside its `kernel.s` has the reasoning and the dead ends; the attempts here are that kernel's kept ones, redone for 8-byte elements and judged again.

What the compiler does with `naive.c` (`clang -O3 -march=native`, clang 21.1.8): `memset`, then a loop unrolled four times of a load, a compare with `bins`, a branch and an `inc` on memory. On codes uniform in [0, 64): 0.20 ns per element at 4,096, 0.27 to 0.29 from 65,536 to 1,048,576, 0.58 at 16,777,216, where the 128 MB of input come from memory. On four distinct values 0.37 to 0.38 up to 1,048,576: an update of a counter waits for the previous update of the same counter. NumPy takes 0.64 ns per element at 262,144 uniform codes; with int64 it has nothing to convert, but it still reads the data for its minimum and maximum first.

## 1. Baseline: four elements per step, counters cleared with 32-byte stores

- **Idea:** `bincount_i16` as it stood after its attempt 2, with an 8-byte load in place of the sign-extending one. Each element is compared with `bins` as an unsigned number, which a negative element fails as a too-large one does, and its counter is incremented in memory; the loop is unrolled four times. The counters are cleared two 32-byte stores per step, the last two ending exactly at the end of `counts`; from 512 counters with `rep stosq`.
- **check:** pass
- **compare:** n/a (first version). `run`: 1.02x [1.01 – 1.02] over the naive C overall, 3.43x over NumPy.
- **Kept:** yes
- **Notes:** the same loop as the C's at the same speed: 1.07x at 4,096 uniform codes, 1.01x to 1.03x from 65,536 to 1,048,576, 1.00x on four distinct values at every size from 4,096. 0.96x at 16 elements with 64 counters (9.9 ns against 9.5).

## 2. Two elements per increment: a table of pairs, for at most 64 counters

- **Idea:** attempt 3 of `bincount_i16`. With at most 64 counters, from 8,192 elements and below 2^32, count pairs in 64 x 64 counters of 32 bits, 16 KB below the stack pointer, cleared from the top down: one increment per two elements, and equal neighbours wait for one another far less often. With 8-byte elements nothing is gained by one wide load, so the four elements of a step are or'ed together from memory and compared with 64 once (the or is below 64 as an unsigned number only if each of them is, so a negative element and one with any high bit set both fail), and each pair index is a load, a shift by 6 and an `or` from memory. Four that are not all pair codes go one at a time into `counts` with the full test against `bins`, so does the tail, and a pair code at or above `bins` is dropped when the row and column totals are added to `counts`.
- **check:** pass. Also outside the judge: 24 sizes around every path change, 22 counter counts from 1 to 65,536 and six data shapes, three of them with elements outside the range, among them codes with 2^32 added, codes shifted up by 32 bits and codes with the sign bit set, with `x` and `counts` each ending at an inaccessible page: 3,168 runs, none wrong. The condition n < 2^32 was not exercised.
- **compare:** significantly faster, 1.085x [1.072 – 1.098]
- **Kept:** yes
- **Notes:** uniform codes, 64 counters: 1.27x to 1.29x from 16,384 to 1,048,576 (0.21 ns per element against 0.27), 1.10x to 1.11x at 4,194,304 and 16,777,216, where reading 8 bytes per element from memory is most of the time. Four distinct values: 1.70x to 1.86x up to 1,048,576 (0.205 ns per element against 0.38), 1.30x to 1.34x above. Less than `bincount_i16` gets from the same table (0.151 ns per element there): a step is 21 instructions' worth of work for four elements here, and from 262,144 elements the input no longer fits the second-level cache. The table with uniform codes and 1,000 counters reads 0.97x [0.95 – 0.98], 0.93x at 16,384, on a path whose instructions did not change; `bincount_i16` read 1.01x there after the same change, and the `compare` of attempt 3, where that path gains only a sample of 256 elements, has the table at 1.01x [1.00 – 1.01]. Taken as noise in where the unchanged loop landed, not confirmed. On data with elements outside 0..63 in most groups of four the pair path is slower than the C, as in `bincount_i16`: 0.93 ns per element against 0.59 at 262,147 elements drawn from [-8, 300) with 30 counters, in a micro-benchmark outside the judge.

## 3. A second table where a sample of the data repeats, up to 1,024 counters

- **Idea:** attempt 6 of `bincount_i16`. With 65 to 1,024 counters and from 8,192 elements, the first 256 elements are compared with their successors (`vpcmpeqq`, `vmovmskpd`, `popcnt`); if at least 32 are equal, every other element is counted in a second table of `bins` counters below the stack pointer, placed half a page from `counts` within a page and added to `counts` at the end. The unsampled form, which `bincount_i16` found slower on data that does not repeat (its attempt 5), was not tried again here.
- **check:** pass. Outside the judge, the 3,168 runs of attempt 2 again, none wrong.
- **compare:** significantly faster, 1.034x [1.031 – 1.036]
- **Kept:** yes
- **Notes:** four distinct values with 1,000 counters 1.33x to 1.36x from 16,384 to 1,048,576 (0.277 ns per element against 0.375), 1.15x to 1.16x above, where the input comes from memory. Uniform codes with 1,000 counters 1.00x to 1.03x. The two tables with 64 counters, which this change does not reach from 8,192 elements, read 0.99x [0.99 – 1.00] and 1.00x n.s.

## Where it stands

Final `judge run` (unlocked): 1.15x [1.15 – 1.15] over the naive C across the four tables, 3.96x over NumPy. By table: uniform codes 1.12x with 64 counters and 1.02x with 1,000; four distinct values 1.34x with 64 counters and 1.16x with 1,000. `NumPy*` is the call from Python, `np.bincount(x, minlength=bins)`, with its per-call overhead.

`uniform`:

| n | kernel | naive C | speedup [95%] | NumPy* | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 8.5 ns | 8.6 ns | 1.01x [1.01 – 1.01] | 285.4 ns | 33.4x [33.40 – 33.45] |
| 64 | 19.0 ns | 19.6 ns | 1.03x [1.03 – 1.03] | 321.4 ns | 16.9x [16.86 – 16.91] |
| 256 | 54.9 ns | 61.1 ns | 1.11x [1.11 – 1.11] | 438.1 ns | 7.97x [7.96 – 7.99] |
| 1,024 | 267.4 ns | 228.4 ns | 0.85x [0.85 – 0.86] | 871.8 ns | 3.26x [3.26 – 3.26] |
| 4,096 | 780.2 ns | 817.9 ns | 1.05x [1.05 – 1.05] | 2.6 us | 3.39x [3.38 – 3.39] |
| 16,384 | 3.6 us | 4.2 us | 1.18x [1.18 – 1.18] | 10.7 us | 2.98x [2.98 – 2.98] |
| 65,536 | 13.8 us | 16.7 us | 1.22x [1.21 – 1.22] | 41.3 us | 3.00x [3.00 – 3.00] |
| 262,144 | 56.0 us | 74.9 us | 1.34x [1.34 – 1.34] | 167.9 us | 3.00x [2.99 – 3.00] |
| 1,048,576 | 218.3 us | 291.4 us | 1.33x [1.33 – 1.34] | 665.5 us | 3.05x [3.04 – 3.05] |
| 4,194,304 | 1.50 ms | 1.72 ms | 1.14x [1.13 – 1.16] | 3.63 ms | 2.41x [2.39 – 2.44] |
| 16,777,216 | 6.53 ms | 7.34 ms | 1.12x [1.10 – 1.16] | 15.61 ms | 2.39x [2.35 – 2.41] |
| overall |  |  | 1.12x [1.12 – 1.12] |  | 4.68x [4.67 – 4.68] |

`uniform, scalars=thousand_bins`:

| n | kernel | naive C | speedup [95%] | NumPy* | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 29.9 ns | 30.7 ns | 1.03x [1.03 – 1.03] | 340.0 ns | 11.4x [11.38 – 11.39] |
| 64 | 38.8 ns | 39.3 ns | 1.01x [1.01 – 1.02] | 369.2 ns | 9.52x [9.50 – 9.53] |
| 256 | 75.2 ns | 84.3 ns | 1.12x [1.12 – 1.12] | 487.8 ns | 6.49x [6.44 – 6.52] |
| 1,024 | 309.0 ns | 308.8 ns | 1.00x [1.00 – 1.00] n.s. | 924.1 ns | 2.99x [2.98 – 2.99] |
| 4,096 | 1.2 us | 1.2 us | 1.07x [1.07 – 1.08] | 2.8 us | 2.44x [2.44 – 2.45] |
| 16,384 | 4.2 us | 4.2 us | 1.01x [1.01 – 1.01] | 10.7 us | 2.57x [2.57 – 2.58] |
| 65,536 | 17.7 us | 17.2 us | 0.97x [0.97 – 0.97] | 41.9 us | 2.37x [2.36 – 2.37] |
| 262,144 | 71.7 us | 71.6 us | 1.00x [1.00 – 1.00] n.s. | 168.3 us | 2.35x [2.34 – 2.35] |
| 1,048,576 | 290.8 us | 301.1 us | 1.04x [1.03 – 1.04] | 674.5 us | 2.32x [2.31 – 2.32] |
| 4,194,304 | 1.79 ms | 1.80 ms | 1.01x [1.00 – 1.02] n.s. | 3.74 ms | 2.09x [2.07 – 2.12] |
| 16,777,216 | 7.50 ms | 7.42 ms | 0.99x [0.97 – 1.02] n.s. | 15.60 ms | 2.08x [2.03 – 2.12] |
| overall |  |  | 1.02x [1.02 – 1.02] |  | 3.42x [3.41 – 3.42] |

`few_values`:

| n | kernel | naive C | speedup [95%] | NumPy* | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 9.2 ns | 9.6 ns | 1.05x [1.05 – 1.06] | 278.2 ns | 30.4x [30.34 – 30.38] |
| 64 | 24.2 ns | 25.5 ns | 1.05x [1.05 – 1.05] | 316.4 ns | 13.1x [13.08 – 13.09] |
| 256 | 90.2 ns | 94.5 ns | 1.05x [1.05 – 1.05] | 460.0 ns | 5.10x [5.10 – 5.10] |
| 1,024 | 385.8 ns | 382.1 ns | 0.99x [0.99 – 0.99] | 1.1 us | 2.73x [2.73 – 2.73] |
| 4,096 | 1.5 us | 1.5 us | 1.01x [1.01 – 1.01] | 3.4 us | 2.24x [2.24 – 2.24] |
| 16,384 | 3.7 us | 6.3 us | 1.71x [1.71 – 1.72] | 12.8 us | 3.49x [3.48 – 3.49] |
| 65,536 | 13.3 us | 24.7 us | 1.86x [1.85 – 1.86] | 50.0 us | 3.76x [3.76 – 3.77] |
| 262,144 | 53.5 us | 98.8 us | 1.85x [1.83 – 1.86] | 200.0 us | 3.74x [3.71 – 3.76] |
| 1,048,576 | 212.1 us | 397.3 us | 1.87x [1.86 – 1.88] | 795.9 us | 3.75x [3.73 – 3.76] |
| 4,194,304 | 1.49 ms | 2.07 ms | 1.39x [1.36 – 1.41] | 4.02 ms | 2.70x [2.65 – 2.74] |
| 16,777,216 | 6.37 ms | 8.73 ms | 1.37x [1.35 – 1.40] | 16.85 ms | 2.64x [2.62 – 2.66] |
| overall |  |  | 1.34x [1.33 – 1.34] |  | 4.52x [4.51 – 4.53] |

`few_values, scalars=thousand_bins`:

| n | kernel | naive C | speedup [95%] | NumPy* | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 29.4 ns | 30.5 ns | 1.04x [1.04 – 1.04] | 340.5 ns | 11.6x [11.56 – 11.57] |
| 64 | 44.3 ns | 44.2 ns | 1.00x [1.00 – 1.00] | 370.6 ns | 8.37x [8.37 – 8.38] |
| 256 | 113.4 ns | 117.0 ns | 1.03x [1.03 – 1.04] | 515.6 ns | 4.55x [4.55 – 4.55] |
| 1,024 | 445.6 ns | 446.6 ns | 1.00x [1.00 – 1.00] | 1.1 us | 2.49x [2.48 – 2.49] |
| 4,096 | 1.6 us | 1.6 us | 1.00x [1.00 – 1.00] | 3.4 us | 2.17x [2.16 – 2.17] |
| 16,384 | 4.5 us | 6.1 us | 1.35x [1.35 – 1.35] | 12.8 us | 2.83x [2.83 – 2.84] |
| 65,536 | 18.4 us | 25.1 us | 1.36x [1.36 – 1.36] | 50.1 us | 2.72x [2.72 – 2.72] |
| 262,144 | 73.3 us | 99.9 us | 1.36x [1.36 – 1.36] | 200.8 us | 2.74x [2.74 – 2.74] |
| 1,048,576 | 290.7 us | 397.8 us | 1.37x [1.37 – 1.37] | 798.0 us | 2.74x [2.74 – 2.75] |
| 4,194,304 | 2.04 ms | 2.36 ms | 1.15x [1.07 – 1.20] | 4.35 ms | 2.13x [2.00 – 2.24] |
| 16,777,216 | 7.22 ms | 8.57 ms | 1.19x [1.12 – 1.20] | 17.03 ms | 2.36x [2.21 – 2.38] |
| overall |  |  | 1.16x [1.15 – 1.16] |  | 3.39x [3.37 – 3.41] |

Output buffer 16 bytes behind its input, counted within a memory page:


`uniform`:

| n | kernel | apart* | speedup [95%] | naive C | speedup [95%] |
|---|---|---|---|---|---|
| 4,096 | 808.2 ns | 781.6 ns | 0.97x [0.97 – 0.97] | 824.2 ns | 1.02x [1.02 – 1.02] |
| 16,384 | 3.6 us | 3.6 us | 0.99x [0.99 – 0.99] | 4.0 us | 1.11x [1.10 – 1.11] |
| 65,536 | 13.9 us | 13.8 us | 0.99x [0.99 – 0.99] | 18.8 us | 1.35x [1.35 – 1.35] |
| 262,144 | 57.5 us | 58.2 us | 1.01x [1.01 – 1.02] | 73.2 us | 1.27x [1.27 – 1.28] |
| 1,048,576 | 234.8 us | 237.6 us | 1.01x [1.00 – 1.02] | 295.5 us | 1.26x [1.25 – 1.26] |
| 4,194,304 | 1.61 ms | 1.67 ms | 1.03x [1.02 – 1.05] | 1.82 ms | 1.13x [1.12 – 1.14] |
| 16,777,216 | 9.44 ms | 9.36 ms | 0.99x [0.93 – 1.25] n.s. | 9.93 ms | 1.05x [0.96 – 1.21] n.s. |
| overall |  |  | 1.00x [0.99 – 1.03] n.s. |  | 1.16x [1.15 – 1.19] |

**What the figures are.** Below 8,192 elements, and with more than 1,024 counters, the kernel is the naive C's loop. From 8,192 elements with at most 64 counters the pair table takes 0.21 ns per element on both kinds of data, where the C takes 0.26 to 0.29 on uniform codes and 0.38 on four values: 1.18x to 1.34x and 1.71x to 1.87x up to 1,048,576. With 65 to 1,024 counters the second table is taken only on data that repeats: 1.35x to 1.37x on four values up to 1,048,576, and uniform codes stay on the single table (0.97x to 1.04x). From 4,194,304 elements every gain shrinks to 1.12x to 1.39x: the input is 32 MB and more, and reading it is most of the time (0.36 to 0.39 ns per element in the pair path).

**1,024 uniform codes with 64 counters reads 0.85x** in the table above (267 ns against 228). It is not the kernel's: two `run --quick` afterwards had 291 ns against 305 (1.05x) and 375 against 378 (1.01x). Both the kernel and the C run the same loop there, and its time moves between 228 and 378 ns from one run to the next with where the judge's buffers fall, as the log of `bincount_u8` found for that kernel's naive C. Both also take 0.29 to 0.30 ns per element at 1,024 and 4,096 elements with 1,000 counters, against 0.19 with 64 counters at 4,096; that was not looked into.

**The output 16 bytes behind the input** costs nothing: 1.00x overall, n.s., between 0.97x and 1.03x at every size.

**Verdict for a person to confirm:** assembly, by less than `bincount_i16`. It ties the C where nothing but the loop is possible, is 1.2x to 1.9x faster from 8,192 to 1,048,576 elements with at most 64 counters and 1.35x on repeating data with up to 1,024, and 1.1x to 1.4x beyond, where memory sets the pace.

What a caller should know (as for `bincount_i16`):

- **Stack.** The pair path uses 16.6 KB below the stack pointer, cleared from the top down, and the second table 12.3 KB, whose first store can land up to 12 KB below the caller's frame.
- **Elements outside `[0, bins)`** are ignored on every path, but the pair path is slow on them (attempt 2). A binding that only sends calls NumPy would accept never sends any.
- **n of 2^32 or more** with at most 64 counters takes the single table or the second table, whose counters are 64 bits. That branch was not run: it needs 32 GB of input.
- No `pext`, `pdep`, gather or masked store is used. `popcnt` is, in the sample that chooses the second table.

Not tried:

- **The dead ends of `bincount_i16` on this element type**: the pair indexes computed in a vector register (its attempt 4), two pair tables, and the second table without the sample (its attempt 5). An attempt rejected for 2-byte elements was rejected for those; none was repeated here.
- **Fewer instructions in the pair step**, for instance `vptest` on a 32-byte load for the range test in place of four `or` from memory.
- **Prefetching the input** from 4,194,304 elements, where it comes from memory.
- **The pair table for more than 64 counters**, a second table above 1,024 counters, and where the one-in-eight line of the sample belongs.
