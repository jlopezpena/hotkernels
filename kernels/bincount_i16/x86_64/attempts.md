# bincount_i16 attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, pinned to one performance core by the judge. The runs are `--unlocked`: `spec.py` and `naive.c` are new in the same pull request and not on `main` yet. Neither was changed after the first line of assembly was written. Other judges were running on other cores of the machine throughout, so sizes from 1,048,576 elements up are noisier than usual.

What the compiler does with `naive.c` (`clang -O3 -march=native`, clang 21.1.8): a call to `memset`, then a loop unrolled four times of a sign-extending load, a compare with `bins`, a branch and an `inc` on memory. On codes uniform in [0, 64) that is 0.20 ns per element from 4,096 to 16,384 elements, a little over one cycle, and 0.23 to 0.25 from 65,536 up. On four distinct values it is 0.37 ns per element at every size: an update of a counter waits for the previous update of the same counter to be stored and read back, and with four values that is every fourth element on average. NumPy takes 0.78 ns per element at 262,144 uniform codes (it converts the int16 to int64 and reads them for their minimum and maximum before it counts).

## 1. Baseline: one element per step

- **Idea:** the reference, by hand. `rep stosq` clears the counters; each step sign-extends one element, compares it with `bins` as an unsigned number, which a negative element fails as a too-large one does, and increments its counter in memory.
- **check:** pass
- **compare:** n/a (first version). `run`: 0.91x [0.90 – 0.92] of the naive C overall, 4.93x over NumPy.
- **Kept:** yes
- **Notes:** slower than the C wherever the counters are not what holds it. Uniform codes, 64 counters: 0.50x at 16 elements and 0.64x at 64 (`rep stosq` of 512 bytes against the C library's `memset`: 16 ns against 8 for the whole call at 16), 0.83x from 4,096 to 16,384 (0.242 ns per element against 0.200; the C's loop is unrolled four times, this one takes a loop branch per element). Four distinct values: 0.97x to 1.00x from 1,024 up, where both wait for the counters. 1,000 counters: 1.00x at 16 elements, where both spend the call clearing 8 KB.

## 2. Four elements per step, counters cleared with 32-byte stores

- **Idea:** the two things attempt 1 lost to the C. The loop is unrolled four times, so the loop branch is taken once per four elements. The counters are cleared two 32-byte stores per step, the last two ending exactly at the end of `counts` and overlapping the ones before; four to seven counters take two overlapping stores, one to three a store each, and from 512 counters `rep stosq` stays.
- **check:** pass
- **compare:** significantly faster, 1.120x [1.107 – 1.133]
- **Kept:** yes
- **Notes:** uniform codes, 64 counters: 2.13x at 16 elements (15.8 ns to 7.1), 1.28x at 4,096 (0.238 ns per element to 0.186), 1.07x to 1.16x from 16,384 up. Four distinct values: 1.94x at 16, 1.00x to 1.02x from 4,096 up, where the wait for the counters is the whole cost. 1,000 counters: 1.03x at 16 elements, `rep stosq` in both. One 32-byte store per step cleared 256 counters in 16 ns for the call against the C's 10 in a micro-benchmark outside the judge; two per step, 9 ns. An 8-byte load with the four codes taken out of the register was not written: a sign-extending load is one instruction already, where `bincount_u8` saved a load per byte.

## 3. Two elements per increment: a table of pairs, for at most 64 counters

- **Idea:** the limit of attempt 2 is one increment per element, and an increment that waits for the previous one of the same counter. With at most 64 counters, count pairs: 64 x 64 counters of 32 bits, 16 KB below the stack pointer (cleared from the top down, so the stack is touched in the order it grows), one increment per two elements, and equal neighbouring elements wait for one another a sixteenth as often with four distinct values. One 8-byte load reads four elements; `test` against `0xFFC0` in every element says whether all four are in 0..63, and a shift by 10 and an `or` put the two pair indexes in the register. Four that are not all pair codes go one at a time into `counts` with the full test, so does the tail, and a pair code at or above `bins` is dropped when the row and column totals are added to `counts`. From 8,192 elements, and below 2^32, where no 32-bit counter can overflow.
- **check:** pass. Also outside the judge: 24 sizes around every path change, 22 counter counts from 1 to 65,536 and six data shapes, three of them with elements outside the range, with `x` and `counts` each ending at an inaccessible page: 3,168 runs, none wrong. The condition n < 2^32 was not exercised; above it the kernel is attempt 2.
- **compare:** significantly faster, 1.189x [1.183 – 1.194]
- **Kept:** yes
- **Notes:** uniform codes, 64 counters: 1.33x at 16,384, 1.47x to 1.52x from 65,536 to 4,194,304 (0.152 ns per element against 0.22), 1.34x at 16,777,216. Four distinct values: 2.26x at 16,384, 2.41x to 2.48x from 65,536 to 4,194,304 (0.151 ns per element against 0.375). Nothing changes below 8,192 elements or with 1,000 counters (1.01x). The table costs about 300 ns to clear and add up (a micro-benchmark outside the judge, the path forced at 1,024 elements), which uniform codes pay back near 6,000 elements and four values near 2,000; the judge times no size between 4,096 and 16,384, so the threshold is the micro-benchmark's. **What it costs on data the judge does not time:** when most groups of four hold an element outside 0..63, each takes the failed test and then the one-at-a-time path: 0.81 ns per element against the C's 0.57 at 262,147 elements drawn from [-8, 300) with 30 counters (0.70x). A caller that only sends elements inside `[0, bins)` never sees this.

## 4. Sixteen elements per step: the pair indexes from `vpmaddwd`

- **Idea:** attempt 3 takes 3.1 cycles for four elements and two increments, more than the stores need, so spend fewer instructions on the indexes: one 32-byte load, `vptest` against `0xFFC0` in every element, `vpmaddwd` with (1, 64) for the eight pair indexes, taken out through `vmovq`, `vpextrq` and a shift each.
- **check:** pass
- **compare:** significantly slower, 0.981x [0.978 – 0.983]
- **Kept:** no
- **Notes:** uniform codes 1.13x to 1.19x from 16,384 to 1,048,576 (0.127 ns per element at 1,048,576 against 0.152); four distinct values 0.73x to 0.77x at the same sizes (0.212 against 0.152). Eight increments whose addresses arrive together run into one another when two of them are the same counter, which with sixteen possible pairs is often. Two variants, in a micro-benchmark outside the judge only. **The indexes stored to the stack and loaded back** as eight 32-bit loads: 0.26 to 0.31 ns per element on uniform codes, half the speed of attempt 3, whether the store was one of 32 bytes or two of 16. **Two pair tables taking alternate pairs** (32 KB), so that neighbouring pairs never share a counter: uniform 0.144 ns per element at 262,144, four values 0.160, both between attempt 3 and this one, and one repeated value 0.35 against attempt 3's 0.62; not taken further, its only clear gain being on data the judge does not time.

## 5. A second table for up to 1,024 counters

- **Idea:** above 64 counters the pair table is too large, and four distinct values still cost 0.375 ns per element there. Count every other element in a second table of `bins` counters below the stack pointer and add it to `counts` at the end, as `bincount_u8` does with four: an element that repeats the one before it then updates another counter. Up to 1,024 counters (8 KB of stack) and from 8,192 elements.
- **check:** pass
- **compare:** significantly faster, 1.035x [1.031 – 1.039]
- **Kept:** no (superseded by attempt 6, which was compared with the same `HEAD`)
- **Notes:** four distinct values with 1,000 counters 1.36x to 1.42x from 16,384 up (0.262 ns per element against 0.37), but uniform codes with 1,000 counters 0.90x to 0.95x at the same sizes (0.242 against 0.217): slower than the naive C on data that does not repeat. Two things found in a micro-benchmark outside the judge. **The second table must not sit at the same place in a page as `counts`:** 4,096 uniform codes into 64 counters took 0.33 ns per element with the table where the stack pointer happened to put it and 0.19 with it half a page from `counts`; a load waits for an earlier store whose address has the same low 12 bits. **That is also the likely cost on uniform codes:** the loads of `x` pass the place in the page of a table of counters twice as often with two tables as with one. Not confirmed.

## 6. The second table only where a sample of the data repeats

- **Idea:** attempt 5, taken only when at least 32 of the first 256 elements equal their successor (16 compares of 32 bytes, `vpmovmskb`, `popcnt`). Uniform codes over 64 values have about 4 such elements in 256 and four distinct values about 64.
- **check:** pass. Outside the judge, the 3,168 runs of attempt 3 again, none wrong.
- **compare:** significantly faster, 1.047x [1.045 – 1.049]
- **Kept:** yes
- **Notes:** four distinct values with 1,000 counters 1.36x to 1.42x from 16,384 up, as in attempt 5; uniform codes with 1,000 counters 0.99x to 1.02x; the two tables with 64 counters 1.00x. The sample is the first 256 elements, so data that changes character after them gets the path that suits its start: both paths are right on any data, and the worse choice costs what attempt 5's figures show, a tenth on data that does not repeat and four tenths on data that does. Where the one-in-eight line should be was not measured: only uniform codes over 64 values and over 4 were timed.

## Where it stands

Final `judge run` (unlocked): 1.26x [1.26 – 1.26] over the naive C across the four tables, 6.83x over NumPy. By table: uniform codes 1.23x with 64 counters and 1.04x with 1,000; four distinct values 1.61x with 64 counters and 1.22x with 1,000. `NumPy*` is the call from Python, `np.bincount(x, minlength=bins)`, with its per-call overhead.

`uniform`:

| n | kernel | naive C | speedup [95%] | NumPy* | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 7.8 ns | 8.0 ns | 1.03x [1.02 – 1.03] | 430.8 ns | 55.5x [55.48 – 55.59] |
| 64 | 24.1 ns | 25.4 ns | 1.06x [1.06 – 1.06] | 476.7 ns | 19.8x [19.78 – 19.82] |
| 256 | 53.2 ns | 60.2 ns | 1.13x [1.13 – 1.13] | 636.1 ns | 12.0x [11.92 – 11.98] |
| 1,024 | 198.6 ns | 211.2 ns | 1.06x [1.06 – 1.07] | 1.2 us | 6.05x [6.04 – 6.05] |
| 4,096 | 764.9 ns | 846.2 ns | 1.11x [1.11 – 1.11] | 3.4 us | 4.48x [4.48 – 4.48] |
| 16,384 | 3.0 us | 3.2 us | 1.07x [1.06 – 1.07] | 12.9 us | 4.26x [4.24 – 4.26] |
| 65,536 | 10.0 us | 14.1 us | 1.41x [1.41 – 1.41] | 49.4 us | 4.95x [4.95 – 4.95] |
| 262,144 | 39.3 us | 59.0 us | 1.50x [1.50 – 1.50] | 209.9 us | 5.34x [5.33 – 5.34] |
| 1,048,576 | 158.2 us | 236.4 us | 1.49x [1.49 – 1.50] | 868.6 us | 5.49x [5.48 – 5.50] |
| 4,194,304 | 720.1 us | 1.05 ms | 1.45x [1.44 – 1.47] | 6.89 ms | 9.56x [9.41 – 9.65] |
| 16,777,216 | 3.17 ms | 4.28 ms | 1.35x [1.32 – 1.38] | 35.63 ms | 11.2x [10.86 – 11.43] |
| overall |  |  | 1.23x [1.22 – 1.23] |  | 8.77x [8.74 – 8.79] |

`uniform, scalars=thousand_bins`:

| n | kernel | naive C | speedup [95%] | NumPy* | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 30.0 ns | 30.9 ns | 1.03x [1.03 – 1.03] | 483.5 ns | 16.1x [16.10 – 16.12] |
| 64 | 38.9 ns | 40.0 ns | 1.03x [1.03 – 1.03] | 525.9 ns | 13.5x [13.52 – 13.55] |
| 256 | 75.7 ns | 82.8 ns | 1.09x [1.09 – 1.10] | 680.5 ns | 8.99x [8.97 – 9.01] |
| 1,024 | 222.9 ns | 236.9 ns | 1.06x [1.06 – 1.06] | 1.3 us | 5.61x [5.61 – 5.62] |
| 4,096 | 773.4 ns | 847.3 ns | 1.10x [1.09 – 1.10] | 3.7 us | 4.84x [4.84 – 4.84] |
| 16,384 | 3.1 us | 3.3 us | 1.08x [1.08 – 1.08] | 12.8 us | 4.17x [4.17 – 4.17] |
| 65,536 | 15.8 us | 16.2 us | 1.03x [1.03 – 1.03] | 49.8 us | 3.15x [3.15 – 3.15] |
| 262,144 | 58.0 us | 59.5 us | 1.02x [1.02 – 1.03] | 204.0 us | 3.52x [3.51 – 3.53] |
| 1,048,576 | 232.7 us | 236.3 us | 1.02x [1.01 – 1.02] | 843.9 us | 3.63x [3.62 – 3.66] |
| 4,194,304 | 999.4 us | 1.01 ms | 1.01x [1.01 – 1.02] | 6.66 ms | 6.66x [6.60 – 6.78] |
| 16,777,216 | 4.05 ms | 4.07 ms | 1.00x [0.99 – 1.02] n.s. | 27.31 ms | 6.74x [6.66 – 6.84] |
| overall |  |  | 1.04x [1.04 – 1.04] |  | 6.05x [6.05 – 6.07] |

`few_values`:

| n | kernel | naive C | speedup [95%] | NumPy* | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 8.0 ns | 8.4 ns | 1.05x [1.05 – 1.05] | 428.0 ns | 53.8x [53.76 – 53.82] |
| 64 | 23.6 ns | 26.2 ns | 1.11x [1.11 – 1.11] | 474.4 ns | 20.1x [20.11 – 20.13] |
| 256 | 93.5 ns | 96.0 ns | 1.03x [1.02 – 1.03] | 656.7 ns | 7.02x [7.01 – 7.03] |
| 1,024 | 373.9 ns | 378.5 ns | 1.01x [1.01 – 1.01] | 1.4 us | 3.76x [3.76 – 3.77] |
| 4,096 | 1.5 us | 1.5 us | 1.01x [1.01 – 1.01] | 4.0 us | 2.72x [2.72 – 2.72] |
| 16,384 | 2.8 us | 6.1 us | 2.14x [2.13 – 2.14] | 14.9 us | 5.27x [5.26 – 5.27] |
| 65,536 | 10.3 us | 24.5 us | 2.39x [2.38 – 2.39] | 57.6 us | 5.61x [5.60 – 5.62] |
| 262,144 | 39.9 us | 98.2 us | 2.46x [2.46 – 2.46] | 240.9 us | 6.04x [6.03 – 6.04] |
| 1,048,576 | 159.4 us | 394.9 us | 2.48x [2.47 – 2.48] | 984.2 us | 6.18x [6.15 – 6.20] |
| 4,194,304 | 721.8 us | 1.60 ms | 2.22x [2.20 – 2.23] | 6.96 ms | 9.64x [9.56 – 9.77] |
| 16,777,216 | 2.89 ms | 6.49 ms | 2.25x [2.22 – 2.27] | 28.66 ms | 9.93x [9.86 – 10.00] |
| overall |  |  | 1.61x [1.61 – 1.61] |  | 7.97x [7.96 – 7.98] |

`few_values, scalars=thousand_bins`:

| n | kernel | naive C | speedup [95%] | NumPy* | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 30.0 ns | 30.6 ns | 1.02x [1.02 – 1.02] | 481.9 ns | 16.0x [16.04 – 16.05] |
| 64 | 44.0 ns | 44.1 ns | 1.00x [1.00 – 1.00] | 519.9 ns | 11.8x [11.81 – 11.83] |
| 256 | 112.7 ns | 117.2 ns | 1.04x [1.04 – 1.04] | 713.7 ns | 6.33x [6.32 – 6.34] |
| 1,024 | 395.4 ns | 401.7 ns | 1.02x [1.01 – 1.02] | 1.4 us | 3.58x [3.56 – 3.58] |
| 4,096 | 1.5 us | 1.5 us | 1.01x [1.01 – 1.01] | 4.1 us | 2.75x [2.75 – 2.75] |
| 16,384 | 4.2 us | 6.1 us | 1.46x [1.46 – 1.46] | 14.9 us | 3.58x [3.57 – 3.58] |
| 65,536 | 17.3 us | 24.9 us | 1.43x [1.43 – 1.44] | 57.9 us | 3.34x [3.32 – 3.34] |
| 262,144 | 68.1 us | 97.8 us | 1.44x [1.44 – 1.44] | 236.9 us | 3.48x [3.47 – 3.49] |
| 1,048,576 | 274.8 us | 393.4 us | 1.43x [1.43 – 1.43] | 975.1 us | 3.55x [3.54 – 3.56] |
| 4,194,304 | 1.15 ms | 1.61 ms | 1.40x [1.39 – 1.41] | 6.95 ms | 6.06x [6.00 – 6.14] |
| 16,777,216 | 4.63 ms | 6.43 ms | 1.39x [1.38 – 1.39] | 28.67 ms | 6.19x [6.15 – 6.21] |
| overall |  |  | 1.22x [1.22 – 1.22] |  | 5.13x [5.13 – 5.14] |

Output buffer 16 bytes behind its input, counted within a memory page:


`uniform`:

| n | kernel | apart* | speedup [95%] | naive C | speedup [95%] |
|---|---|---|---|---|---|
| 4,096 | 746.5 ns | 749.8 ns | 1.00x [1.00 – 1.00] | 822.7 ns | 1.10x [1.10 – 1.10] |
| 16,384 | 2.9 us | 2.7 us | 0.93x [0.93 – 0.93] | 4.7 us | 1.60x [1.60 – 1.60] |
| 65,536 | 10.0 us | 10.3 us | 1.03x [1.03 – 1.03] | 15.6 us | 1.57x [1.57 – 1.57] |
| 262,144 | 39.3 us | 40.1 us | 1.02x [1.02 – 1.02] | 57.4 us | 1.46x [1.46 – 1.46] |
| 1,048,576 | 159.1 us | 158.6 us | 1.00x [1.00 – 1.00] | 235.0 us | 1.48x [1.47 – 1.48] |
| 4,194,304 | 637.3 us | 648.8 us | 1.02x [1.01 – 1.03] | 950.9 us | 1.49x [1.48 – 1.50] |
| 16,777,216 | 2.86 ms | 2.90 ms | 1.01x [1.00 – 1.02] | 4.07 ms | 1.42x [1.41 – 1.43] |
| overall |  |  | 1.00x [1.00 – 1.00] n.s. |  | 1.44x [1.43 – 1.44] |

**What the figures are.** Below 8,192 elements, and with more than 1,024 counters, the kernel is the naive C's loop with a cheaper start: 1.00x to 1.13x. The gains are the two paths from 8,192 elements. With at most 64 counters the pair table takes 0.151 ns per element on both kinds of data where the C takes 0.225 on uniform codes and 0.375 on four values: 1.35x to 1.50x from 65,536 up and 2.14x to 2.48x from 16,384 up. With 65 to 1,024 counters the second table is taken only on data that repeats: 1.39x to 1.46x on four values, and uniform codes stay on the single table at 1.00x to 1.08x. The naive C is not steady from run to run at single sizes (16,384 uniform codes: 3.2 us here, 4.7 us in the second table of the same run), as the log of `bincount_u8` found; the kernel's 3.0 us at 16,384 in the first table is 2.5 to 2.8 us in every `compare`.

**The output 16 bytes behind the input** costs nothing: 1.00x overall, n.s., with 0.93x at 16,384 and 1.02x to 1.03x at 65,536 and 262,144.

**Verdict for a person to confirm:** assembly. It ties the C where nothing but the loop is possible, is 1.35x to 2.5x faster from 65,536 elements with at most 64 counters (1.07x and 2.14x at 16,384 in this run), and 1.4x faster on repeating data with up to 1,024.

What a caller should know:

- **Stack.** The pair path uses 16.6 KB below the stack pointer and the second table 12.3 KB, against the 8.2 KB `bincount_u8` uses. The pair table is cleared from the top down, so the stack is touched in the order it grows. The second table is not: it starts up to a page above the stack pointer and only the counters in use are cleared, so the first store can land up to 12 KB below the caller's frame, as `bincount_u8`'s lands 8 KB below.
- **Elements outside `[0, bins)`** are ignored on every path, but the pair path is slow on them: 0.70x of the C when most groups of four hold one (attempt 3). A binding that only sends calls NumPy would accept never sends any.
- **n of 2^32 or more** with at most 64 counters takes the single table or the second table, whose counters are 64 bits. That branch was not run: it needs 8 GB of input.
- No `pext`, `pdep`, gather or masked store is used. `popcnt` is, in the sample that chooses the second table.

Not tried:

- **The pair table for more than 64 counters.** With 128 it is 64 KB of 32-bit counters, more than the first-level cache.
- **Two pair tables in the kept kernel**, for data that is one value repeated (0.62 ns per element in the pair path, against the C's 1.39; 0.35 with two tables in the micro-benchmark of attempt 4). The judge times no such data.
- **A second table above 1,024 counters, or four tables,** and where the one-in-eight line of the sample belongs.
- **Giving up the pair path** after some number of groups with an element outside the range, which would bound the 0.70x above.
- **`vpgatherdd`** for the counters: an increment is a load, an add and a store, and the gather replaces only the load.
