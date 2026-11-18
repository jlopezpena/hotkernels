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
