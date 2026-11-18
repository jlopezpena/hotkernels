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
