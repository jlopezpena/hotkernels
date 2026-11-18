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
