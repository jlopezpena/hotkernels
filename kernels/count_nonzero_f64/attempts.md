# count_nonzero_f64 attempts

All runs on an Apple M1 Pro with `--unlocked` (the spec was not on `main` yet), while other
agents were timing other kernels on the same machine, so absolute timings are noisier than usual.

What the compiler does with `naive.c` (`clang -O3 -mcpu=native`, Apple clang 21): eight elements
per step, four 16-byte loads, then per vector `fcmeq #0.0`, `mvn` and `sub` into four accumulators
of 64-bit lanes, and a scalar loop for up to seven leftover elements. That is three vector
instructions per vector on four vector ports: 0.75 cycles per 16 bytes, measured at every size
from 1,024 to 1,048,576 elements, so even the sizes that live in the L2 cache (which delivers
16 bytes in 0.59 cycles) are limited by the loop and not by the cache. Only from 4,194,304
elements is it limited by memory (0.87).

## 1. Baseline: `cmtst` against a 63-bit mask and `sub`, one vector per step, one accumulator

- **Idea:** simplest vector version. A double is zero exactly when its low 63 bits are zero, and every NaN has some of them set, so test the bits: `cmtst` against 0x7fffffffffffffff gives -1 per non-zero lane, subtracted from one accumulator of 64-bit lanes. Odd last element in a general register.
- **check:** pass
- **compare:** n/a (first version). `run`: 0.45x [0.45 – 0.45] vs naive C overall; 0.62x at 16, 0.38–0.45x from 1,024 up.
- **Kept:** yes
- **Notes:** starting point only. One accumulator makes every step wait for the previous `sub` (two cycles per vector). Testing bits instead of comparing as floating point also makes the result independent of the caller's floating-point mode (a flush-to-zero mode would make `fcmeq` treat denormals as zero).
