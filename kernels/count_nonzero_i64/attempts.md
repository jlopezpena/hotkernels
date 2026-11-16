# count_nonzero_i64 attempts

All runs on an Apple M1 Pro with `--unlocked` (the spec was not on `main` yet), while other
agents were timing other kernels on the same machine, so absolute timings are noisier than usual.

What the compiler does with `naive.c` (`clang -O3 -mcpu=native`, Apple clang 21): eight elements
per step, four 16-byte loads, four `cmtst` and four `sub` into four accumulators of 64-bit lanes,
then a scalar loop for up to seven leftover elements. That is two vector instructions per vector
on four vector ports: two vectors, 32 bytes, per cycle (0.51 cycles per 16 bytes measured from 256
to 16,384 elements). From 65,536 elements it runs at what the L2 cache delivers (0.59 cycles per
16 bytes) and from 4,194,304 at what memory delivers (0.86).

## 1. Baseline: `cmtst` and `sub`, one vector per step, one accumulator

- **Idea:** simplest vector version. Load two elements, `cmtst` gives -1 per non-zero lane, subtract from one accumulator of 64-bit lanes. Odd last element in a general register.
- **check:** pass
- **compare:** n/a (first version). `run`: 0.36x [0.36 – 0.36] vs naive C overall; 0.62x at 16, 0.25–0.31x from 1,024 to 1,048,576, 0.43x at the two largest sizes.
- **Kept:** yes
- **Notes:** starting point only. One accumulator makes every step wait for the previous `sub` (two cycles per vector).
