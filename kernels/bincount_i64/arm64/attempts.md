# bincount_i64 attempts (arm64)

All runs on an Apple M1 Pro (macOS, native), locked, while other kernel runs were timing on the same machine, so timings from about 100,000 elements up are noisier than usual.

This kernel was written straight after `bincount_i16` on the same machine, whose log beside its `kernel.s` has the reasoning and the dead ends. Each idea kept there was tried again here and judged on its own.

What the naive C costs here (Apple clang, `-O3`): on codes uniform in [0, 64) 0.35 ns per element from 4,096 up; on four distinct values 0.93 ns per element, where an increment waits for the store of the previous increment of the same counter.

## 1. Baseline: clear 64 bytes per step, eight elements per step

- **Idea:** `bincount_i16` as it stood after its attempt 2, with 8-byte elements read by `ldp`: each compared with `bins` as an unsigned number (a negative one is a huge one), a branch, and a load, add and store of its counter. The counters are cleared with `stp q0, q0`, 64 bytes per step, the last step ending exactly at the end of `counts`.
- **check:** pass
- **compare:** n/a (first version). `run`: 1.03x [1.03 – 1.04] over the naive C overall, 4.01x over NumPy.
- **Kept:** yes
- **Notes:** uniform codes 1.10x–1.13x from 1,024 up (0.31 ns per element against 0.35); four values 0.97x from 1,024 up (0.95 ns against 0.93), where both wait for the counters; 0.89x at 16 elements with 64 counters (18.9 ns against 16.8).
