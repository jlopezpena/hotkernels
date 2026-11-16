# histogram_f64 attempts

All runs on an Apple M1 Pro with `--unlocked` (the spec was not on `main` yet), while other
agents were timing other kernels on the same machine, so timings are noisier than usual.

## 1. Baseline: scalar, one element per iteration

- **Idea:** simplest correct version: naive.c translated one operation at a time (separate `fmul` and `fadd` for the edges), without the redundant `edge(b) == hi` select the compiler keeps for the first correction.
- **check:** pass
- **compare:** n/a (first version). `run`: 1.09x [1.09 – 1.09] vs naive C overall; 1.12x from 4,096 up (2.26 ns per element against 2.54), 0.93x at n=16.
- **Kept:** yes
- **Notes:** both versions take 7–8 cycles per element, far more than the instruction count explains.
