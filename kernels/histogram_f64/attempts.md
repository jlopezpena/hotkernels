# histogram_f64 attempts

All runs on an Apple M1 Pro with `--unlocked` (the spec was not on `main` yet), while other
agents were timing other kernels on the same machine, so timings are noisier than usual.

## 1. Baseline: scalar, one element per iteration

- **Idea:** simplest correct version: naive.c translated one operation at a time (separate `fmul` and `fadd` for the edges), without the redundant `edge(b) == hi` select the compiler keeps for the first correction.
- **check:** pass
- **compare:** n/a (first version). `run`: 1.09x [1.09 – 1.09] vs naive C overall; 1.12x from 4,096 up (2.26 ns per element against 2.54), 0.93x at n=16.
- **Kept:** yes
- **Notes:** both versions take 7–8 cycles per element, far more than the instruction count explains.

## 2. NEON: two elements per vector, both edges computed in the vector, per-lane scalar fallback

- **Idea:** all the per-element arithmetic is independent, so do it two doubles at a time with the same individually rounded operations as naive.c: `fsub`, `fdiv`, `fmul`, `fcvtzs`/`frintz`, then both edges as `fmul` + `fadd`, two `fcmge`, and the masks subtracted from the index. A lane is accepted with one unsigned compare when it lands in bins 0..nbins-2; anything else (NaN, out of range, the last bin, v == hi) is redone by a scalar copy of naive.c. The vector path is only used when `step` is positive and finite and nbins < 2^32; the header comment of kernel.s has the argument for why an accepted lane equals naive.c.
- **check:** pass
- **compare:** significantly faster, 2.460x [2.457 – 2.465]
- **Kept:** yes
- **Notes:** 2.2x at n=16 up to 2.6x at 1,024; 0.88 ns per element against 2.26. Micro-benchmarks explain the scalar version: `fcmp` runs at one per cycle on this core and naive.c needs four per element; all the vector operations run at four per cycle, `fdiv v.2d` at one per cycle, and `fmov`/`umov` to a general register at two per cycle.
