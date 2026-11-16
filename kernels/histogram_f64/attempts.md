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

## 3. Two passes over 512-element chunks (measured privately, never run through `compare`)

- **Idea:** pass 1 computes the index vectors into a stack buffer, pass 2 reads them back as scalars and counts; meant to keep the counting instructions from waiting in the out-of-order window behind the divide, and to replace the vector-to-general-register moves with loads.
- **check:** private micro-benchmark with its own result check against naive.c, which passed
- **compare:** not run. Micro-benchmark, 65,536 normal values: 1.03 ns per element against 0.88 for attempt 2.
- **Kept:** no
- **Notes:** pass 1 alone takes 0.65 ns per element, so the vector arithmetic is the bottleneck, not the window. The fused loop is faster because the counting runs on the integer ports while the vector ports are busy; splitting the passes puts the two costs end to end.

## 4. Move indices to general registers through the stack (measured privately)

- **Idea:** `stp q, q` to the stack then `ldp x, x` instead of `fmov`/`umov`, which run at only two per cycle.
- **check:** private micro-benchmark result check passed
- **compare:** not run. Micro-benchmark: 1.10–1.15 ns per element against 0.88–0.91.
- **Kept:** no
- **Notes:** storing a vector and reloading its halves is slower than the register moves.

## 5. Eight elements per iteration instead of four (measured privately)

- **Idea:** less loop overhead and more independent work visible per iteration.
- **check:** private micro-benchmark result check passed
- **compare:** not run. Micro-benchmark: 0.863–0.867 ns per element against 0.869–0.880 (normal), within run-to-run noise.
- **Kept:** no
- **Notes:** ablation of attempt 2's loop: vector arithmetic alone 0.645 ns per element (14 vector operations per two elements, about 4.2 cycles against a 3.5-cycle port limit), plus the moves to general registers 0.753, plus checks and counting 0.875. The way forward is fewer vector operations per element, not a different loop shape.

## 6. Edge table: compute each edge once per bin, look both edges up per element

- **Idea:** the edges depend only on the bin, so build a stack table of {edge(i), edge(i+1)} pairs once (same `fmul` + `fadd`), move the first estimate b to a general register, load the pair for each lane, `zip` the two pairs into e0/e1 vectors and compare. 9 vector operations per two elements instead of 14, at the price of a second pair of moves to general registers. A -inf entry made out-of-range lanes fail the range check without a branch on the load.
- **check:** pass
- **compare:** not run; the micro-benchmark was clear. 65,536 normal values: 1.11 ns per element against 0.88 for attempt 2.
- **Kept:** no
- **Notes:** ablation: first estimate plus two moves 0.34 ns per element; plus table loads, `zip`, compares 0.60; plus the second pair of moves 0.75; plus checks and counting 1.11. Every move between vector and general registers costs about half a cycle and the costs add up, so trading five vector operations for two more moves and two loads loses. Any design that needs a second round trip through the general registers is out.
