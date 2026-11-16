# cumsum_i64 attempts

All timings on an Apple M1 Pro, with `--unlocked` (the spec was not on `main`
yet) and with other agents timing other kernels on the same machine, so the
numbers are noisier than a quiet run.

Reference point: the naive C (`clang -O3`, unrolled eight times by the
compiler, still one `add` per element on a single chain) runs at about
0.31 ns per element from 256 to 1,048,576 elements, which is one cycle per
element at 3.2 GHz. That is the latency of the carried `add`, not the width of
the core.

## 1. Simplest scalar loop

- **Idea:** one `ldr`/`add`/`str` per element with post-indexed pointers and a
  down-counter; the starting point, to get `check` passing.
- **check:** pass
- **compare:** not run (first version). `judge run`: 0.80x [0.80 – 0.81] of
  naive C overall, 0.56x at n=16 and about 0.85x from 4,096 up.
- **Kept:** yes (baseline)
- **Notes:** slower than the C because the C is unrolled eight times and this
  pays the loop branch on every element (0.37 ns against 0.31 ns per element).
