# cumsum_f32 attempts

All timings on an Apple M1 Pro, with `--unlocked` (the spec was not on `main`
yet) and with other agents timing other kernels on the same machine.

The spec is exact: the output must be bit-identical to adding left to right.
That fixes the order of the additions, so every element costs one `fadd` on a
single chain. `fadd` has a latency of 3 cycles on this core; at 3.228 GHz that
is 0.929 ns per element, and it is the floor for any exact kernel on one long
array. `kernels/cumsum_f64/attempts.md` has the same analysis for doubles,
including the regrouping experiment that fails `check`; it was not repeated
here.

## 1. Straightforward scalar loop

- **Idea:** copy the first element, then one `ldr`/`fadd`/`str` per element
  with post-indexed pointers. The starting point.
- **check:** pass
- **compare:** not run (first version). `judge run` against naive C: 0.97x
  [0.97 – 0.98] overall, with a noisy-machine warning (5.5%); 0.76x at n=16
  and 1.00x at every size from 64 up.
- **Kept:** yes (baseline to compare against)
- **Notes:** from 1,024 elements up both this and the naive C take 0.95 to
  0.99 ns per element (974.0 ns for 1,024 elements, 15.98 ms for 16,777,216):
  the `fadd` latency. The C is unrolled eight times, which only shows at n=16
  (5.3 ns against 6.9 ns).
