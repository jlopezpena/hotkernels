# cumsum_f64 attempts

All timings on an Apple M1 Pro, with `--unlocked` (the spec was not on `main`
yet) and with other agents timing other kernels on the same machine.

The spec is exact: the output must be bit-identical to adding left to right.
That fixes the order of the additions, so every element costs one `fadd` on a
single chain. `fadd` has a latency of 3 cycles on this core; at 3.228 GHz that
is 0.929 ns per element, and it is the floor for any exact kernel on one long
array.

## 1. Straightforward scalar loop

- **Idea:** copy the first element, then one `ldr`/`fadd`/`str` per element
  with post-indexed pointers. The starting point.
- **check:** pass
- **compare:** not run (first version). `judge run` against naive C: 0.98x
  [0.97 – 0.98] overall; 0.77x at n=16, and 1.00x (not significant) at every
  size from 64 up.
- **Kept:** yes (baseline to compare against)
- **Notes:** from 1,024 elements up both this and the naive C take 0.926 to
  0.97 ns per element (948.4 ns for 1,024 elements, 16.27 ms for 16,777,216):
  2.99 to 3.1 cycles, which is the `fadd` latency. The C is unrolled eight
  times by the compiler; that does not matter once the chain is the limit, and
  it is why the C is faster at n=16 (5.2 ns against 6.8 ns).
