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

## 2. Same chain, elements loaded and stored in pairs (`ldp`/`stp`)

- **Idea:** as in `cumsum_f64` attempt 2. Below about 1,000 elements the
  benchmark does not measure the chain of one call: successive calls are
  independent, the core runs several at once, and the time per call depends
  on how many it can hold in flight (the naive C takes 5.1 ns for 16
  elements, a third of the `fadd` latency per element). Storing two results
  with one `stp s, s` halves the store operations waiting on the chain. Eight
  elements per iteration, a separate first block of eight (element 0 is
  stored as loaded), same order of additions and same operand order.
- **check:** pass (18,136 cases)
- **compare:** significantly faster than attempt 1, 1.15x [1.15 – 1.15]
- **Kept:** yes
- **Notes:** against attempt 1: 2.30x at 16, 1.54x at 64, 1.27x at 256, 1.04x
  at 1,024, 1.00x at 4,096, then 0.99x [0.99 – 0.99] from 16,384 to
  1,048,576 and 1.00x above. The 1% loss at the larger sizes is real: in the
  standalone timer this loop takes 3.02 to 3.03 cycles per element there
  against 3.00 for the naive C. See attempt 3.
