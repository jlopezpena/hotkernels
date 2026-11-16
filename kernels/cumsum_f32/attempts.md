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

## 3. Sixteen elements per iteration

- **Idea:** the paired loop of attempt 2 runs about 1% over the latency floor
  once the data is outside the first-level cache (3.02 to 3.03 cycles per
  element; the naive C is at 3.00). The loss shrinks as the loop body grows:
  in the standalone timer, pair by pair 3.04, 8 per iteration 3.02 to 3.03,
  16 per iteration 3.01, 32 per iteration 3.01. Use 16.
- **check:** pass (18,136 cases)
- **compare:** significantly faster, 1.009x [1.006 – 1.010]; rerun 1.005x
  [1.004 – 1.006]
- **Kept:** yes
- **Notes:** 1.00x to 1.02x at every size; the point is to give back the 1%
  that attempt 2 lost from 16,384 elements up, not to gain anything new. Why
  the paired loop is slower than the floor there was not established; it
  does not happen at 4,096, where everything is in the first-level cache.

## Screened in the standalone timer only (not taken to the judge)

Nanoseconds per call at n = 16 / 64 / 256, cycling through a pool of inputs
as the judge does; naive C is 5.0 / 30 / 205 and attempt 2 is 2.9 / 19.4 /
162.

- **Four results per 16-byte store:** put each pair of sums into one register
  with `mov v.s[1], v.s[0]` after all the adds, and store two such registers
  with `stp d, d`: 2.85 / 20.1 / 168 with 8 per iteration, 2.80 / 20.3 / 168
  with 16, 2.85 / 20.6 / 168 with 32. Slightly worse than plain pairs: the
  two extra moves per four elements wait on the chain just as the stores do,
  so the count of waiting operations goes up, not down.
- **32 elements per iteration:** 2.94 / 19.0 / 163, and 3.01 cycles per
  element at 65,536. Same as 16 with twice the code.
- **Per pair `ldp`, `fadd`, `fadd`, `stp` (interleaved), or the same with
  post-indexed addressing:** 2.91 / 19.8 / 161 and 3.04 cycles per element
  at 65,536 and 262,144. Slower than loads, adds and stores in groups.

## Result and why the long-array time cannot improve

- `judge run` (see the final numbers in the commit that added this section):
  the kernel is faster than the naive C only up to about 1,000 elements.
- From 4,096 elements up the naive C is at 3.00 to 3.1 cycles per element
  and the `fadd` latency is 3. Bit-identical output fixes the order
  `((x0 + x1) + x2) + ...`, each result is an input of the next add, so no
  exact kernel can take less than n times that latency on one array.
- Regrouping the additions removes the chain and is 3.2 times faster on
  doubles, but it changes the rounding and fails `check` on plain random
  data (`cumsum_f64` attempt 3).
- What is left is the short-array gain of attempt 2, which comes from the
  core overlapping independent calls. A caller that feeds each result into
  the next call would not see it.
