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

## 2. Same chain, elements loaded and stored in pairs (`ldp`/`stp`)

- **Idea:** the chain cannot be shortened, but below about 1,000 elements the
  chain is not what the benchmark measures. The naive C takes 5.2 ns for 16
  elements, 0.33 ns per element, a third of the `fadd` latency: successive
  calls are independent, so the core runs several of them at once, and the
  time per call is set by how many elements it can hold in flight. Each
  element holds one store-queue entry until its `fadd` finishes. `stp d, d`
  stores two elements with one entry, so twice as many elements fit. Eight
  elements per iteration; the first eight have their own block (element 0 is
  stored as loaded) so sizes that are multiples of eight never enter the
  leftover code. Order of additions and operand order are unchanged.
- **check:** pass (18,136 cases)
- **compare:** significantly faster than attempt 1, 1.15x [1.15 – 1.15]
- **Kept:** yes
- **Notes:** `judge run` against naive C: 1.12x [1.12 – 1.12] overall (with a
  noisy-machine warning, 3.8%). By size: 1.73x at 16 (3.0 ns against 5.2 ns),
  1.54x at 64, 1.26x at 256, 1.05x at 1,024, 1.01x (not significant) at
  4,096, and 1.00x from 16,384 to 16,777,216, where both take 0.95 to 0.97 ns
  per element. So the gain is entirely at sizes where calls overlap; on a
  long array the kernel is exactly as fast as the C, at the latency floor.
  Measurements behind it, from a standalone timer that cycles through a pool
  of inputs the way the judge does (ns per call, naive C first):
  - pairs with the first pair special and a pair loop for leftovers: 3.10 at
    16 (C: 4.99), 19.3 at 64 (C: 29.7), 160 at 256 (C: 204).
  - plus the separate first block of eight (this attempt): 2.86 at 16; other
    sizes unchanged.
  - 16 or 32 elements per iteration: 2.86 and 2.90 at 16, 19.0 and 19.1 at
    64, 160 and 161 at 256. No gain over 8, so 8 was kept.
  - packing two sums into a q register and storing four elements with
    `stp q, q`: 2.85 at 16, 19.2 at 64, 161 at 256. No gain: a 32-byte pair
    store still takes two store-queue entries.
  - timing-only variants with wrong results, to see what limits the overlap:
    half of the `stp` removed: 2.40 at 16, 17.7 at 64, 134 at 256; all stores
    removed: 2.17, 15.8, 126. Stores are still the main limit after pairing,
    and 16 bytes per store is the most one entry holds, so there is no
    further step of this kind for 8-byte elements.
