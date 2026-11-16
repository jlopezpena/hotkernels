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
  element has an add and a store waiting until the chain reaches it.
  `stp d, d` stores two elements with one store operation, so fewer
  operations wait per element and more elements fit. Eight
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
    `stp q, q`: 2.85 at 16, 19.2 at 64, 161 at 256. No gain.
  - timing-only variants with wrong results, to see what limits the overlap:
    half of the `stp` removed: 2.40 at 16, 17.7 at 64, 134 at 256; all stores
    removed: 2.17, 15.8, 126. So waiting stores are still a limit after
    pairing, but nothing was found that reduces them further: wider stores
    need extra operations to assemble their data, and those wait on the chain
    too (see the packed-store result in `cumsum_f32/attempts.md`, which was
    slightly worse). Which structure of the core fills up first (store queue,
    scheduler, reorder buffer) was not established.

## 3. Regrouped additions (block prefix sums), to confirm they are not exact

- **Idea:** the trick that works for `cumsum_i64`: in a block of four build
  a+b, (a+b)+c, ((a+b)+c)+d from the inputs only and add the running sum to
  each, so the carried chain is one `fadd` per four elements. This changes
  which values are rounded, so it must fail an exact check. Tried once, in
  the main loop only (first block and leftovers left strictly sequential).
- **check:** FAIL, wrong result:
  ```
  wrong value in out[9]
  case:     n=16 arr=end seed=1 x=random
  expected: -1.909338883059168
  got:      -1.9093388830591684
  ```
  One unit in the last place, on plain random data, at the second element of
  the first regrouped block.
- **compare:** not run (check failed)
- **Kept:** no, reverted
- **Notes:** for the record, in the standalone timer the regrouped loop takes
  0.29 to 0.30 ns per element from 1,024 to 1,048,576 elements against 0.92
  to 0.97 for the exact chain: 3.2x. That is what the exactness requirement
  costs on long arrays, and no reordering-free code can recover it. A spec
  with a tolerance would be a different kernel.

## 4. Sixteen elements per iteration

- **Idea:** the paired loop of attempt 2 runs just over the latency floor
  once the data is outside the first-level cache: in the standalone timer
  3.02 to 3.03 cycles per element at 65,536 and 1,048,576 against 3.00 to
  3.01 for the naive C. With 16 or 32 elements per iteration it is 3.01 to
  3.02. Use 16.
- **check:** pass (18,136 cases)
- **compare:** significantly faster, 1.004x [1.002 – 1.005]; rerun 1.004x
  [1.004 – 1.006]
- **Kept:** yes
- **Notes:** 1.00x to 1.01x at every size. It gives back most of the half
  percent that attempt 2 lost at the larger sizes; nothing more. Why the
  paired loop is slower than the floor there was not established; it does
  not happen at 4,096, where everything is in the first-level cache.

## Screened in the standalone timer only (not taken to the judge)

- **A separate unpaired loop for long arrays** (one `ldr`, `fadd`, `str` per
  element from about 8,192 elements, as the compiler generates), to remove
  the last few tenths of a percent over the floor: in the standalone timer it
  was within the run-to-run noise of the paired loop at 16,384, 65,536,
  1,048,576 and 4,194,304 (differences of 0.1% to 2% in both directions), so
  it was dropped.

## Final numbers

`judge run` against naive C: 1.12x [1.12 – 1.13] overall (noisy-machine
warning, 3.1%; another run gave 1.13x [1.12 – 1.13] with a 3.0% warning). By
size: 1.76x at 16, 1.56x at 64, 1.27x at 256, 1.05x at 1,024, 1.01x at 4,096,
and 1.00x at every size from 16,384 to 16,777,216. At those large sizes the
kernel is 0.1% to 0.3% behind the C (982.4 us against 979.4 us at 1,048,576),
which the table rounds to 1.00x.

## Why the exact kernel cannot beat the C on long arrays

- Bit-identical output means the additions happen in the order
  `((x0 + x1) + x2) + ...`. Each result is an input of the next, so n elements
  take at least n times the latency of one floating-point add.
- That latency is 3 cycles here: 0.929 ns at 3.228 GHz. Measured, from 4,096
  to 16,777,216 elements: naive C 0.95 to 0.97 ns per element (3.9 us for
  4,096, 16.00 ms for 16,777,216), this kernel 0.95 to 0.97 (3.9 us,
  15.97 ms). Both are within 5% of the bound; the judge reports 1.00x or
  1.01x with intervals that include 1.0 at every one of those sizes.
- Everything else in the loop (two loads, a store and the loop overhead per
  element) needs about 0.4 cycles per element of throughput, so there is
  nothing to save by tightening it: unrolling by 8, 16 or 32, `ldp`/`stp`
  and 32-byte stores all measure the same from 4,096 elements up.
- Any change that makes the chain shorter (attempt 3, pairwise sums, several
  accumulators, vector lanes) changes the rounding and fails `check`.
- The only gain an exact kernel has is at short lengths (attempt 2), where
  the benchmark runs many independent calls back to back and the limit is
  how many of them fit in the core at once, not the chain of one call.
