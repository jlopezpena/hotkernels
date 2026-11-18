# cumsum_f64 attempts

This log holds two architectures. ARM64 (Apple M1 Pro) comes first. x86-64 (Intel Core i5-14600KF) follows under [x86-64](#x86-64), with its own numbering.

**The spec changed after attempt 4.** Attempts 1 to 4, the screening notes
after them and the sections "Final numbers" and "Why the exact kernel cannot
beat the C on long arrays" were written when the spec was exact (output
bit-identical to adding left to right). From attempt 5 on the spec is in
tolerance mode: `out[i]` may differ from NumPy by 4 units in the last place
of `(i + 1) * (|x[0]| + ... + |x[i]|)`, so the additions may be regrouped,
while NaN and infinity must still appear exactly where NumPy has them. Read
the earlier entries with that in mind; attempt 3 in particular failed only
because the spec was exact then.

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

## 5. Regrouped NEON blocks with a sum-of-squares check (tolerance spec)

- **Idea:** the `cumsum_i64` scheme on doubles. Two elements per vector;
  consecutive output vectors differ by `x[i..i+1] + x[i-1..i]`; a block of
  four vectors sums its four steps from the inputs only and adds the last
  output vector to each, so the carried chain is one `fadd` per 8 elements.
  Non-finite handling: the array is done in chunks of 1,024 elements, every
  output vector is squared and accumulated (`fmla`, four accumulators), and a
  chunk is accepted only if that sum is finite. That proves every output of
  the chunk is finite and below sqrt(DBL_MAX), which in turn proves the
  left-to-right sum is finite there too (the two differ by rounding only, and
  rounding cannot reach from sqrt(DBL_MAX) to overflow). A rejected chunk is
  redone one element at a time from the last output before it: a NaN or
  infinite input switches to the plain left-to-right loop for the rest of the
  array (from there on the kind of every output depends on the inputs only),
  and a finite sum above 2^1023 restarts the whole array with the plain loop,
  because whether the left-to-right sum overflows there can depend on its
  exact rounding. The proof is in the `kernel.s` header. `out[0]` is stored
  from the bits of `x[0]`; everything before the array is -0.0, so sums of
  negative zeros keep their sign as well.
- **check:** pass (18,136 cases)
- **compare:** significantly faster than attempt 4, 3.83x [3.83 – 3.84]
- **Kept:** yes
- **Notes:** against attempt 4 by size: 0.75x at 16 (3.8 ns against 2.8 ns),
  1.66x at 64, 3.80x at 256, 5.33x at 1,024, 5.67x at 4,096, 5.52x at
  16,384, 5.42x at 65,536, 5.63x at 262,144, 5.49x at 1,048,576, 4.13x at
  4,194,304, 4.81x at 16,777,216. In the first-level cache the loop takes
  about 4.2 cycles per 8 elements (0.163 ns per element). The loss at 16 is
  the fixed cost of a call (setting up, the final check) plus 62 operations
  for two blocks, against 46 for the exact chain, whose latency a benchmark of
  independent calls does not see; see attempt 6.
  Measurements behind the design, from a standalone timer that cycles through
  a pool of inputs as the judge does (cycles per 8 elements at n=1,024 unless
  said otherwise; the `cumsum_i64` loop is 3.30):
  - no check at all (wrong on non-finite data, timing only): 3.68. With the
    carried chain cut as well: 3.30. So the 3-cycle `fadd` on the chain costs
    about 0.4 and the check about 0.6.
  - check with 8, 4, 2 accumulators: 4.36, 4.29, 7.4. An `fmla` into its
    accumulator has a latency of 4 cycles, so two accumulators (four `fmla`
    each per 16 elements) are chain-bound; four are not.
  - each `fmla` removed from a block (wrong, timing only) saves 0.15: 4.13
    with three, 3.98 with two. Replacing `fmla` by `fadd` (which would detect
    NaN and infinity but not large finite values): 4.30, no cheaper.
  - four extra integer operations per block cost nothing (4.29); two extra
    vector `fadd` per block cost 0.15. The loop is limited by the four vector
    pipes (16 vector operations per block), not by the number of
    micro-operations.
  - order of the operations inside a block: with the carry add placed before
    the other two additions that wait for the carry, 4.22; with each `fmla`
    right after its output, 4.36. The first is what was committed.
  - chunk of 512, 1,024, 2,048, 4,096, 8,192, 65,536 elements: 4.29 to 4.33
    at every size from 1,024 to 262,144, no difference, so the smallest
    convenient one was taken (a rejected chunk is redone element by element,
    and the square test is per chunk).
  - the first shifted vector of a block loaded from `x - 8` instead of built
    with `ext` (a separate first block for element 0): 4.31, and 12.5 at
    262,144 elements, the same collapse `cumsum_i64` saw with more loads per
    iteration. Not used.
  - carry advanced once per 16 elements (one extra add per iteration): 4.28
    with the check, 3.68 without. No gain.

## 6. Fewer than 32 elements: the exact chain of attempt 4

- **Idea:** attempt 5 lost at n=16 (3.8 ns against 2.8 ns): two blocks plus
  the setup and the final check are about 99 operations, the paired exact
  chain is 46, and the judge times independent calls, which the processor
  overlaps, so the 3-cycle latency per element of the chain does not show.
  Send arrays below 32 elements to the paired chain of attempts 2 and 4 (a
  first block of 8, then blocks of 8, then 4, 2, 1). It adds in the reference
  order, so it needs no check and its results are bit-identical to NumPy's.
- **check:** pass (18,136 cases)
- **compare:** significantly faster, 1.022x [1.018 – 1.026]
- **Kept:** yes
- **Notes:** 1.32x at 16 (2.9 ns against 3.8 ns); 1.00x at every other size
  (65,536 was 0.96x [0.92 – 1.01], not significant, a noisy sample). The
  threshold comes from the standalone timer, ns per element, chain then
  blocks: 0.176 / 0.238 at 16, 0.186 / 0.207 at 24, 0.204 / 0.211 at 28,
  0.211 / 0.195 at 32, 0.242 / 0.188 at 40. This is a throughput choice: a
  caller whose next step waits for the result sees the chain's latency (up
  to 93 cycles for 31 elements) where the blocks would take about 25.

## 7. Shorter chunk header and an integer test at the end of the check

- **Idea:** two things outside the loop. One instruction fewer in the chunk
  header (`csel` straight into the chunk size, `subs` from it), and the check
  ends with a move of the sum of squares to a general register and a test of
  its exponent, instead of `fsub` and `fcmp` (one vector operation fewer).
- **check:** pass (18,136 cases)
- **compare:** significantly faster, 1.001x [1.001 – 1.001]
- **Kept:** yes, by the rule; the gain is a tenth of a percent
- **Notes:** 163.9 ns against 164.1 ns at 1,024, 41.7 against 41.8 at 256,
  657.7 against 658.4 at 4,096; no difference at 16 and from 65,536 up. In
  the standalone timer it could not be told apart from the previous version.

## 8. Both blocks' carry-independent work before the additions that wait for the carry

- **Idea:** the loop is limited by the four vector pipes, and which operations
  are ready when matters a little. Give the second block of an iteration its
  own registers (v22 to v29) and order the iteration as: loads, steps and
  block sums of both blocks, then outputs, stores and squares of the first
  block, then of the second. Same operations, same results.
- **check:** pass (18,136 cases)
- **compare:** significantly faster, 1.004x [1.004 – 1.005]
- **Kept:** yes
- **Notes:** 1.01x at 64, 16,384, 262,144 and 16,777,216; 1.00x with
  intervals above 1.0 at 256, 1,024, 4,096 and 1,048,576; not significant at
  16, 65,536 and 4,194,304. In the standalone timer 4.11 against 4.13 cycles
  per 8 elements at 1,024. Other orders tried there, none better: second
  block's loads before the first block's outputs (4.14), first block's
  squares after the second block's steps (4.14), loop aligned to 64 bytes
  (4.13, and 3% slower at 64 elements).

## Screened in the standalone timer only, tolerance spec (not taken to the judge)

Cycles per 8 elements at n=1,024 unless said otherwise; attempt 5 was 4.22
when these were measured and attempt 8 is 4.11.

- **One check per block instead of four (not provable, so not used):** compute
  out1 as out0 + st1 so that out0, out1 and out2 form a chain, square only
  out2, and rely on the carry for out3. A NaN or infinity anywhere in a block
  then always reaches a squared vector, so every non-finite output of the
  regrouped sum is still detected, with 13 vector operations per block
  instead of 16: 3.73, which is 12% faster from 256 to 16,384 elements. What
  is lost is the size bound on the three outputs that are not squared. A
  running sum can then come within rounding distance of the largest double at
  one of those positions and fall back below 1e154 before the next squared
  one; there the left-to-right sum may round to infinity (and stay infinite)
  while the regrouped one stays finite, and nothing notices. It needs an
  input within about n * 2^-52 of DBL_MAX in relative terms and a running sum
  that lands on the overflow boundary to the last bit, so the judge's cases
  would not find it, but "exactly where NumPy has them" could not be claimed.
  Closing the gap needs information about each unchecked vector, which is one
  operation per vector again (squares of the inputs, of the steps or of the
  outputs all cost four per block).
- **Software prefetch:** `prfm pldl1keep` 512 or 1,024 bytes ahead of the
  loads: 0.2228 and 0.2236 against 0.2229 ns per element at 4,194,304,
  0.1889 and 0.1904 against 0.1890 at 16,777,216, and 4% slower at 65,536.
  With `pstl1keep` on the output as well: three times slower at 4,194,304.
- **`stnp` for the stores:** 0.2186 against 0.2229 at 4,194,304, 0.1906
  against 0.1890 at 16,777,216, 0.1753 against 0.1739 at 1,048,576. Noise.
- **Second block's first shifted vector loaded from `x + 56` instead of
  `ext`:** 4.21 at 1,024, 4.06 at 4,096, and 8.5 at 262,144 elements. The
  load overlaps, in its low address bits, the store to `out + 32..63` of the
  block before it when the two arrays sit at the same offset in their pages,
  and that store is still waiting for its data. Not used.
- **Relative position of the two arrays:** with `out` moved by 16, 64, 96,
  128, 192 and 2,048 bytes relative to `x`, 4.13 at 4,096 and 4.39 to 4.40
  at 65,536 in every case. The committed loop has no such sensitivity.

## Where it stands (tolerance spec)

- Final `judge run` against naive C, no noise warning: 4.44x [4.37 – 4.44]
  overall. By size: 1.74x at 16 (2.8 ns against 5.0 ns), 2.62x at 64, 4.84x
  at 256, 5.61x at 1,024, 5.76x at 4,096, 5.57x at 16,384, 5.43x at 65,536,
  5.64x at 262,144, 5.44x at 1,048,576, 4.14x at 4,194,304, 4.77x at
  16,777,216.
- From 256 to 1,048,576 elements the kernel takes 0.16 to 0.17 ns per element
  (4.1 cycles per 8 elements in the first-level cache) against 0.93 for the
  chain. `cumsum_i64` takes 3.3 cycles: the difference is the check (four
  `fmla` per block, about 0.6 cycles) and the 3-cycle `fadd`. The loop runs
  16 vector operations per block on four vector pipes, so it is within 3% of
  what that count allows; going faster means fewer operations, and the only
  removable ones are the checks (see the first screened item).
- At 4,194,304 and 16,777,216 elements it runs at 0.19 to 0.23 ns per
  element, the same as `cumsum_i64`: memory speed.
- Outside the judge, a harness compared the kernel with the left-to-right sum
  and with a double-double reference on 80,303 inputs (22 classes: random,
  equal, sorted, mixed magnitudes, positive, zeros, denormals, infinities,
  NaN, largest-finite values, values that overflow, values between 1e154 and
  the overflow range, a signalling NaN first; every n from 0 to 300 and 19
  sizes up to 4,194,304; buffers against inaccessible pages, inputs
  read-only). No non-finite position or kind differed, out[0] always had the
  bits of x[0], and no zero differed in sign. In units of
  ulp(|x[0]| + ... + |x[i]|) * sqrt(i + 1): the kernel was within 0.87 of
  the exact sum on every class except all-equal 0.1 (33, where the
  left-to-right sum is off by 229), and within 1.34 of the left-to-right sum
  on every class except that one (262, the left-to-right sum's own error).
- Not tried: a check that costs fewer vector operations and still bounds
  every output. Everything considered needs one operation per output vector.

# x86-64

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, locked, pinned to one performance core by the judge. Read with the `cumsum_i64` log, where the regrouping and its ceiling on this machine were worked out: beyond the first-level cache a plain copy takes 0.115 ns per `float64`, and no running sum can be faster.

What the compiler does with `naive.c` (`clang -O3 -march=native`): the scalar chain, one `vaddsd` per element at two cycles each: 0.38 ns per element from 1,024 to 1,048,576. NumPy takes 1.7 ns per element at 1,048,576.

## x86-64 1. Baseline: window sums from overlapping loads, a magnitude test per chunk

- **Idea:** the regrouping of `cumsum_i64`'s wide version in floating point: `D` = input plus the load one element back, `W` = `D` plus `D` two elements back (one `vperm2f128`), output = previous output plus `W`. The chain is one `vaddpd` per four elements, two cycles, where the reference takes two per element. Before the array every value is -0.0. **The proof that non-finite values land where NumPy has them is the ARM64 kernel's, with the design's candidate in place of the sum of squares:** per chunk of 1,024 elements the largest output magnitude is tracked with `vpand` and `vpmaxud` on the bit patterns, and the chunk is accepted if every output is below 2^1023. A chunk that fails is redone one element at a time under watch, and from there the ARM64 kernel's three cases apply unchanged (store, finish in the plain order, or restart from element 0). The argument is in the `kernel.s` header. Fewer than 8 elements take the plain chain.
- **check:** pass
- **compare:** not run (first version). `run`: 1.78x over naive C overall (2.6x to 3.0x from 1,024 to 65,536), 15.7x over NumPy.
- **Kept:** yes
- **Notes:** **the close-placement table showed a stall:** with the output 16 bytes behind the input, 0.40x at 262,144 and 0.44x at 1,048,576 against the same kernel with the output clear, and nothing at the other sizes. The naive C loses about as much there.

## x86-64 2. Loads ahead of the stores, and stores past the caches from 24 MB

- **Idea:** two things from `cumsum_i64`. The `D` of the next vector is loaded before the current output is stored, so that no load follows a store it could be mistaken for a read of. And from 24 MB of output the vectors are stored with `vmovntpd`, after 0 to 3 elements in the reference order that bring `out` to a 32-byte boundary; their sum takes the place of -0.0 as the output before the first vector.
- **check:** pass
- **compare:** no significant difference, 0.995x [0.989 – 1.001]
- **Kept:** no, in this form
- **Notes:** 1.18x at 4,194,304 and 1.21x at 16,777,216, 1.00x in between, and 0.69x at 16 elements: 4.1 ns against 2.8. Neither a last-pair step that avoids two single-vector steps (1.000x overall, 0.73x at 16) nor deciding the kind of store from the registers instead of a flag below the stack pointer (0.996x, 0.67x) moved it. In a micro-benchmark with warm caches the same two versions differ by 13% at 16 elements, not 50%; the judge cycles through several inputs at small sizes.

## x86-64 3. The same, with the baseline's loop kept for fewer than 32 elements

- **Idea:** attempt 2 for 32 elements and more, and below that one chunk of the baseline's vectors: nothing loaded ahead and no choice of store.
- **check:** pass. The paths the judge does not reach (the cache-bypassing stores, a chunk redone and followed by the fast path, the restart) were tested outside it: 20,990 runs against the left-to-right sum over 41 sizes and five large ones up to 3,200,005, at random alignments, on random data, small integers, -0.0, one or two infinities or NaN at random places, values near the largest double and near 1e154. Positions of NaN and of each infinity must match exactly, `out[0]` must be a copy of `x[0]`, and finite values must be within the spec's tolerance. No failure.
- **compare:** significantly faster, 1.027x [1.019 – 1.035]
- **Kept:** yes
- **Notes:** 1.20x at 4,194,304, 1.21x at 16,777,216, no significant difference anywhere else. The close-placement stall is gone: 1.00x at 262,144 (table below).

## x86-64: where it stands

Final `judge run`:

`random`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 3.0 ns | 3.1 ns | 1.03x [1.02 – 1.04] | 725.8 ns | 242x [238 – 245] |
| 64 | 10.4 ns | 14.9 ns | 1.43x [1.43 – 1.43] | 781.7 ns | 75.2x [75.1 – 75.3] |
| 256 | 37.4 ns | 85.0 ns | 2.27x [2.27 – 2.27] | 1.1 us | 30.5x [30.5 – 30.6] |
| 1,024 | 146.4 ns | 381.3 ns | 2.60x [2.60 – 2.61] | 2.5 us | 16.8x [16.7 – 16.8] |
| 4,096 | 574.4 ns | 1.6 us | 2.74x [2.73 – 2.74] | 7.9 us | 13.8x [13.7 – 13.8] |
| 16,384 | 2.1 us | 6.3 us | 3.01x [3.01 – 3.01] | 28.9 us | 13.8x [13.8 – 13.8] |
| 65,536 | 9.0 us | 25.2 us | 2.81x [2.81 – 2.81] | 113.4 us | 12.6x [12.6 – 12.6] |
| 262,144 | 64.3 us | 101.4 us | 1.58x [1.57 – 1.58] | 449.0 us | 6.98x [6.96 – 6.99] |
| 1,048,576 | 293.8 us | 412.4 us | 1.40x [1.39 – 1.42] | 1.79 ms | 6.10x [6.03 – 6.16] |
| 4,194,304 | 1.95 ms | 2.53 ms | 1.30x [1.29 – 1.32] | 9.08 ms | 4.65x [4.64 – 4.67] |
| 16,777,216 | 7.98 ms | 10.41 ms | 1.30x [1.30 – 1.31] | 36.07 ms | 4.52x [4.51 – 4.54] |
| overall |  |  | 1.83x [1.82 – 1.83] |  | 16.2x [16.2 – 16.2] |

With the output buffer 16 bytes behind its input ("apart" is the same kernel with the output clear of the input):

`random`:

| n | kernel | apart | speedup [95%] | naive C | speedup [95%] |
|---|---|---|---|---|---|
| 4,096 | 528.3 ns | 587.1 ns | 1.11x [1.11 – 1.11] | 1.6 us | 2.98x [2.97 – 2.98] |
| 16,384 | 2.1 us | 2.3 us | 1.08x [1.08 – 1.08] | 6.3 us | 3.02x [3.02 – 3.02] |
| 65,536 | 9.1 us | 9.3 us | 1.02x [1.02 – 1.03] | 25.2 us | 2.78x [2.78 – 2.79] |
| 262,144 | 65.6 us | 65.5 us | 1.00x [1.00 – 1.00] | 464.9 us | 7.09x [7.07 – 7.11] |
| 1,048,576 | 420.0 us | 525.2 us | 1.25x [1.24 – 1.27] | 1.65 ms | 3.93x [3.90 – 3.95] |
| 4,194,304 | 1.85 ms | 1.97 ms | 1.06x [1.06 – 1.07] | 2.46 ms | 1.33x [1.32 – 1.34] |
| 16,777,216 | 7.80 ms | 7.94 ms | 1.02x [1.02 – 1.02] | 10.24 ms | 1.31x [1.31 – 1.32] |
| overall |  |  | 1.08x [1.07 – 1.08] |  | 2.76x [2.76 – 2.76] |

What holds it at each size:

- **16 and 64:** the steps of a call; 16 elements are level with the naive C.
- **256 to 65,536:** 0.13 to 0.15 ns per element, against 0.115 for a plain copy beyond the first-level cache. The vector loop itself needs two cycles per four elements (0.097 ns per element): the chain is one add of two cycles, and the three adds and the permute of a vector fill the two ports that take them.
- **262,144 and 1,048,576:** the third-level cache, 1.4x to 1.6x over the naive C.
- **4,194,304 and 16,777,216:** memory, with a third less traffic than the naive C: 1.3x.

The candidates the design lists for this family (`docs/x86-64-design.md`):

| Candidate | Outcome |
|---|---|
| A prefix sum across the register with an extra permute | Kept (attempt 1): the shift by one element is an overlapping load, and one `vperm2f128` per vector does the rest |
| Non-temporal stores above the third-level cache | Kept (attempt 3) |
| The overflow test as a compare and a `vmovmskpd` instead of a sum of squares | Kept (attempt 1), as an integer maximum of the magnitudes per vector and one compare and `vmovmskpd` per chunk |
| Masked loads and stores for leftover elements | Not tried: open issue 3 of the design |
| An output just behind the input | It stalled the baseline at 262,144 and 1,048,576 (0.40x and 0.44x). Loading ahead of the stores removes it (attempt 3) |
| Alignment of data | Not tried as a general step: rejected on `cumsum_i64` (its attempt 2), and the judge's two buffers cannot both be aligned. Used where the stores need it |
| Alignment of code | The loops start on 64-byte boundaries. Not varied |
| Instructions slow on other x86-64 processors | None used |

**What differs from the ARM64 kernel for a caller.**

- **The fast path reaches further.** The ARM64 kernel's sum of squares limits it to running sums below about 1e154. The magnitude test here accepts everything below 2^1023, about 9e307.
- **The test raises no floating-point flag.** The sum of squares raises overflow on finite data once the squares overflow. The regrouped additions themselves can still raise overflow or invalid where the left-to-right sum does not, so the binding's handling of flags is still needed.
- **Everything else is the same:** not bit-identical to NumPy; adjacent outputs come from separate running sums (four here, two on ARM64), so non-negative input does not give a non-decreasing output; `out` must not overlap `x`.

Left untried: two interleaved chains (the output two vectors back plus the sum of two `W`), which halves the chain but adds half an add per vector to ports that are already full; the adds that are not on the chain done as fused multiply-adds by one to move them to another port (slower on `cumsum_f32`, its attempt 2); the cache-bypassing stores between 2 MB and 24 MB of output (see `cumsum_i64`, attempt 5).

Found here and worth trying on ARM64: the magnitude test in place of the sum of squares. It would lift the ARM64 fast path's limit from 1e154 to 9e307 for `float64` and stop the test itself from raising overflow. Whether an integer maximum per vector is as cheap there as the `fmla` is for the ARM64 run to measure.
