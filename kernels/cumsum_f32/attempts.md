# cumsum_f32 attempts

This log holds two architectures. ARM64 (Apple M1 Pro) comes first. x86-64 (Intel Core i5-14600KF) follows under [x86-64](#x86-64), with its own numbering.

**The spec changed after attempt 3.** Attempts 1 to 3, the screening notes
after them and the section "Result and why the long-array time cannot
improve" were written when the spec was exact (output bit-identical to adding
left to right). From attempt 4 on the spec is in tolerance mode: `out[i]` may
differ from NumPy by 4 units in the last place of
`(i + 1) * (|x[0]| + ... + |x[i]|)`, so the additions may be regrouped, while
NaN and infinity must still appear exactly where NumPy has them. Read the
earlier entries with that in mind.

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
- **A separate unpaired loop for long arrays** (one `ldr`, `fadd`, `str` per
  element from about 8,192 elements, as the compiler generates), to remove
  the last few tenths of a percent over the floor: in the standalone timer it
  was within the run-to-run noise of the paired loop at 16,384, 65,536,
  1,048,576 and 4,194,304 (differences of 0.1% to 2% in both directions), so
  it was dropped.

## Result and why the long-array time cannot improve

- Final `judge run` against naive C: 1.12x [1.12 – 1.13] overall (noisy-
  machine warning, 3.0%; a second run gave the same figure with a 5.0%
  warning). By size: 1.76x at 16, 1.56x at 64, 1.27x at 256, 1.05x at 1,024,
  1.01x at 4,096, and 1.00x at every size from 16,384 to 16,777,216. At those
  large sizes the kernel is in fact 0.1% to 0.4% behind the C (981.6 us
  against 977.8 us at 1,048,576), which the table rounds to 1.00x.
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

## 4. Regrouped NEON blocks with a sum-of-squares check (tolerance spec)

- **Idea:** four elements per vector, four separate running sums (one per
  lane). Consecutive output vectors differ by W, the sum of the four elements
  ending at each lane. W takes two additions: p = input vector plus an
  overlapping load one element back, then W = p + (p two elements back), the
  second operand an `ext` across the previous p and this one. A block of four
  vectors sums its four W from the inputs only and adds the last output
  vector to each, so the carried chain is one `fadd` per 16 elements. 32
  elements per iteration, then one block of 16, then single vectors, then up
  to three scalar elements.
  Non-finite handling as in `cumsum_f64` attempt 5: chunks of 1,024 elements,
  every output vector squared and accumulated with `fmla` into four
  accumulators, a chunk accepted only if that sum is finite, otherwise redone
  one element at a time (a NaN or infinite input switches to the plain loop
  for the rest; a finite sum above 2^64 restarts the array with the plain
  loop). The proof in the `kernel.s` header differs from the float64 one in
  one respect: float32 rounding is coarse enough that over a very long array
  the left-to-right sum could drift to overflow on its own, so the argument
  bounds that drift (a factor e^(n / 2^24)) and holds below 2^29 elements;
  arrays of 2^29 elements or more use the plain loop throughout.
- **check:** pass (18,136 cases)
- **compare:** significantly faster than attempt 3, 5.33x [5.32 – 5.33]
- **Kept:** yes
- **Notes:** against attempt 3 by size: 0.83x at 16 (3.4 ns against 2.8 ns),
  2.14x at 64, 5.08x at 256, 7.19x at 1,024, 7.66x at 4,096, 7.83x at
  16,384, 7.85x at 65,536, 7.87x at 262,144, 7.83x at 1,048,576, 7.17x at
  4,194,304, 7.27x at 16,777,216. The loop takes about 6.1 cycles per 16
  elements (0.119 ns per element) from 1,024 to 262,144 elements in the
  standalone timer: 24 vector operations per block on four vector pipes. The
  loss at 16 is the fixed cost of a call; see attempt 5.
  The measurements that fixed the design (four accumulators, the order of
  the additions in a block, `ext` for the first shifted vector, chunk size)
  were made on doubles and are in `cumsum_f64/attempts.md`, attempt 5.

## 5. Fewer than 20 elements: the exact chain of attempt 3

- **Idea:** as in `cumsum_f64` attempt 6. At n=16 one block plus the setup
  and the final check is more operations than the paired exact chain, and
  the judge times independent calls, which the processor overlaps, so the
  chain's 3-cycle latency per element does not show. Send arrays below 20
  elements to the paired chain of attempts 2 and 3 (a first block of 8, then
  blocks of 8, then 4, 2, 1). It adds in the reference order, so it needs no
  check and its results are bit-identical to NumPy's.
- **check:** pass (18,136 cases)
- **compare:** significantly faster, 1.016x [1.014 – 1.017]
- **Kept:** yes
- **Notes:** 1.20x at 16 (2.9 ns against 3.4 ns); 0.99x [0.98 – 1.00] at 64
  (8.9 ns both; two more instructions at entry); not significant at any
  other size. The threshold comes from the standalone timer, ns per element,
  chain then blocks: 0.176 / 0.194 at 16, 0.185 / 0.186 at 20, 0.186 / 0.174
  at 24, 0.212 / 0.162 at 32. A throughput choice: a caller whose next step
  waits for the result sees the chain's latency (up to 57 cycles for 19
  elements).

## 6. Cheaper call overhead: one constant, a shorter chunk header, `umaxv` check

- **Idea:** three small things outside the loop, which matter at 64 and 256
  elements. (a) Build the -0.0 vector once with `movi` and copy it with two
  register moves instead of three `movi`. (b) One instruction fewer in the
  chunk header (`csel` straight into the chunk size, `subs` from it).
  (c) End the check with `umaxv` over the four lanes of the sum of squares
  and an integer test of the exponent (one vector operation and a move to a
  general register) instead of two `faddp`, `fsub` and `fcmp` (four vector
  operations). A finite sum of squares is +0.0 or more, so its bits as an
  unsigned integer are below those of +inf, and the largest pattern has an
  all-ones exponent exactly when some lane is infinite or NaN.
- **check:** pass (18,136 cases)
- **compare:** significantly faster, 1.003x [1.003 – 1.004]
- **Kept:** yes (a marginal gain, but significant at most sizes)
- **Notes:** 1.01x [1.01 – 1.02] at 64, 1.00x elsewhere with intervals that
  exclude 1.0 at 256, 1,024, 16,384, 65,536 and from 1,048,576 up. In the
  standalone timer: 0.1386 against 0.1404 ns per element at 64, 0.1268
  against 0.1271 at 128, no difference from 256.

## 7. Both blocks' carry-independent work before the additions that wait for the carry

- **Idea:** as in `cumsum_f64` attempt 8. The W of the second block of an
  iteration go to their own registers (v22, v28, v29, v30) and the iteration
  is ordered as: loads, p, W and block sums of both blocks, then outputs,
  stores and squares of the first block, then of the second.
- **check:** pass (18,136 cases)
- **compare:** significantly faster, 1.001x [1.000 – 1.001]; rerun, the same
  figures
- **Kept:** yes, by the rule; the gain is a tenth of a percent
- **Notes:** significant only at 65,536, 1,048,576 and 16,777,216 (2.15 ms
  against 2.16 ms); every other size 1.00x and not significant. The float64
  loop gained 0.4% from the same change.

## Screened in the standalone timer only, tolerance spec (not taken to the judge)

- **One check per block instead of four (not provable, so not used):** see
  the same item in `cumsum_f64/attempts.md`. It would take the float32 block
  from 24 to 21 vector operations; it was measured on doubles only (12%
  faster from 256 to 16,384 elements) and has the same gap: a running sum
  within rounding distance of FLT_MAX at a position that is not squared.
- **Second block's first shifted vector loaded from `x + 60` instead of
  `ext`:** 3.00 against 3.07 cycles per 8 elements at 4,096, 3.09 against
  3.06 at 1,024, 3.45 against 3.32 at 4,194,304. On doubles the same change
  doubled the time at 262,144 elements when the two arrays sat at the same
  offset in their pages (the load overlaps a pending store in its low
  address bits). Not used.
- **Ending the check with a move to a general register after two `faddp`**
  (instead of `fsub` and `fcmp`): no difference at any size; `umaxv`
  (attempt 6) was the one that showed at 64 elements.
- **Relative position of the two arrays:** with `out` moved by 16 to 2,048
  bytes relative to `x`, 3.06 cycles per 8 elements at 4,096 and 65,536 in
  every case.

## Where it stands (tolerance spec)

- Final `judge run` against naive C, no noise warning: 6.10x [5.98 – 6.10]
  overall. By size: 1.74x at 16 (2.8 ns against 5.0 ns), 3.36x at 64, 6.48x
  at 256, 7.56x at 1,024, 7.76x at 4,096, 7.86x at 16,384 and 65,536, 7.85x
  at 262,144, 7.83x at 1,048,576, 7.13x at 4,194,304, 7.18x at 16,777,216.
- From 1,024 to 1,048,576 elements the kernel takes 0.119 ns per element
  (6.1 cycles per 16 elements) against 0.93 for the chain, and 0.130 at
  4,194,304 and 16,777,216. A block is 24 vector operations on four vector
  pipes, so the loop is at what that count allows.
- Outside the judge, the harness described in `cumsum_f64/attempts.md` ran
  80,303 inputs in float32. No non-finite position or kind differed, out[0]
  always had the bits of x[0], and no zero differed in sign. In units of
  ulp(|x[0]| + ... + |x[i]|) * sqrt(i + 1): the kernel was within 0.90 of
  the exact sum on every class except all-equal 0.1 (17, where the
  left-to-right sum is off by 271). Against the left-to-right sum it was
  within 1.2 on random and mixed-magnitude data, 2.3 on sorted data, 23 on
  positive data spanning nine orders of magnitude and 254 on all-equal 0.1;
  in the last three the difference is the left-to-right sum's own error
  (2.3, 23 and 271 against the exact sum), which at 4,194,304 float32
  elements is far larger than the kernel's.
- The path for 2^29 elements or more was tested by assembling a copy with
  the limit lowered to 256 elements; the chunk logic by assembling copies
  with chunks of 32 (float32) and 16 (float64) elements. Both passed the
  same harness.

# x86-64

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, locked, pinned to one performance core by the judge. Read with the `cumsum_i64` and `cumsum_f64` logs: this kernel was written after both and took its structure from `cumsum_f64` as that stood after its attempt 3.

What the compiler does with `naive.c` (`clang -O3 -march=native`): the scalar chain, one `vaddss` per element at two cycles each: 0.38 ns per element from 1,024 to 1,048,576. NumPy takes 1.6 ns per element at 1,048,576.

## x86-64 1. Baseline: window sums of eight from overlapping loads, a magnitude test per chunk

- **Idea:** eight elements per vector. `P` = input plus the load one element back; `P'` = the same two elements further back, two more loads; `D` = `P` + `P'`, the four elements ending at each lane; `W` = `D` plus `D` four elements back, one `vperm2f128`; output = previous output plus `W`. Four loads, five adds and one permute per eight elements, and a chain of one `vaddps`. The shift by two elements is done with loads and not in registers because across the two halves of a register it takes two shuffles, and shuffles share a port with the adds. From `cumsum_f64`: loads ahead of the stores, `vmovntps` from 24 MB of output, a short path below 64 elements, and the magnitude test per chunk. **The proof and its limit are the ARM64 kernel's:** a chunk is accepted when every output is below 2^64, which is what a finite sum of squares guarantees there, and arrays of 2^29 elements or more take the plain loop. Fewer than 16 elements take the plain chain.
- **check:** pass. Tested outside the judge as `cumsum_f64` was: 20,990 runs, large sizes up to 6,400,010 elements so that the cache-bypassing stores are reached. No failure.
- **compare:** not run (first version). `run`: 2.97x over naive C overall, 26.7x over NumPy.
- **Kept:** yes
- **Notes:** 4.4x to 4.9x over the naive C from 1,024 to 65,536. No stall in the close-placement table.

## x86-64 2. The adds with a memory operand as fused multiply-adds by one

- **Idea:** the five adds and the permute of a vector all go to two ports, three cycles per vector. `vfmadd231ps` with a register of ones computes the same sum with the same single rounding and runs on a third port. Used for the two adds per `D` that have a load as their operand.
- **check:** pass
- **compare:** significantly slower, 0.979x [0.976 – 0.982]
- **Kept:** no
- **Notes:** 0.94x to 0.95x at 64, 1,024 and 4,096; within 2% elsewhere. The multiply-add takes four cycles where the add takes two, and the ports it frees were not what held the loop in the judge's timing.

## x86-64: where it stands

Final `judge run`:

`random`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 2.3 ns | 3.1 ns | 1.34x [1.34 – 1.34] | 732.9 ns | 317x [317 – 318] |
| 64 | 6.5 ns | 13.9 ns | 2.13x [2.12 – 2.13] | 787.9 ns | 121x [120 – 121] |
| 256 | 20.8 ns | 84.5 ns | 4.06x [4.05 – 4.06] | 1.1 us | 53.4x [53.3 – 53.4] |
| 1,024 | 81.2 ns | 380.1 ns | 4.68x [4.66 – 4.71] | 2.4 us | 29.7x [29.6 – 29.9] |
| 4,096 | 322.2 ns | 1.6 us | 4.86x [4.86 – 4.86] | 7.5 us | 23.3x [23.3 – 23.3] |
| 16,384 | 1.4 us | 6.3 us | 4.38x [4.38 – 4.38] | 27.9 us | 19.4x [19.4 – 19.4] |
| 65,536 | 5.7 us | 25.2 us | 4.39x [4.39 – 4.40] | 109.0 us | 19.0x [19.0 – 19.0] |
| 262,144 | 27.6 us | 101.0 us | 3.66x [3.66 – 3.66] | 430.4 us | 15.6x [15.6 – 15.6] |
| 1,048,576 | 152.5 us | 404.0 us | 2.65x [2.64 – 2.66] | 1.71 ms | 11.2x [11.2 – 11.3] |
| 4,194,304 | 1.12 ms | 1.71 ms | 1.52x [1.51 – 1.53] | 6.85 ms | 6.10x [6.05 – 6.13] |
| 16,777,216 | 4.49 ms | 7.02 ms | 1.57x [1.55 – 1.57] | 30.99 ms | 6.91x [6.86 – 6.93] |
| overall |  |  | 2.89x [2.89 – 2.89] |  | 26.1x [26.1 – 26.1] |

With the output buffer 16 bytes behind its input ("apart" is the same kernel with the output clear of the input):

`random`:

| n | kernel | apart | speedup [95%] | naive C | speedup [95%] |
|---|---|---|---|---|---|
| 4,096 | 305.5 ns | 307.8 ns | 1.01x [1.01 – 1.01] | 1.6 us | 5.13x [5.13 – 5.13] |
| 16,384 | 1.4 us | 1.5 us | 1.07x [1.07 – 1.07] | 6.3 us | 4.38x [4.38 – 4.38] |
| 65,536 | 5.7 us | 6.1 us | 1.06x [1.06 – 1.06] | 25.2 us | 4.40x [4.40 – 4.40] |
| 262,144 | 26.9 us | 42.3 us | 1.57x [1.57 – 1.58] | 114.4 us | 4.25x [4.25 – 4.26] |
| 1,048,576 | 135.4 us | 153.7 us | 1.13x [1.13 – 1.14] | 548.9 us | 4.05x [4.05 – 4.06] |
| 4,194,304 | 1.03 ms | 1.26 ms | 1.22x [1.21 – 1.23] | 3.77 ms | 3.66x [3.65 – 3.68] |
| 16,777,216 | 4.19 ms | 4.51 ms | 1.08x [1.07 – 1.08] | 15.76 ms | 3.76x [3.75 – 3.78] |
| overall |  |  | 1.15x [1.15 – 1.15] |  | 4.21x [4.21 – 4.21] |

No stall; the close placement is the faster of the two at 262,144.

What holds it at each size:

- **16 and 64:** the steps of a call.
- **256 to 65,536:** 0.08 to 0.09 ns per element. The loop needs three cycles per eight elements (0.073 ns per element) for its five adds and one permute on two ports.
- **262,144 and 1,048,576:** the third-level cache: 3.7x and 2.65x over the naive C.
- **4,194,304 and 16,777,216:** memory: 1.5x to 1.6x. Only 16,777,216 has an output of 24 MB or more and uses the cache-bypassing stores.

The candidates the design lists for this family: as in the `cumsum_f64` log. The prefix sum across the register is done with overlapping loads and one `vperm2f128`; non-temporal stores are kept from 24 MB; the overflow test is a compare of magnitudes, with the ARM64 kernel's limit of 2^64 because its proof for `float32` needs that limit; no instruction that is slow on other x86-64 processors is used.

**What differs from the ARM64 kernel for a caller:** the test raises no floating-point flag of its own. The limits are the same: running sums of about 1.8e19 and above take the slow path, and arrays of 2^29 elements or more take the plain loop throughout. Adjacent outputs come from eight separate running sums (four on ARM64), so non-negative input does not give a non-decreasing output. `out` must not overlap `x`.

Left untried: the shift by two elements in registers; two interleaved chains; the cache-bypassing stores between 2 MB and 24 MB of output.
