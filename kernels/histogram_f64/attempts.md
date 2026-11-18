# histogram_f64 attempts

This log holds two architectures. ARM64 (Apple M1 Pro) comes first. x86-64 (Intel Core i5-14600KF) follows under [x86-64](#x86-64), with its own numbering.

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

## 7. Zone path: skip the edge corrections where they provably cannot fire (committed, then taken out again)

- **Idea:** NOT a literal transcription of naive.c. Compute t with naive's three operations, convert to 32.32 fixed point with one `fcvtzs #32`, and accept a lane into bin trunc(t) when the fraction of t is at least 2^-12 away from 0 and 1 and the bin is not the last one. An error bound on t and on the computed edges (in the kernel.s header of that commit) shows neither correction can fire for such an element, given an entry check on nbins, step and max(|lo|, |hi|) * nbins / width. Everything else (about 1 element in 2,048, the last bin, out of range, NaN) goes to the scalar copy of naive.c. 4 vector operations per two elements instead of 14.
- **check:** pass. Also a private differential test against naive.c: 200,000 rounds, 410 million elements, random nbins/lo/hi (including offset and extreme-scale ranges) and adversarial values (on and within a few ulps of computed edges, around the 2^-12 threshold, around lo and hi, NaN, infinities, huge values): no mismatch. The same test with the zone check disabled failed in 1,387 of 2,000 rounds, so it can see the errors it is looking for.
- **compare:** significantly faster, 1.299x [1.298 – 1.300] (1.13x at 16, 1.24x at 64, 1.31–1.38x from 256 up)
- **Kept:** no. The brief for this kernel asked for naive.c's operations to be followed one by one, and this path skips two of them on most elements, relying on a proof instead. It is committed so it can be picked up if that trade is acceptable, and the next commit restores the literal kernel.
- **Notes:** two other shapes of the per-lane check were measured privately and were no better: bias and half-swap done in the vector (`add`, `rev64`) so each lane needs two compares and no shift, 0.64–0.68 ns per element, same as this version; one branch per lane via `ccmp`, 0.71–0.75, slower.

## 8. Narrow four indices to 32 bits and move them out two per register (measured privately)

- **Idea:** halve the vector-to-general-register moves in the literal kernel: `sqxtn` + `sqxtn2` pack the four q of a block into one vector (saturating, so an out-of-range q stays out of range), then one `fmov` and one `umov` carry two each and a shift splits them.
- **check:** private micro-benchmark result check passed
- **compare:** not run. Micro-benchmark, three rounds: 0.90–0.93 ns per element against 0.88–0.91 for HEAD (normal); 0.90–0.95 against 0.89–0.94 (uniform). No gain.
- **Kept:** no
- **Notes:** two narrowing operations cost as much as the two moves they save.

## 9. Adopt the zone path of attempt 7

- **Idea:** attempt 7 was left out because the judge could test one bin count and range only, and the path rests on an entry check of exactly those. The spec now lists thirteen further bin counts and ranges (one and two bins, exact and inexact widths, 4,096 bins, ranges of width 2e-3, 2e300 and 2e-310, a range that excludes zero), so the kernel of attempt 7 is restored unchanged.
- **check:** pass, 53,652 cases across the fourteen scalar sets. Also a differential test outside the judge against `np.histogram` itself (not `naive.c`): 430,000 rounds, 126 million elements, random bin counts and ranges from six families (ordinary, far from zero and narrow, extreme scales, exact binary widths, integer edges, denormal-sized), with values on and up to three ulps either side of NumPy's edges, around the 2^-12 threshold, at lo and hi and one ulp outside, NaN, infinities and the largest magnitudes: no mismatch. A histogram without the edge corrections differs from NumPy in 2,086 of 3,000 rounds of the same inputs, so the test sees what it is looking for.
- **compare:** significantly faster, 1.298x [1.295 – 1.301] (1.14x at 16, 1.23x at 64, 1.31–1.36x from 256 up)
- **Kept:** yes
- **Notes:** the judge's own data seldom lands within rounding distance of a bin edge, so the fourteen sets mostly exercise the entry check, the table sizes and the out-of-range handling. The evidence that the 2^-12 margin is enough is the proof in the `kernel.s` header and the two differential tests.

## 10. Correctness fix: the zone path's entry check accepted an overflowed product

- **Idea:** not an optimisation. The entry check `max(|lo|, |hi|) * nbins <= width * 2^39` is made in doubles. For a narrow range near the largest float64 values both products overflow to infinity, the compare sees them as equal, and the zone path ran where its error bound does not hold. Found in review of the pull request by a differential test against `np.histogram`: with `lo = 1e308`, `hi = 1e308 + 1e298` and 1,000 bins, 42 of 2,000 elements placed on and one step above the edges were counted in a neighbouring bin. The check now also sends the call to the compute path when `max(|lo|, |hi|) * nbins` is not finite. With that product finite the rounded comparison is safe: an infinite right side means it really is the larger, and two finite products are each within one rounding of their real values, which the bound's slack absorbs (it needs 2^-12 and uses about 2^-14).
- **check:** pass, 65,533 cases across sixteen scalar sets. Two were added at the ends of the float64 range (`top_of_range`, `bottom_of_range`), inside the region that failed. They would not have caught this: the only values the judge generates inside those ranges are the largest magnitudes, which sit on the outer edge and never take the zone path. The evidence is a differential test outside the judge against `np.histogram`, with values on and up to four steps either side of every edge, at fractions 2^-12, 1 - 2^-12 and 1/2 of every bin, and random in-range values: over 6,000 random ranges of magnitude 2^960 to 2^1024 (636 with both products infinite, 1.6 billion elements) no mismatch, where the kernel before the fix was wrong in 28 of 3,000 such ranges and the reference in none.
- **compare:** no significant difference, 0.998x [0.992 – 1.004]. Four instructions once per call.
- **Kept:** yes, as a correctness fix; the one rule's speed test does not apply.
- **Notes:** the proof in the header was right and stays as it was; the code did not establish one of its conditions. A range this far from the data is something the judge cannot test until its inputs can depend on the scalars.

## Where this stands

The literal kernel needs 14 vector operations per two elements (three for t, two for the truncation as integer and as double, five for the two edges, two compares, two to apply them) plus two moves to general registers, and measures about 5.5 cycles per vector against a port limit near 4.5. I found no way to do the same individually rounded operations with fewer vector operations; the ideas that cut them (edge table, attempt 6) pay for it in moves. Attempt 7, adopted as attempt 9, gets past that by changing what is computed per element: four vector operations per two elements on the path nearly every element takes.

## Not tried

- Several count tables. On this kernel the increments are not the limit: an input that never repeats a bin within 50 elements runs at 0.83 ns per element against 0.88 for normal data, and extra tables need zeroing and summing for an unbounded nbins.
- Replacing the divide by a multiply with the reciprocal: it would change the rounding of the first estimate, which the brief rules out.

# x86-64

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, locked, pinned to one performance core by the judge.

What the compiler does with `naive.c` (`clang -O3 -march=native`): the scalar loop with its divide and its three data-dependent branches, 3.9 ns per element at every size. NumPy takes 4.0 ns per element at 1,048,576.

**The divide.** `vdivpd` on four doubles takes eight cycles on this processor, measured, so a kernel that computes naive.c's estimate as naive.c does cannot go below two cycles per element. The ARM64 kernel's fast path does compute it that way.

## x86-64 1. Baseline: a scaled estimate in fixed point, the exact route for elements near an edge

- **Idea:** the ARM64 kernel's zone path, with three changes that this architecture suggests. **(1) No divide per element.** The position among the bins is estimated as `(v - lo) * S` with `S = nbins / width` computed once. The proof never needed naive.c's own estimate, only an estimate within `4 u N` of the real position; this one is within `3.01 u N`. **(2) Fixed point through `vcvttpd2dq`.** `S` is scaled by 4,096, so the conversion gives the bin in the upper bits and twelve bits of fraction below. An element is accepted when that number is at most `nbins * 4096 - 1` as an unsigned number and its fraction bits are neither all zero nor all one, which is the ARM64 kernel's `DELTA = 2^-12` on either side of an edge. The design's candidate holds: a NaN, and anything too large, converts to 0x80000000 and fails the same unsigned compare as a bin past the last. **(3) The last bin is accepted too.** naive.c never takes its last correction there, so only the lower edge has to be certain, and the header shows the element is below `hi`. The ARM64 kernel sends the whole last bin to its scalar route. Everything not accepted, and every call whose bin count and range fail the once-per-call check (the ARM64 kernel's, plus `nbins < 2^19` and a finite `S`), goes through a scalar copy of naive.c. The proof is in the `kernel.s` header.
- **check:** pass (65,533 cases, the spec's sixteen bin counts and ranges)
- **compare:** not run (first version). `run`: 6.67x over naive C overall, 27.8x over NumPy.
- **Kept:** yes
- **Notes:** the judge's data seldom lands near a bin edge, so the evidence about edges is a differential test outside it against `naive.c` compiled by clang: random bin counts (1 to 600,000, including both sides of 2^19) and ranges (around zero, far from zero, narrow against their distance from zero, up to the top of the float64 range), with elements placed on bin edges, within three units in the last place of them, on and around the 2^-12 threshold, at `lo` and `hi`, outside the range, and NaN, infinities and denormals. **60,000 rounds, 213 million elements, no mismatch.** The same test on a copy of the kernel with the fraction test removed fails 2,264 of 3,000 rounds, so it sees what it is meant to see.

## x86-64 2. Four count tables, bins read back from memory

- **Idea:** two neighbouring elements in the same bin make the second update wait for the first to be stored. With at most 512 bins and at least 16 elements per bin, the four elements of a block are counted in four tables (three on the stack, added to `counts` at the end), as `bincount_u8` does. The four bins are stored below the stack pointer and read back as indices, which is fewer instructions than taking them out through `vmovq` and `vpextrq`.
- **check:** pass; differential test 20,000 rounds, no mismatch
- **compare:** not run; a micro-benchmark outside the judge was enough
- **Kept:** no
- **Notes:** ns per element at 4,096 elements, before and after: normal data 0.438 and 0.509; bins in rotation, so that no two neighbours share one, 0.357 and 0.426; one bin for every element 1.258 and 0.467. Slower on everything but data that is all in one bin.

## x86-64 3. Two count tables, bins through registers

- **Idea:** the same with the first and third element of a block in `counts` and the second and fourth in one table on the stack, and the bins taken out as in attempt 1.
- **check:** pass; differential test 20,000 rounds, no mismatch
- **compare:** significantly slower, 0.939x [0.914 – 0.965]
- **Kept:** no
- **Notes:** in the micro-benchmark: normal data 0.434 ns per element against 0.438, one bin 0.717 against 1.258. On normally distributed data over 50 bins two neighbours share a bin 3.4% of the time, which is not what holds the loop; clearing and adding the table costs more than it saves.

## x86-64 4. The acceptance tests by additions and logic only

- **Idea:** a calibration of the loop's instructions showed where its time goes: the vector part alone, without taking the bins out or counting, runs at 4.8 cycles per block of four where its instruction count needs 2.5. A compare or a minimum of 32-bit lanes runs on two of the three vector ports, the same two the multiply and the conversion need. Both tests can leave their verdict in a sign bit using additions and logic, which run on all three: `((z + 1) & 0xffe) - 1` is negative exactly when the fraction bits are all zero or all one, and `(limit - z) | z` is negative exactly when z is negative or above the limit.
- **check:** pass; differential test 30,000 rounds, 106 million elements, no mismatch
- **compare:** significantly faster, 1.009x [1.003 – 1.015]
- **Kept:** yes
- **Notes:** 1.02x to 1.03x from 1,024 to 4,194,304, 0.97x at 16 and 64. Smaller than the calibration promised.

## x86-64: where it stands

Final `judge run`:

`random`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 18.9 ns | 71.7 ns | 3.79x [3.78 – 3.79] | 15.8 us | 835x [834 – 835] |
| 64 | 53.7 ns | 280.9 ns | 5.24x [5.23 – 5.24] | 16.1 us | 300x [299 – 300] |
| 256 | 131.6 ns | 993.8 ns | 7.55x [7.55 – 7.56] | 17.3 us | 131x [131 – 132] |
| 1,024 | 436.7 ns | 4.0 us | 9.05x [9.05 – 9.06] | 20.7 us | 47.5x [47.4 – 47.5] |
| 4,096 | 1.7 us | 15.8 us | 9.06x [9.06 – 9.07] | 31.7 us | 18.2x [18.1 – 18.2] |
| 16,384 | 7.2 us | 64.4 us | 8.94x [8.93 – 8.94] | 76.7 us | 10.6x [10.6 – 10.6] |
| 65,536 | 34.2 us | 255.0 us | 7.46x [7.45 – 7.48] | 572.5 us | 16.8x [16.7 – 16.8] |
| 262,144 | 138.3 us | 1.04 ms | 7.54x [7.53 – 7.55] | 1.07 ms | 7.73x [7.71 – 7.76] |
| 1,048,576 | 557.7 us | 4.10 ms | 7.35x [7.31 – 7.40] | 4.20 ms | 7.53x [7.49 – 7.57] |
| 4,194,304 | 2.53 ms | 16.44 ms | 6.50x [6.47 – 6.52] | 17.14 ms | 6.77x [6.74 – 6.82] |
| 16,777,216 | 10.10 ms | 65.73 ms | 6.51x [6.49 – 6.52] | 68.60 ms | 6.80x [6.77 – 6.80] |
| overall |  |  | 6.98x [6.98 – 6.99] |  | 29.3x [29.2 – 29.3] |

With the output buffer 16 bytes behind its input ("apart" is the same kernel with the output clear of the input):

`random`:

| n | kernel | apart | speedup [95%] | naive C | speedup [95%] |
|---|---|---|---|---|---|
| 4,096 | 1.8 us | 1.9 us | 1.06x [1.06 – 1.06] | 15.8 us | 9.00x [9.00 – 9.01] |
| 16,384 | 8.1 us | 7.0 us | 0.87x [0.86 – 0.87] | 63.7 us | 7.88x [7.88 – 7.89] |
| 65,536 | 34.6 us | 29.1 us | 0.84x [0.84 – 0.84] | 255.8 us | 7.40x [7.38 – 7.41] |
| 262,144 | 121.3 us | 136.3 us | 1.12x [1.12 – 1.13] | 1.03 ms | 8.45x [8.45 – 8.46] |
| 1,048,576 | 559.2 us | 518.4 us | 0.93x [0.92 – 0.94] | 4.10 ms | 7.32x [7.29 – 7.37] |
| 4,194,304 | 2.48 ms | 2.55 ms | 1.03x [1.02 – 1.04] | 16.41 ms | 6.61x [6.55 – 6.65] |
| 16,777,216 | 10.07 ms | 10.09 ms | 1.00x [1.00 – 1.00] | 65.73 ms | 6.53x [6.52 – 6.54] |
| overall |  |  | 0.97x [0.97 – 0.98] |  | 7.55x [7.54 – 7.56] |

The counters are the output here, and with 50 bins they are 400 bytes just behind the start of x. The close placement costs up to a sixth at some sizes (0.84x at 65,536) and nothing at others; the naive C is not affected at all, being forty times slower per element than a stalled load.

What holds it:

- **1,024 to 16,384:** 0.42 to 0.44 ns per element, 2.2 cycles. Knock-out variants in a micro-benchmark: the vector part with its tests 0.22 ns, taking the four bins out through general registers 0.09 more, the four counter updates 0.05 more, and the rest is the elements outside the range, 0.27% of normal data, each of which sends its block of four through the scalar route behind a mispredicted branch. The vector part is held by the three vector ports: eleven vector instructions per block before the bins are taken out, and four more to take them out.
- **65,536 and up:** 0.52 to 0.60 ns per element; the elements come from further away.
- **16 and 64:** clearing 50 counters and the once-per-call check with its three divides.

The candidates the design lists for this kernel (`docs/x86-64-design.md`):

| Candidate | Outcome |
|---|---|
| Four elements per vector | Kept (attempt 1) |
| `vcvttpd2dq` returns 0x80000000 for NaN and for anything too large, so a NaN needs no test of its own | Kept (attempt 1): one unsigned test rejects NaN, negative positions and bins past the last |
| No scatter, so the counter updates stay scalar | As expected: four `inc` on memory per block |
| The error-bound proof carries over and must be rechecked against the instruction that rounds here | Rechecked and changed: the estimate is a multiply by a precomputed scale, not naive.c's divide and multiply, with its own error term (3.01 u N); the fraction is truncated to twelve bits by the conversion, which makes the accepted zone exactly `2^-12 <= fraction < 1 - 2^-12`; the last bin is accepted. The header has the argument. Tested as described under attempt 1 |
| Masked loads and stores for leftover elements | Not needed: the last 0 to 3 elements take the scalar route |
| Alignment of data, non-temporal stores | Not tried: one 32-byte load per block, and an output of `nbins` counters |
| An output just behind the input | Up to 0.84x; see above |
| Instructions slow on other x86-64 processors | None used |

**What differs from the ARM64 kernel for a caller.**

- **Bin counts of 2^19 (524,288) and more take the scalar route,** at the speed of the naive C. The ARM64 kernel's limit is 2^31.
- **Few bins are not a slow case.** With one bin the ARM64 kernel sends every element to its scalar route, and with two bins half of them; here the last bin is on the fast path.
- **Everything else is the same:** a finite range with `lo < hi` and at least one bin; ranges that fail the once-per-call check run at the speed of the naive C, with correct results.

Left untried: taking the bins out with one store and four loads without the four tables (attempt 2 changed both at once); the tests in general registers on two elements at a time; skipping the scalar route for a block whose only rejected elements are out of range, which needs no divide to decide.

Found here and worth trying on ARM64: **(1)** the multiply by a precomputed scale in place of the divide. The proof allows it, and the ARM64 fast path spends a `fdiv` per two elements. **(2)** The last bin on the fast path.
