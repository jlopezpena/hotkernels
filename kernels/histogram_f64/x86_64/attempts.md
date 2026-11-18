# histogram_f64 attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, locked, pinned to one performance core by the judge.

What the compiler does with `naive.c` (`clang -O3 -march=native`): the scalar loop with its divide and its three data-dependent branches, 3.9 ns per element at every size. NumPy takes 4.0 ns per element at 1,048,576.

**The divide.** `vdivpd` on four doubles takes eight cycles on this processor, measured, so a kernel that computes naive.c's estimate as naive.c does cannot go below two cycles per element. The ARM64 kernel's fast path does compute it that way.

## 1. Baseline: a scaled estimate in fixed point, the exact route for elements near an edge

- **Idea:** the ARM64 kernel's zone path, with three changes that this architecture suggests. **(1) No divide per element.** The position among the bins is estimated as `(v - lo) * S` with `S = nbins / width` computed once. The proof never needed naive.c's own estimate, only an estimate within `4 u N` of the real position; this one is within `3.01 u N`. **(2) Fixed point through `vcvttpd2dq`.** `S` is scaled by 4,096, so the conversion gives the bin in the upper bits and twelve bits of fraction below. An element is accepted when that number is at most `nbins * 4096 - 1` as an unsigned number and its fraction bits are neither all zero nor all one, which is the ARM64 kernel's `DELTA = 2^-12` on either side of an edge. The design's candidate holds: a NaN, and anything too large, converts to 0x80000000 and fails the same unsigned compare as a bin past the last. **(3) The last bin is accepted too.** naive.c never takes its last correction there, so only the lower edge has to be certain, and the header shows the element is below `hi`. The ARM64 kernel sends the whole last bin to its scalar route. Everything not accepted, and every call whose bin count and range fail the once-per-call check (the ARM64 kernel's, plus `nbins < 2^19` and a finite `S`), goes through a scalar copy of naive.c. The proof is in the `kernel.s` header.
- **check:** pass (65,533 cases, the spec's sixteen bin counts and ranges)
- **compare:** not run (first version). `run`: 6.67x over naive C overall, 27.8x over NumPy.
- **Kept:** yes
- **Notes:** the judge's data seldom lands near a bin edge, so the evidence about edges is a differential test outside it against `naive.c` compiled by clang: random bin counts (1 to 600,000, including both sides of 2^19) and ranges (around zero, far from zero, narrow against their distance from zero, up to the top of the float64 range), with elements placed on bin edges, within three units in the last place of them, on and around the 2^-12 threshold, at `lo` and `hi`, outside the range, and NaN, infinities and denormals. **60,000 rounds, 213 million elements, no mismatch.** The same test on a copy of the kernel with the fraction test removed fails 2,264 of 3,000 rounds, so it sees what it is meant to see.

## 2. Four count tables, bins read back from memory

- **Idea:** two neighbouring elements in the same bin make the second update wait for the first to be stored. With at most 512 bins and at least 16 elements per bin, the four elements of a block are counted in four tables (three on the stack, added to `counts` at the end), as `bincount_u8` does. The four bins are stored below the stack pointer and read back as indices, which is fewer instructions than taking them out through `vmovq` and `vpextrq`.
- **check:** pass; differential test 20,000 rounds, no mismatch
- **compare:** not run; a micro-benchmark outside the judge was enough
- **Kept:** no
- **Notes:** ns per element at 4,096 elements, before and after: normal data 0.438 and 0.509; bins in rotation, so that no two neighbours share one, 0.357 and 0.426; one bin for every element 1.258 and 0.467. Slower on everything but data that is all in one bin.

## 3. Two count tables, bins through registers

- **Idea:** the same with the first and third element of a block in `counts` and the second and fourth in one table on the stack, and the bins taken out as in attempt 1.
- **check:** pass; differential test 20,000 rounds, no mismatch
- **compare:** significantly slower, 0.939x [0.914 – 0.965]
- **Kept:** no
- **Notes:** in the micro-benchmark: normal data 0.434 ns per element against 0.438, one bin 0.717 against 1.258. On normally distributed data over 50 bins two neighbours share a bin 3.4% of the time, which is not what holds the loop; clearing and adding the table costs more than it saves.

## 4. The acceptance tests by additions and logic only

- **Idea:** a calibration of the loop's instructions showed where its time goes: the vector part alone, without taking the bins out or counting, runs at 4.8 cycles per block of four where its instruction count needs 2.5. A compare or a minimum of 32-bit lanes runs on two of the three vector ports, the same two the multiply and the conversion need. Both tests can leave their verdict in a sign bit using additions and logic, which run on all three: `((z + 1) & 0xffe) - 1` is negative exactly when the fraction bits are all zero or all one, and `(limit - z) | z` is negative exactly when z is negative or above the limit.
- **check:** pass; differential test 30,000 rounds, 106 million elements, no mismatch
- **compare:** significantly faster, 1.009x [1.003 – 1.015]
- **Kept:** yes
- **Notes:** 1.02x to 1.03x from 1,024 to 4,194,304, 0.97x at 16 and 64. Smaller than the calibration promised.

## Where it stands

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
