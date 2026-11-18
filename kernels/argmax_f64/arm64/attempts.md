# argmax_f64 attempts (arm64)

## 1. Scalar baseline

- **Idea:** simplest correct version: one element per step, `fcmp` for NaN and for strictly greater.
- **check:** pass
- **compare:** n/a (first version)
- **Kept:** yes
- **Notes:** starting point only; same shape as naive C.

## 2. Port of the tuned argmax_f32 design

- **Idea:** the final argmax_f32 kernel with two 64-bit lanes per vector (see `kernels/argmax_f32/attempts.md` for how it was reached). Above 64 elements: 32-element blocks, 15 `fmax` and one scalar `fmaxp` per block, one `fcmp` against the running maximum driving `b.vs` (NaN in this block), `csel` (pointer to the best block, strictly greater only) and a scalar `fmax`; at the end only the best block is rescanned (16 `fcmeq`, `uzp1` narrowing, bit weights, `addp`, then `rbit` + `clz`). Up to 64 elements: straight-line code over a front half and an overlapping back half; from 17 elements the search takes two steps through 8-element "quads". Below 4 elements: scalar.
- **check:** pass
- **compare:** significantly faster, 7.06x [7.03 – 7.30] (machine reported as noisy; the margin is far outside the noise)
- **Kept:** yes
- **Notes:** 0.057 ns per element while the data fits the first-level cache, about 0.10 in the second level, the same bytes-per-second as the float32 kernel. Lessons carried over from argmax_f32: software prefetch hurts (its attempt 5); the phase after the block loop is paid for by its latency, so a deeper two-step rescan there is slower (its attempt 9); lane-wise block-number tracking with `uminv` at the end is slower than the scalar `csel` tracking (its attempt 10).

## 3. Two-step search for 8 < n <= 16

- **Idea:** the 16-element path compared all 8 vectors (25 vector operations). Compare the four pair maxima the tree already produces, then reload and compare only the two vectors of the first pair that holds the maximum: 22 vector operations and a shorter narrowing.
- **check:** pass
- **compare:** significantly faster, 1.006x [1.005 – 1.007] (first run, on a noisy machine: 1.006x [1.002 – 1.010])
- **Kept:** yes
- **Notes:** n = 16 goes from 2.5–2.7 ns to 2.4–2.5 ns (1.06x in both runs); nothing else changes. A probe outside the judge showed the opposite change for 32 < n <= 64 (sending it to the block loop instead of the two-step quad search) costs 6.3 -> 9 ns at n = 64, so that class stays as it is.

## Final result (after attempt 3)

`uv run judge run argmax_f64 --unlocked`: check PASS (18,081 cases). Overall 12.1x [11.89 – 12.19] over naive C and 16.6x [16.38 – 16.79] over NumPy. Other kernels were being timed on the machine at the same time, which widens the intervals at 4M and 16M.

| n | kernel | naive C | speedup | NumPy | speedup |
|---|---|---|---|---|---|
| 16 | 2.5 ns | 10.2 ns | 4.10x [4.06 – 4.14] | 609.4 ns | 246x [243 – 248] |
| 64 | 6.4 ns | 54.4 ns | 8.45x [8.38 – 8.49] | 666.7 ns | 104x [102 – 104] |
| 256 | 19.1 ns | 300.0 ns | 15.7x [15.48 – 15.84] | 791.8 ns | 41.4x [40.82 – 41.70] |
| 1,024 | 61.5 ns | 1.3 us | 21.3x [21.13 – 21.54] | 1.3 us | 21.8x [21.58 – 22.11] |
| 4,096 | 233.4 ns | 5.4 us | 23.0x [22.73 – 23.12] | 3.5 us | 14.9x [14.77 – 15.02] |
| 16,384 | 924.7 ns | 21.6 us | 23.3x [23.12 – 23.53] | 12.5 us | 13.5x [13.29 – 13.58] |
| 65,536 | 6.5 us | 86.5 us | 13.4x [13.19 – 13.53] | 47.2 us | 7.28x [7.18 – 7.38] |
| 262,144 | 25.9 us | 342.6 us | 13.2x [12.92 – 13.47] | 184.7 us | 7.14x [6.98 – 7.27] |
| 1,048,576 | 117.9 us | 1.36 ms | 11.6x [11.19 – 11.87] | 728.1 us | 6.17x [6.02 – 6.37] |
| 4,194,304 | 699.7 us | 5.44 ms | 7.77x [7.48 – 8.49] | 2.94 ms | 4.21x [4.04 – 4.58] |
| 16,777,216 | 2.70 ms | 21.52 ms | 7.98x [7.06 – 8.50] | 11.55 ms | 4.28x [3.80 – 4.56] |

Per byte this matches argmax_f32 (0.056 ns per element in the first-level cache, about 0.10 in the second level), so the large sizes are limited by memory in the same way.

Not measured by the judge, timed outside it (best of 7, one input): ascending, descending and all-equal inputs take the same time as random ones (about 900 ns at 16,384 in all four cases). A NaN as the first element returns in about 3 ns at any size; a NaN as the last element costs 25 ns at 256 and 919 ns at 16,384.

Differences from the float32 kernels, all because of which sizes the judge times: this kernel has a two-step search for 9..16 (timed at 16) and an eight-quad class for 33..64 (timed at 64); the float32 kernels instead have a sixteen-quad class for 129..256 (timed at 256). The matching classes for the untimed ranges were not added, since the rule cannot keep a change the judge cannot time.
