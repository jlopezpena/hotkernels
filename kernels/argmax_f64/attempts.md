# argmax_f64 attempts

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
