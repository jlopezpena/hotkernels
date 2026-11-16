# argmax_f32 attempts

## 1. Scalar baseline

- **Idea:** simplest correct version: one element per step, `fcmp` for NaN and for strictly greater.
- **check:** pass
- **compare:** n/a (first version)
- **Kept:** yes
- **Notes:** starting point only; same shape as naive C.

## 2. Branch-free block maxima with per-lane block-number tracking

- **Idea:** `fmax` propagates NaN, so reduce 64-element blocks to four lane maxima (15 `fmax`), and per lane keep the running maximum and the number of the first block that reached it (`fcmgt` + `bit`, no branch). At the end rescan only the winning block: compare with the maximum, narrow the compare results to a 64-bit mask (`uzp1` x12, `and` with bit weights, `addp` x4), `rbit` + `clz`. NaN is tested once per 1,024 blocks; a scalar loop then finds the first NaN. Inputs of 8 to 64 elements use straight-line code over a front half and an overlapping back half; below 8 stays scalar. A tail that is not a whole block is one extra overlapping block ending at `n`.
- **check:** pass
- **compare:** significantly faster, 13.3x [13.22 – 13.29]
- **Kept:** yes
- **Notes:** 0.028 ns per element while the data fits the first-level cache (16,384), 0.047 to 0.051 in the second level, 0.069 at 16.7M. No data-dependent branch, so ascending data costs the same as random. Chosen over "compare block maximum with running best, rescan on a hit" because that needs the same number of vector operations per block plus a mispredicted branch per hit.

## 3. Broadcast the maximum with two `fmaxp` instead of `fmaxv` + `dup`

- **Idea:** two pairwise maxima leave the overall maximum in every lane, one vector operation fewer than reducing to a scalar and duplicating it.
- **check:** pass
- **compare:** no significant difference, 1.002x [0.998 – 1.004]
- **Kept:** no
- **Notes:** not significantly faster at any size.

## 4. Two-step search for 32 < n <= 64

- **Idea:** the 64-element path was limited by vector operations (55 for 16 loads). The maximum tree already passes through four "quads" (lane maxima of 16 elements), so compare those with the maximum to find the first quad that holds it, then reload and compare only that quad's four vectors: 18 search operations instead of 36.
- **check:** pass
- **compare:** significantly faster, 1.021x [1.017 – 1.032]
- **Kept:** yes
- **Notes:** n = 64 goes from 5.3 ns to 4.0 ns (1.33x); nothing else changes.
