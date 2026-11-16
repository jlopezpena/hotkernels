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
