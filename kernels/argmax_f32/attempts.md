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

## 5. Software prefetch in the block loop

- **Idea:** two `prfm pldl1keep` per 256-byte block, 1,024 bytes ahead, to help once the data no longer fits the first-level cache.
- **check:** pass
- **compare:** significantly slower, 0.949x [0.943 – 0.956]
- **Kept:** no
- **Notes:** 8 to 13% slower from 4,096 to 1,048,576 (the prefetches take load slots, and the hardware prefetcher already follows the stream); no measurable gain at 4M and 16M.

## 6. Test for n <= 16 first at entry

- **Idea:** n = 16 runs at about 43 operations per call including the caller's loop, so it is limited by how many operations the core can issue per cycle. Checking the smallest class first removes two compare-and-branch pairs from that path at the cost of one more for larger inputs.
- **check:** pass
- **compare:** significantly faster, 1.007x [1.002 – 1.010]
- **Kept:** yes
- **Notes:** n = 16 goes from 1.8 ns to 1.7 ns (1.10x); other sizes unchanged.

## 7. One operation fewer in the index arithmetic for 8 <= n <= 16

- **Idea:** select the offset (0 or n-16) with `csel` and fold the divide-by-four into the final `add`, saving one integer operation on the n = 16 path.
- **check:** pass
- **compare:** not significantly faster. Three runs: 1.004x [1.001 – 1.007], then 1.002x [0.997 – 1.007] and 1.002x [0.998 – 1.010].
- **Kept:** no
- **Notes:** n = 16 itself was 2 to 3% faster in every run, but the overall figure did not hold up when repeated.

## 8. Vector search for the first NaN on the large path

- **Idea:** when a super-block reports a NaN, the scalar loop walks up to 65,536 elements to find it. Step over whole blocks with the `fmax` tree first and run the scalar loop only inside the block that holds the NaN.
- **check:** pass
- **compare:** no significant difference, 0.999x [0.997 – 1.001]
- **Kept:** no
- **Notes:** the judge times random data without NaN, so it cannot see this path, and the rule says not to keep it. Measured outside the judge (NaN as the last element, best of 7): 16,384 elements 5,711 ns -> 898 ns; 1,048,576 elements 72.2 us -> 52.7 us. Worth keeping if NaN inputs ever become a timed distribution.

## 9. Two-step rescan of the winning block on the large path

- **Idea:** same trick as attempt 4 for the final rescan: rebuild the block's four quads (12 `fmax`), find the first quad that holds the maximum, then compare only its four vectors. 30 vector operations instead of 37, at the cost of 4 more loads.
- **check:** pass
- **compare:** significantly slower, 0.962x [0.960 – 0.964]
- **Kept:** no
- **Notes:** 256: 13.0 -> 15.0 ns, 1,024: 35.7 -> 43.2 ns, 4,096: 122.7 -> 132.5 ns. The extra cost is about 24 cycles per call from 1,024 up, which matches the extra length of the dependency chain (quad tree, a second move to a general register, a second load), not the operation count. Lesson: after the block loop the final phase is paid for by its latency, so it should be kept short and shallow, not merely small.

## 10. Track the best block with a scalar compare and `csel`, NaN test per block

- **Idea:** attempt 9 showed the final phase costs its latency. A probe outside the judge that skipped the lane/block selection (wrong answers, timing only) was 13 cycles faster at 1,024. So reduce every block to one scalar (`fmaxv`), `fcmp` it against the running maximum, and let that one compare drive `b.vs` (NaN in this block), `csel` (pointer to the best block, strictly greater only) and a scalar `fmax`. The best block's address then sits in a general register before the loop ends, and the selection chain (`fcmeq`, `orn`, `uminv`, `fmov`) disappears. Same number of vector operations per block as the lane-wise tracking it replaces.
- **check:** pass
- **compare:** significantly faster, 1.016x [1.008 – 1.021] (a first, noisy run gave 1.020x [0.971 – 1.024])
- **Kept:** yes
- **Notes:** 1,024: 36.2 -> 33.0 ns (1.10x), 4,096 and 16,384: 1.04x, 256: 1.02x; no change once the data is outside the first-level cache. Side effects: the super-block loop is gone, a NaN is now noticed within its own 64-element block, and block numbers no longer limit n.

## 11. Reduce each block to a scalar with two `fmaxp` instead of `fmaxv`

- **Idea:** `fmaxp` on the vector, then scalar `fmaxp` on the low pair, in case `fmaxv` is the slower way to fold four lanes.
- **check:** pass
- **compare:** no significant difference, 1.002x [0.994 – 1.006]
- **Kept:** no
- **Notes:** mixed: 256 was 6% faster (12.8 -> 12.1 ns) but 1,024 and 4,096 were 3 to 4% slower. The change is one instruction, so part of this is probably code placement.

## 12. Align the block loop to 64 bytes

- **Idea:** check whether the mixed result of attempt 11 was code placement.
- **check:** pass
- **compare:** no significant difference, 1.002x [0.994 – 1.006]
- **Kept:** no
- **Notes:** no size moved, so placement is not what attempt 11 saw.

## 13. Peel the block that ends at n and give only it the two-`fmaxp` reduction

- **Idea:** if attempt 11 helped 256 through latency (the rescan waits for the last block's result), keep `fmaxv` in the loop and use the pairwise form only for the last block. The loop then covers every whole block except one ending at n, and the block ending at n always runs separately, which also removes the tail flag.
- **check:** pass
- **compare:** no significant difference, 0.998x [0.994 – 1.004]
- **Kept:** no
- **Notes:** 256 gained only 1%, so the 6% in attempt 11 came from using the pairwise form in every block, not from the last block's latency. Not understood; not pursued, since the same form costs 3 to 4% at 1,024 and 4,096.

## 14. Shallower end of the rescan: four nibble masks and a `csel` tree

- **Idea:** probes outside the judge (timed like the judge, cycling through a pool of inputs) split the rescan's cost at 256 into about 10 cycles of work and 4 cycles of waiting for the best-block pointer, so latency after the loop looked worth buying. Replace `and` + three levels of `addp` + one `fmov` by `shrn #4` on each of the four byte vectors, four `fmov` to general registers, and a `cmp`/`csel` tree that picks the first non-empty group before `rbit` + `clz`. One vector operation and the constant load fewer, eleven integer operations more, a shorter chain on paper.
- **check:** pass
- **compare:** significantly slower, 0.994x [0.989 – 1.000]
- **Kept:** no
- **Notes:** 256: 13.2 -> 13.6 ns, 4,096: 117.3 -> 122.8 ns. Four vector-to-integer moves cost more than the `addp` levels they replace. The single-`fmov` bit mask stays.

## 15. Straight-line path for 128 < n <= 256 over sixteen quads

- **Idea:** on argmax_f64 the dedicated two-step path for 64 doubles beats the block loop by 30% (6.3 ns against 9 ns, measured outside the judge). The same bytes for float32 are 129..256 elements: build sixteen quad maxima (eight from the start, eight ending at n), take the maximum, find the first quad that holds it from one 64-bit mask, then compare that quad's 16 elements. 68 loads and 111 vector operations instead of 80 and 114, and no per-block scalar work.
- **check:** pass
- **compare:** significantly faster, 1.006x [1.004 – 1.008], repeated: 1.005x [1.004 – 1.005]
- **Kept:** yes
- **Notes:** n = 256 goes from 13.2 to 12.5 ns (1.05x, 1.04x on the repeat); no other timed size changes. Only 256 is timed in this range, so the gain for 129..255 is not measured, only checked for correctness. 65..128 still uses the block loop; a matching eight-quad class would be the natural next step but no timed size falls in it.
