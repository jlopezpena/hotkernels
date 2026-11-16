# argmin_f64 attempts

## 1. Scalar baseline

- **Idea:** simplest correct version: one element per step, `fcmp` for NaN and for strictly smaller.
- **check:** pass
- **compare:** n/a (first version)
- **Kept:** yes
- **Notes:** starting point only; same shape as naive C.

## 2. Port of the tuned argmax design

- **Idea:** the argmax_f64 kernel with the comparison reversed (see `kernels/argmax_f32/attempts.md` for how the design was reached and `kernels/argmax_f64/attempts.md` for the two-lane port): `fmin`/`fminp` instead of `fmax`/`fmaxp`, and `csel ... mi` (strictly less) for the best-block pointer. Above 64 elements: 32-element blocks, 15 `fmin` and one scalar `fminp` per block, one `fcmp` driving `b.vs` (NaN in this block), `csel` and a scalar `fmin`; at the end only the best block is rescanned. Up to 64 elements: straight-line code over a front half and an overlapping back half; two-step search through 8-element "quads" above 16. Below 4 elements: scalar.
- **check:** pass
- **compare:** significantly faster, 7.43x [7.38 – 7.47]
- **Kept:** yes
- **Notes:** timings match argmax_f64 size for size. Lessons carried over from argmax_f32: software prefetch hurts (its attempt 5); a deeper two-step rescan after the block loop is slower because that phase is paid for by its latency (9); lane-wise block-number tracking is slower than scalar `csel` tracking (10).

## 3. Two-step search for 8 < n <= 16

- **Idea:** same change as argmax_f64 attempt 3: compare the four pair minima the tree already produces, then reload and compare only the two vectors of the first pair that holds the minimum (22 vector operations instead of 25).
- **check:** pass
- **compare:** significantly faster in three of five runs: 1.005x [1.004 – 1.005], 1.006x [1.004 – 1.007], 1.006x [1.001 – 1.009]. The other two were not significant, with wide intervals from noise at the large sizes: 1.004x [1.000 – 1.012] and 0.992x [0.978 – 1.029].
- **Kept:** yes
- **Notes:** n = 16 was 1.06x in all five runs (2.6 -> 2.4 ns) and no other size is touched by the change. The overall gain is small enough that noise at 4M and 16M can hide it; kept because the per-size effect is stable and the same change passed twice out of twice on argmax_f64.
