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
