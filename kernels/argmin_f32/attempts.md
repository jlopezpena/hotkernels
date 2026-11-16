# argmin_f32 attempts

## 1. Scalar baseline

- **Idea:** simplest correct version: one element per step, `fcmp` for NaN and for strictly smaller.
- **check:** pass
- **compare:** n/a (first version)
- **Kept:** yes
- **Notes:** starting point only; same shape as naive C.

## 2. Port of the tuned argmax_f32 design

- **Idea:** the final argmax_f32 kernel with the comparison reversed (see `kernels/argmax_f32/attempts.md` for how it was reached): `fmin`/`fminv` instead of `fmax`/`fmaxv`, `csel ... mi` (strictly less) for the best-block pointer, and the running minimum starts from `x[0]` instead of an infinity constant. Above 64 elements: 64-element blocks, 15 `fmin` and one `fminv` per block, one `fcmp` driving `b.vs` (NaN in this block), `csel` and a scalar `fmin`; at the end only the best block is rescanned. Up to 64 elements: straight-line code over a front half and an overlapping back half; two-step search through "quads" above 32. Below 8 elements: scalar.
- **check:** pass
- **compare:** significantly faster, 14.1x [14.04 – 14.13]
- **Kept:** yes
- **Notes:** timings match argmax_f32 size for size (0.027 ns per element in the first-level cache, 0.049 in the second level, about 0.07 at 16.7M). Lessons carried over from argmax_f32: `fmaxp` broadcast instead of `fmaxv` + `dup` makes no difference (its attempt 3); software prefetch hurts (5); a deeper two-step rescan after the block loop is slower because that phase is paid for by its latency (9); lane-wise block-number tracking is slower than scalar `csel` tracking (10).
