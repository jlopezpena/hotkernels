# clip_f64 attempts

The design was worked out on `clip_f32` first; see `kernels/clip_f32/attempts.md` for the experiments behind each step. The entries here are the same steps measured again for float64.

## 1. Compare and select, 8 elements per iteration, overlapping tail

- **Idea:** the same four operations per vector that clang emits for `naive.c` (two `fcmgt`, two `bit`), four vectors per iteration, with no entry checks and no scalar remainder: a leftover of 1 to 7 elements is one more block ending exactly at the end, 2 to 7 elements are a front piece and an overlapping back piece, and a single element is scalar `fcmp`/`fcsel`.
- **check:** pass
- **compare:** n/a (first version)
- **Kept:** yes
- **Notes:** starting point. Bit-exact for any bounds by construction: nothing is arithmetic, so NaN elements, NaN bounds, signed zeros and `lo > hi` all follow the C expression.
