# clip_f32 attempts

## 1. Compare and select, 16 elements per iteration, overlapping tail

- **Idea:** the same four operations per vector that clang emits for `naive.c` (two `fcmgt`, two `bit`), four vectors per iteration, but with no entry checks and no scalar remainder: a leftover of 1 to 15 elements is one more block ending exactly at the end, and 4 to 15 elements are a front piece and an overlapping back piece. Below 4 is a scalar `fcmp`/`fcsel` loop.
- **check:** pass
- **compare:** n/a (first version)
- **Kept:** yes
- **Notes:** starting point. Bit-exact for any bounds by construction: nothing is arithmetic, so NaN elements, NaN bounds, signed zeros and `lo > hi` all follow the C expression.
