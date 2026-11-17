# cast_u8_f32 attempts

## 1. `tbl` + `ucvtf`, 16 bytes per iteration, overlapping tail

- **Idea:** the same loop body clang emits for `naive.c` (one 16-byte load, four `tbl` to spread the bytes into 32-bit lanes, four `ucvtf`, four stores), but with no overlap test at entry, two constant loads instead of four `adrp`/`ldr` pairs, and no scalar remainder: a leftover of 1 to 15 bytes is one more block ending exactly at the end, 4 to 15 bytes are a front piece and an overlapping back piece, and below 4 is a scalar loop.
- **check:** pass
- **compare:** n/a (first version)
- **Kept:** yes
- **Notes:** starting point.
