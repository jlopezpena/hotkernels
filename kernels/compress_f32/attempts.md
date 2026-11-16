# compress_f32 attempts (arm64)

All judge commands in this run were passed `--unlocked`, because the spec is not on `main` yet. Other agents were timing kernels on the same machine, so intervals are wider than usual.

## 1. Scalar baseline

- **Idea:** the naive loop in assembly: store every element at `out[k]`, advance `k` by the mask byte.
- **check:** pass
- **compare:** not run (first version, nothing to compare against)
- **Kept:** yes
- **Notes:** starting point only; expected to match naive C.

## 2. Eight elements per step with a nibble shuffle table

- **Idea:** load 8 mask bytes as a u64; one multiply packs them into an 8-bit mask, two more give the selected counts (scaled to bytes) for all eight and for the low four. Each nibble indexes a 16-entry table of `tbl` patterns; each 4-lane half is compacted and stored whole at the write pointer. The write-pointer chain is one add per 8 elements. Scalar tail for the last 0 to 7.
- **check:** pass
- **compare:** significantly faster, 2.50x [2.48 – 2.50]
- **Kept:** yes
- **Notes:** half_selected 2.45x, few_selected 2.55x; the code is branchless so the two distributions run the same. About 0.125 ns per element in cache (roughly 3.2 cycles per 8 elements).

## 3. Cheaper mask and count extraction

- **Idea:** take both counts from one multiply (`m * 0x0404…04` leaves 4x the running count in every byte: `ubfx` byte 3 and `lsr` byte 7) and take the two nibbles straight from the mask product (`ubfx #56,#4` and `lsr #60`). Two instructions fewer per step, one multiply fewer.
- **check:** pass
- **compare:** significantly faster, 1.09x [1.09 – 1.09]
- **Kept:** yes
- **Notes:** the loop is limited by instruction count, not by a dependency chain: removing 2 of about 22 instructions gave 9%.
