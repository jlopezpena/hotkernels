# cast_u8_f32 attempts

## 1. `tbl` + `ucvtf`, 16 bytes per iteration, overlapping tail

- **Idea:** the same loop body clang emits for `naive.c` (one 16-byte load, four `tbl` to spread the bytes into 32-bit lanes, four `ucvtf`, four stores), but with no overlap test at entry, two constant loads instead of four `adrp`/`ldr` pairs, and no scalar remainder: a leftover of 1 to 15 bytes is one more block ending exactly at the end, 4 to 15 bytes are a front piece and an overlapping back piece, and below 4 is a scalar loop.
- **check:** pass
- **compare:** n/a (first version)
- **Kept:** yes
- **Notes:** starting point.

## 2. 32 bytes per iteration, loaded with `ldnp`, for every n >= 16

- **Idea:** `ldnp` (a non-temporal load pair, the load the system `memmove` uses) is what made `clip_f32` faster beyond the first-level cache. It only exists as a pair, so the loop takes 32 bytes and writes eight vectors per iteration; 16 to 31 leftover bytes go through one 16-byte block.
- **check:** pass
- **compare:** no significant difference, 1.004x [0.991 – 1.017]
- **Kept:** no
- **Notes:** 16,777,216 gained (1.17x) but n = 16 lost (1.6 ns to 1.9 ns, 0.83x): it now passed two extra compare-and-branch pairs before reaching its single block. Everything between was within noise.

## 3. The same, entered with one test so n = 16 runs the old code

- **Idea:** keep the 16-byte loop of attempt 1 untouched and branch to the 32-byte `ldnp` loop when n >= 32.
- **check:** pass
- **compare:** not reliably faster. Three runs: 1.010x [0.979 – 1.043] not significant, 1.014x [0.997 – 1.062] not significant, 1.018x [1.013 – 1.033] significant.
- **Kept:** no
- **Notes:** 16,777,216 was 1.17 to 1.19x in every run and 256 to 4,096 were 1.01 to 1.02x, but n = 64 was 0.97 to 0.99x in every run: two iterations of the 32-byte loop plus its exit tests cost more than four of the 16-byte loop.

## 4. 32-byte `ldnp` loop from 128 elements

- **Idea:** as attempt 3 with the threshold raised so n = 64 also keeps the 16-byte loop.
- **check:** pass
- **compare:** significantly faster in three runs out of three: 1.031x [1.002 – 1.035], 1.018x [1.001 – 1.026], 1.017x [1.009 – 1.040]
- **Kept:** yes
- **Notes:** the whole gain is at 16,777,216 (1.16 to 1.18x in each run), where the input no longer fits the second-level cache, plus about 1% from 256 to 4,096. Local timing with the two variants interleaved shows what `ldnp` does here: no difference from 4,096 to 4,194,304 elements (the same loop with `ldp` is equally fast there), 0.216 instead of 0.255 cycles per element at 16.7M and 0.211 instead of 0.260 at 67M. Unlike `clip`, the sizes in the second-level cache do not move, because the output is four times the size of the input and the stores are the limit there.
