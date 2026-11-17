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

## 5. Post-indexed stores, lowest pair last

- **Idea:** save the separate `add` on the output pointer by writing the upper vectors first and the lowest pair last with a post-indexed `stp`, which is the store order clang uses in its own loop.
- **check:** pass
- **compare:** significantly slower, 0.834x [0.831 – 0.837]
- **Kept:** no
- **Notes:** 64 to 1,024 gained 1 to 3%, but every size from 65,536 up dropped to 0.62 to 0.71x. Once the output no longer fits the first-level cache, the stores have to reach each cache line in ascending address order. Checked locally with the output placed at each offset within a cache line, 65,536 to 1,048,576 elements, quiet machine: stores in ascending order run at 0.14 to 0.17 cycles per element at every offset; this attempt's order runs at 0.23 to 0.27 at offsets 0, 32 and 48; and clang's order for `naive.c` (upper pair, then lower pair) is as fast as ascending when the output is 64-byte aligned but runs at 0.24 to 0.25 at offsets 16, 32 and 48. The judge's buffers at these sizes are page-aligned, so it measures `naive.c` in its best case and this difference does not show in its numbers.

## Where it stands

- In the first-level cache the loop is limited by its stores: 64 bytes of freshly computed data every 2.5 cycles. A loop with the four `tbl` and no `ucvtf`, one with `ushll` widening (ten vector operations instead of eight) and one that stores a loaded vector four times with no vector operations at all run at that same speed, and so does the system `memmove`. Clang's loop is already there.
- From 65,536 to 4,194,304 elements the stores are still the limit: `ldp` and `ldnp` time the same, `stnp` changes nothing, and `dc zva` on the output is far slower (see `clip_f32` attempt 3).
- What assembly gains over `naive.c` under the judge is therefore at the ends: n = 16 (no overlap test, two constant loads instead of four, no vector-width dispatch), n = 64 slightly, and 16.7M where `ldnp` helps.

## Result

`uv run judge run cast_u8_f32 --unlocked`, `kernel.s` as of commit 3101e7f, Apple M1 Pro, other work running on the machine at the time:

| n | kernel | naive C | speedup [95% interval] |
|---|---|---|---|
| 16 | 1.6 ns | 2.2 ns | 1.40x [1.40 – 1.40] |
| 64 | 3.6 ns | 4.0 ns | 1.13x [1.10 – 1.15] |
| 256 | 13.4 ns | 13.2 ns | 0.99x [0.97 – 1.00] |
| 1,024 | 51.6 ns | 51.9 ns | 1.01x [0.98 – 1.02] n.s. |
| 4,096 | 205.2 ns | 207.7 ns | 1.01x [1.00 – 1.02] |
| 16,384 | 791.2 ns | 795.0 ns | 1.00x [0.98 – 1.04] n.s. |
| 65,536 | 2.8 us | 2.8 us | 1.00x [0.99 – 1.15] n.s. |
| 262,144 | 23.3 us | 23.3 us | 1.00x [0.89 – 1.08] n.s. |
| 1,048,576 | 74.4 us | 78.1 us | 1.05x [0.90 – 1.64] n.s. |
| 4,194,304 | 276.6 us | 276.5 us | 1.00x [0.86 – 1.00] n.s. |
| 16,777,216 | 1.17 ms | 1.35 ms | 1.16x [1.15 – 1.17] |
| overall | | | 1.06x [1.04 – 1.11] |

Assembly is kept, because the overall figure is significantly above 1.0 (1.06x here, 1.06x [1.01 – 1.07] in the run before it), but it is a narrow win: three sizes gain (16, 64 and 16.7M) and the eight between are within about 1% of `naive.c`. In `judge run` the times from 262,144 to 4,194,304 are about twice those of `judge compare` for kernel and `naive.c` alike (11.6 us at 262,144 in `compare`). The likely cause is the NumPy contender timed between the native samples: in a standalone test, a NumPy call or a 10 ms sleep between samples slows the next sample in the same way. The ratio is not affected.
