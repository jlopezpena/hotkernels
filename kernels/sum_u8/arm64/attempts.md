# sum_u8 attempts (arm64)

**Verdict on ARM64: assembly kept.** Final `judge run`: 5.83x [5.82 – 5.84] faster than naive C overall
(3.3x at 16 bytes, 8.2–9.0x from 1,024 to 65,536, 5.0x from 262,144 to 4,194,304, 4.4x at 16,777,216).

All runs on an Apple M1 Pro with `--unlocked` (the spec was not on `main` yet), while other
agents were timing other kernels on the same machine, so absolute timings are noisier than usual.

What the compiler does with `naive.c` (`clang -O3 -mcpu=native`, Apple clang 21): it widens every
byte to 64 bits before adding. Each 32-byte step is eight `tbl` (to spread the bytes into 32-bit
lanes) and sixteen `uaddw`/`uaddw2` into sixteen 64-bit accumulators: 24 vector instructions on
four vector ports, six cycles per 32 bytes. That is 0.059 ns per byte (about 17 GB/s) at every
size, well below what the caches or memory can deliver, so there is room at every size.

## 1. Baseline: `uaddlv` per 16 bytes, scalar tail

- **Idea:** simplest vector version. One 16-byte load per step, `uaddlv` adds its sixteen bytes to one 16-bit value, moved to a general register and added to the total. Leftover bytes one at a time.
- **check:** pass
- **compare:** n/a (first version). `run`: 2.43x [2.43 – 2.43] vs naive C overall; 2.75x at 16, 2.33x at 64, 2.2–2.9x above.
- **Kept:** yes
- **Notes:** 0.026 ns per byte at large sizes, about 2.4 cycles per 16 bytes: one `uaddlv` and one vector-to-general move per step.

## 2. `uadalp` into 16-bit accumulators, sixteen vectors per step

- **Idea:** `uadalp acc.8h, v.16b` adds a whole vector of bytes into 16-bit lanes in one instruction, so a vector costs one load and one vector operation instead of the compiler's twelve. A lane grows by at most 510 per step, so widen to a 64-bit total every 128 steps. Sixteen accumulators and 256 bytes per step from 2,048 bytes; a lean path below that (eight accumulators, 128 bytes per step, then 64, 32, 16 and a final vector that overlaps the previous one and is masked from a constant table) whose accumulators can be added as 16-bit lanes without overflow; general registers only below 16 bytes.
- **check:** pass
- **compare:** significantly faster, 2.182x [2.181 – 2.182]
- **Kept:** yes
- **Notes:** 3.4–3.9x over the baseline from 4,096 to 65,536, 2.2–2.3x from 262,144 to 4,194,304, 1.7x at 16,777,216, but 0.80x at 16 (three taken branches before `ret`). A private micro-benchmark chose the loop shape, in cycles per 16 bytes inside the L1 cache: four accumulators 0.75 (so `uadalp` takes three cycles from accumulator to accumulator), eight accumulators 0.37, sixteen accumulators 0.333, which is exactly three 16-byte loads per cycle. Beyond L1 every variant ran at the same 0.59 (L2) and 0.79 (16 MB), so those sizes are bound by the caches and memory, not by the loop. Also checked privately, because the judge has no all-0xFF value class: every n up to 70,000 and nine sizes up to 40 MB on all-0xFF data give n * 255.

## 3. Lay out the short paths for few taken branches; four accumulators below 128 bytes

- **Idea:** at 16 and 64 bytes the call costs about one cycle per taken branch. Make n = 16 fall through to `ret`, move the 32-, 16- and leftover-byte blocks out of line so an absent block costs no taken branch, repeat the final reduction after the 128-byte loop, and use four accumulators (three `add` to reduce) when fewer than 128 bytes are left.
- **check:** pass
- **compare:** significantly faster, 1.079x [1.079 – 1.079]
- **Kept:** yes
- **Notes:** 1.51x at 16 (1.6 ns to 1.0 ns), 1.31x at 64, 1.10x at 256, 1.03x at 1,024, unchanged above.

## 4. Two accumulators below 128 bytes; end pointer only where it is needed

- **Idea:** fewer instructions on the short paths: two accumulators instead of four when fewer than 128 bytes are left (one `add` to reduce instead of three, two registers to zero instead of four), and the end-of-array pointer computed only in the leftover-bytes block.
- **check:** pass
- **compare:** significantly faster, 1.011x [1.011 – 1.011]
- **Kept:** yes
- **Notes:** 1.06x at 64 and 1.05x at 256, nothing elsewhere. At 16 the call is at the floor of three taken branches (call, return, loop back-edge of the timing harness): 1.0 ns, 3.2 cycles.

## 5. Software prefetch to L1, close ahead, in the 256-byte loop (measured privately, never run through `compare`)

- **Idea:** the sizes beyond L1 run at 0.59 cycles per 16 bytes (L2) and 0.79 (16 MB) whatever the loop does; `prfm pldl1keep` 256 to 4,096 bytes ahead might raise that.
- **check:** not run (private micro-benchmark with its own result check)
- **compare:** not run. Micro-benchmark, cycles per 16 bytes, no prefetch against the best prefetch distance: 0.333 against 0.417 at 16,384 and 65,536 bytes; 0.59–0.60 against 0.67 from 262,144 to 4,194,304; 0.788 against 0.803 at 16,777,216; 0.854 against 0.852 at 67,108,864.
- **Kept:** no
- **Notes:** slower wherever it changes anything: four more instructions per step in cache, and the hardware prefetcher already keeps up with a single forward stream. Dead end.

## 6. Two streams, front half and back half at once (measured privately, never run through `compare`)

- **Idea:** if the transfer rate from L2 is limited per stream, reading the two halves of the array in parallel might double it.
- **check:** not run (private micro-benchmark with its own result check)
- **compare:** not run. Micro-benchmark, cycles per 16 bytes, one stream against two: 0.600 against 0.590 at 262,144; 0.590 against 0.585 at 1,048,576; 0.598 against 0.597 at 4,194,304; 0.788 against 0.790 at 16,777,216; 0.333 against 0.339 in L1.
- **Kept:** no
- **Notes:** within 1–2% everywhere, inside the noise of a loaded machine, and it would complicate the tails. The L2 rate (about 27 bytes per cycle) is not a per-stream limit.

## 7. `ldnp` in the 256-byte loop

- **Idea:** `ldnp` is the load pair that hints the data will not be needed again soon. A private micro-benchmark of the count_nonzero loop showed it 3–6% faster than `ldp` once the array no longer fits in the L2 cache and identical while it does; here only the 16 MB size is beyond L2.
- **check:** pass
- **compare:** significantly faster, 1.009x [1.008 – 1.011]
- **Kept:** yes
- **Notes:** 1.11x at 16,777,216 (259 us to 233 us), no significant difference at any other size.

## 8. Prefetching copy of the loop from 16 MB

- **Idea:** a second copy of the 256-byte loop for n >= 2^24 that also issues two `prfm pldl2keep` per step, 256 KB ahead. In the private micro-benchmark that was 6–8% faster than `ldnp` alone at 16, 32 and 128 MB, and 3–5% slower at 8 MB and below (hence the threshold).
- **check:** pass
- **compare:** no significant difference, 1.002x [0.998 – 1.003]; rerun once because 262,144 showed 0.98x from code that had not changed: significantly faster, 1.001x [1.000 – 1.003]
- **Kept:** no
- **Notes:** the only size it can touch, 16,777,216, came out 1.04x in the first run and 1.01x in the second. At exactly 16 MB most of the array still sits in the 12 MB L2 cache, so there is little for a prefetch to do; the judge has no larger size. One compare of two is not a gain worth a second loop. It did pay for the count_nonzero kernels, whose two largest sizes are 32 MB and 128 MB.

## Where it stands

Final `judge run` (30 samples per point, machine under load from other agents' timing runs; the
first run of the final version printed the judge's noisy-machine warning, this is the rerun):

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 1.0 ns | 3.4 ns | 3.33x [3.33 – 3.33] | 1.3 us | 1227x [1223 – 1229] |
| 64 | 1.6 ns | 6.5 ns | 4.20x [4.20 – 4.21] | 1.3 us | 829x [826 – 830] |
| 256 | 2.9 ns | 17.7 ns | 6.00x [6.00 – 6.01] | 1.3 us | 438x [437 – 439] |
| 1,024 | 7.6 ns | 62.4 ns | 8.16x [8.16 – 8.17] | 1.5 us | 195x [195 – 195] |
| 4,096 | 28.5 ns | 252.4 ns | 8.86x [8.74 – 8.89] | 2.1 us | 73.5x [72.42 – 73.79] |
| 16,384 | 107.1 ns | 960.2 ns | 8.96x [8.96 – 8.97] | 4.3 us | 40.5x [40.40 – 40.60] |
| 65,536 | 435.3 ns | 3.8 us | 8.77x [8.76 – 8.78] | 13.3 us | 30.5x [30.44 – 30.51] |
| 262,144 | 3.0 us | 15.3 us | 5.03x [5.01 – 5.04] | 48.4 us | 16.0x [15.91 – 15.97] |
| 1,048,576 | 12.0 us | 61.0 us | 5.07x [5.00 – 5.07] | 189.1 us | 15.7x [15.52 – 15.72] |
| 4,194,304 | 48.7 us | 244.2 us | 5.02x [4.92 – 5.03] | 752.4 us | 15.5x [15.15 – 15.51] |
| 16,777,216 | 228.6 us | 996.7 us | 4.36x [4.31 – 4.37] | 3.02 ms | 13.2x [13.05 – 13.23] |
| overall | | | 5.83x [5.82 – 5.84] | | 74.7x [74.47 – 74.72] |

Stopped here: from 4,096 to 65,536 bytes the kernel runs within 1–5% of three 16-byte loads per
cycle, which is all the core can load; from 262,144 to 4,194,304 bytes it runs at the rate the L2
cache delivers, which no loop shape, prefetch or second stream changed; at 16 bytes it is at the
taken-branch floor; and from 64 to 1,024 bytes it is held by the instruction count of a call
(about 24 cycles at 1,024, where the loads alone need 21.3).

Not exercised by the judge's timing: sizes below 16 and sizes that are not a multiple of 16, so
the general-register path and the masked last vector are checked for correctness but never
timed. Not exercised by the judge at all: worst-case all-0xFF data (checked privately, see
attempt 2; rerun on the final version).
