# sum_u8 attempts

**Verdict: assembly kept.** Final `judge run`: 5.77x [5.77 – 5.78] faster than naive C overall
(3.3x at 16 bytes, 8.3–8.9x from 1,024 to 65,536, 5.0x from 262,144 to 4,194,304, 3.9x at 16,777,216).

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

## 5. Software prefetch in the 256-byte loop (measured privately, never run through `compare`)

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

## Where it stands

Final `judge run` (30 samples per point, machine under load from other agents' timing runs):

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 1.0 ns | 3.4 ns | 3.33x [3.33 – 3.33] | 1.2 us | 1181x [1180 – 1182] |
| 64 | 1.6 ns | 6.5 ns | 4.20x [4.20 – 4.21] | 1.2 us | 797x [797 – 799] |
| 256 | 2.9 ns | 17.7 ns | 6.00x [6.00 – 6.01] | 1.2 us | 423x [422 – 423] |
| 1,024 | 7.5 ns | 62.4 ns | 8.33x [8.32 – 8.33] | 1.4 us | 193x [193 – 194] |
| 4,096 | 27.6 ns | 245.1 ns | 8.87x [8.86 – 8.87] | 2.0 us | 71.4x [71.35 – 71.52] |
| 16,384 | 110.7 ns | 960.0 ns | 8.67x [8.67 – 8.70] | 4.3 us | 38.7x [38.61 – 38.80] |
| 65,536 | 434.3 ns | 3.8 us | 8.79x [8.78 – 8.80] | 13.2 us | 30.3x [30.29 – 30.39] |
| 262,144 | 3.0 us | 15.3 us | 5.01x [5.01 – 5.03] | 48.4 us | 15.9x [15.87 – 15.93] |
| 1,048,576 | 12.0 us | 61.0 us | 5.09x [5.09 – 5.09] | 189.2 us | 15.8x [15.77 – 15.81] |
| 4,194,304 | 48.5 us | 244.2 us | 5.03x [5.03 – 5.04] | 752.8 us | 15.5x [15.51 – 15.53] |
| 16,777,216 | 254.0 us | 994.9 us | 3.92x [3.91 – 3.92] | 3.02 ms | 11.9x [11.86 – 11.88] |
| overall | | | 5.77x [5.77 – 5.78] | | 72.7x [72.64 – 72.70] |

Stopped here: from 4,096 to 65,536 bytes the kernel runs within 1–5% of three 16-byte loads per
cycle, which is all the core can load; from 262,144 bytes it runs at the rate the L2 cache and
memory deliver, which no loop shape, prefetch or second stream changed; at 16 bytes it is at the
taken-branch floor; and from 64 to 1,024 bytes it is held by the instruction count of a call
(about 24 cycles at 1,024, where the loads alone need 21.3).

Not exercised by the judge's timing: sizes below 16 and sizes that are not a multiple of 16, so
the general-register path and the masked last vector are checked for correctness but never
timed. Not exercised by the judge at all: worst-case all-0xFF data (checked privately, see
attempt 2).
