# count_nonzero_i64 attempts (arm64)

**Verdict: assembly kept.** Final `judge run`: 1.28x [1.28 – 1.28] faster than naive C overall.
The gain is where the array fits in the L1 cache (1.35–1.73x from 16 to 16,384 elements) and
where it does not fit in the L2 cache (1.17x at 4,194,304, 1.07x at 16,777,216). In between, from
65,536 to 1,048,576 elements, both versions run at what the L2 cache delivers and tie
(1.00x, 0.99x, 1.02x).

All runs on an Apple M1 Pro with `--unlocked` (the spec was not on `main` yet), while other
agents were timing other kernels on the same machine, so absolute timings are noisier than usual.

What the compiler does with `naive.c` (`clang -O3 -mcpu=native`, Apple clang 21): eight elements
per step, four 16-byte loads, four `cmtst` and four `sub` into four accumulators of 64-bit lanes,
then a scalar loop for up to seven leftover elements. That is two vector instructions per vector
on four vector ports: two vectors, 32 bytes, per cycle (0.51 cycles per 16 bytes measured from 256
to 16,384 elements). From 65,536 elements it runs at what the L2 cache delivers (0.59 cycles per
16 bytes) and from 4,194,304 at what memory delivers (0.86).

## 1. Baseline: `cmtst` and `sub`, one vector per step, one accumulator

- **Idea:** simplest vector version. Load two elements, `cmtst` gives -1 per non-zero lane, subtract from one accumulator of 64-bit lanes. Odd last element in a general register.
- **check:** pass
- **compare:** n/a (first version). `run`: 0.36x [0.36 – 0.36] vs naive C overall; 0.62x at 16, 0.25–0.31x from 1,024 to 1,048,576, 0.43x at the two largest sizes.
- **Kept:** yes
- **Notes:** starting point only. One accumulator makes every step wait for the previous `sub` (two cycles per vector).

## 2. `umaxp` tree: sixteen elements to sixteen bytes, `sadalp` into 16-bit counters

- **Idea:** compare-and-subtract costs two vector instructions per vector whatever the accumulator width, which is exactly the compiler's loop. Reduce before comparing instead: a 64-bit element is non-zero exactly when the larger of its two 32-bit halves is, and `umaxp.4s` computes that for two vectors in one instruction. Repeating it at 16 and 8 bits turns eight vectors into sixteen bytes with seven `umaxp`; one `cmtst` and one `sadalp` (16-bit lanes holding minus the count) finish the group. Nine vector instructions per eight vectors instead of sixteen. Two groups per step, counts flushed to a general register every 8,192 steps. Leftovers: fewer than sixteen elements are loaded into the same eight registers with the missing ones zeroed (an odd element with `ldr d`, which zeroes the upper half) and counted by the same tree, so nothing is read past the end and nothing is counted twice.
- **check:** pass (first try failed `memory safety` at n=524,287: the flush wrote its result to `x0`, the cursor)
- **compare:** significantly faster, 3.438x [3.432 – 3.446]
- **Kept:** yes
- **Notes:** `run` after this: 1.24x [1.24 – 1.24] vs naive C; 1.35–1.58x from 16 to 16,384, 1.00x at every size above. A private micro-benchmark put the loop at 0.335 cycles per 16 bytes in L1, which is three 16-byte loads per cycle, all the core can load; the compiler's shape with eight accumulators (`cmtst`, `sub`) measured 0.505, the same as the compiler. Checked privately because the judge's random 64-bit integers are almost never zero: all-non-zero, all-zero and a mix of awkward bit patterns (a lone bit in either half, only the sign bit, all ones) at every n up to 3,000, each pattern alone at every position, and sizes up to 3,145,728 across the flush boundaries.

## 3. Below 32 elements, add the group bytes directly

- **Idea:** with at most two groups the 16-bit accumulators are overhead. Add the two byte vectors, `addv` the sixteen bytes (minus the count, which fits in a signed byte), `smov` and subtract: three instructions after the tree instead of six. Arrays with leftovers after the main loop reuse the same code.
- **check:** pass
- **compare:** significantly faster, 1.009x [1.009 – 1.010]
- **Kept:** yes
- **Notes:** 1.11x at 16 (1.6 ns to 1.4 ns); nothing elsewhere.

## 4. `ldnp` in the main loop

- **Idea:** `ldnp` is the load pair that hints the data will not be needed again soon. In a private micro-benchmark of this loop it was identical to `ldp` while the array fits in L1 or L2 (0.346 against 0.346 cycles per 16 bytes at 8,192 elements, 0.612 against 0.612 at 65,536) and faster beyond: 0.61–0.62 against 0.63–0.66 at 1,048,576, 0.78 against 0.84 at 4,194,304, 0.84 against 0.85 at 16,777,216, the same in three runs.
- **check:** pass
- **compare:** significantly faster, 1.007x [1.003 – 1.009]
- **Kept:** yes
- **Notes:** 1.06x at 4,194,304, 1.01x at 16,777,216, 1.02x (not significant) at 1,048,576, and 0.98x [0.95 – 1.00] at 262,144, which the micro-benchmark does not reproduce.

## 5. Prefetching copy of the loop for arrays beyond the L2 cache

- **Idea:** from 2,097,152 elements (16 MB; the L2 cache is 12 MB) run a copy of the loop that also issues two `prfm pldl2keep` per 256-byte step, 256 KB ahead. Private micro-benchmark, cycles per 16 bytes, `ldnp` alone against `ldnp` with these prefetches: 0.775 against 0.74 at 4,194,304 and 0.84 against 0.80 at 16,777,216. Four prefetches per step were worse than two, distances from 32 KB to 256 KB were the same, 1 MB was worse, and `ldp` with the same prefetches gained only half as much.
- **check:** pass
- **compare:** significantly faster, 1.009x [1.007 – 1.010]
- **Kept:** yes
- **Notes:** 1.06x at 4,194,304 and 1.05x at 16,777,216. The prefetches must stay off below the threshold: in the micro-benchmark four per step took 262,144 elements from 0.59 to 0.67 cycles per 16 bytes, and two per step took 1,048,576 from 0.61–0.62 to 0.64. Near the end of the array the prefetched addresses lie beyond it, which is harmless (a prefetch never faults) but worth knowing.

## 6. Plain loop without block logic below one block

- **Idea:** below 262,144 elements no flush is ever needed, so give those sizes a loop without the per-block setup (five instructions and a branch per call).
- **check:** pass
- **compare:** significantly faster, 1.002x [1.000 – 1.004]
- **Kept:** yes
- **Notes:** marginal: 1.01x at 64, nothing clear elsewhere. Kept because the judge calls it significant and it costs nothing above.

## 7. Ideas for the sizes that live in the L2 cache (measured privately, never run through `compare`)

- **Idea:** from 65,536 to 1,048,576 elements the kernel and the compiler's loop both run at 0.59–0.61 cycles per 16 bytes, the rate at which the L2 cache refills L1. Tried on the tree loop: `prfm pldl1keep` 1 KB to 16 KB ahead, `prfm pldl2keep` 16 KB ahead, reading the array backwards, and four streams (the four quarters of the array at once).
- **check:** not run (private micro-benchmark with its own result check)
- **compare:** not run. Micro-benchmark at 65,536 and 262,144 elements, cycles per 16 bytes: plain 0.61, any prefetch 0.69–0.70, backwards 0.61, four streams 0.61.
- **Kept:** no
- **Notes:** nothing moves the L2 rate, and prefetching costs 13% there. These three sizes stay a tie with naive C.

## Where it stands

Final `judge run` (30 samples per point, machine under load from other agents' timing runs):

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 1.4 ns | 2.5 ns | 1.73x [1.73 – 1.73] | 186.0 ns | 130x [129 – 130] |
| 64 | 4.3 ns | 5.7 ns | 1.35x [1.35 – 1.35] | 200.6 ns | 47.2x [47.03 – 47.22] |
| 256 | 14.1 ns | 20.6 ns | 1.47x [1.46 – 1.47] | 239.7 ns | 17.0x [17.00 – 17.04] |
| 1,024 | 54.1 ns | 85.6 ns | 1.58x [1.58 – 1.58] | 512.4 ns | 9.46x [9.45 – 9.47] |
| 4,096 | 215.0 ns | 323.0 ns | 1.50x [1.50 – 1.50] | 1.5 us | 6.81x [6.80 – 6.82] |
| 16,384 | 859.6 ns | 1.3 us | 1.49x [1.48 – 1.49] | 5.4 us | 6.30x [6.28 – 6.33] |
| 65,536 | 6.0 us | 6.0 us | 1.00x [1.00 – 1.00] n.s. | 20.7 us | 3.43x [3.43 – 3.44] |
| 262,144 | 24.1 us | 23.9 us | 0.99x [0.99 – 1.00] | 81.7 us | 3.39x [3.37 – 3.41] |
| 1,048,576 | 100.5 us | 102.5 us | 1.02x [1.00 – 1.02] n.s. | 327.3 us | 3.26x [3.19 – 3.27] |
| 4,194,304 | 473.4 us | 554.9 us | 1.17x [1.17 – 1.18] | 1.31 ms | 2.76x [2.75 – 2.77] |
| 16,777,216 | 2.08 ms | 2.22 ms | 1.07x [1.07 – 1.07] | 5.24 ms | 2.52x [2.52 – 2.52] |
| overall | | | 1.28x [1.28 – 1.28] | | 8.19x [8.17 – 8.20] |

Stopped here: in L1 the loop runs at three 16-byte loads per cycle (0.34 cycles per 16 bytes
against a limit of 0.333), from 65,536 to 1,048,576 elements it runs at the L2 rate like the
compiler's loop, and beyond that at what memory delivers with `ldnp` and prefetch. At 262,144
this run shows the kernel 1% behind naive C (0.99x [0.99 – 1.00]); both are at the L2 rate there
and the private micro-benchmark shows no difference, so treat that size as a tie.

Not exercised by the judge's timing: sizes below 16 and sizes that are not a multiple of 32, so
the leftover code is checked for correctness but never timed; and data with zeros in it (the
timing input is uniformly random 64-bit integers, so the count always equals n; the code has no
data-dependent branch, so the timing does not depend on it).
