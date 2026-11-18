# count_nonzero_f64 attempts

This log holds two architectures. ARM64 (Apple M1 Pro) comes first. x86-64 (Intel Core i5-14600KF) follows under [x86-64](#x86-64), with its own numbering.

**Verdict: assembly kept.** Final `judge run`: 1.41x [1.41 – 1.41] faster than naive C overall,
significantly faster at every size: 1.39–1.74x from 16 to 16,384 elements, 1.26–1.30x from 65,536
to 1,048,576, 1.19x at 4,194,304 and 1.08x at 16,777,216.

All runs on an Apple M1 Pro with `--unlocked` (the spec was not on `main` yet), while other
agents were timing other kernels on the same machine, so absolute timings are noisier than usual.

What the compiler does with `naive.c` (`clang -O3 -mcpu=native`, Apple clang 21): eight elements
per step, four 16-byte loads, then per vector `fcmeq #0.0`, `mvn` and `sub` into four accumulators
of 64-bit lanes, and a scalar loop for up to seven leftover elements. That is three vector
instructions per vector on four vector ports: 0.75 cycles per 16 bytes, measured at every size
from 1,024 to 1,048,576 elements, so even the sizes that live in the L2 cache (which delivers
16 bytes in 0.59 cycles) are limited by the loop and not by the cache. Only from 4,194,304
elements is it limited by memory (0.87).

## 1. Baseline: `cmtst` against a 63-bit mask and `sub`, one vector per step, one accumulator

- **Idea:** simplest vector version. A double is zero exactly when its low 63 bits are zero, and every NaN has some of them set, so test the bits: `cmtst` against 0x7fffffffffffffff gives -1 per non-zero lane, subtracted from one accumulator of 64-bit lanes. Odd last element in a general register.
- **check:** pass
- **compare:** n/a (first version). `run`: 0.45x [0.45 – 0.45] vs naive C overall; 0.62x at 16, 0.38–0.45x from 1,024 up.
- **Kept:** yes
- **Notes:** starting point only. One accumulator makes every step wait for the previous `sub` (two cycles per vector). Testing bits instead of comparing as floating point gives the non-zero mask directly (no `mvn`) and does not depend on the caller's floating-point mode, where a flush-to-zero mode can make a floating-point compare treat denormals as zero.

## 2. Sixteen elements per group: `cmtst`, `uzp1` tree to bytes, `sadalp` into 16-bit counters

- **Idea:** two vector instructions per vector instead of the compiler's three, in the shape of the count_nonzero_i64 kernel: eight `cmtst`, then `uzp1` at 32, 16 and 8 bits packs the eight masks into sixteen bytes (seven instructions), and one `sadalp` adds them into 16-bit lanes. Two groups per step, counts flushed every 8,192 steps, leftovers through the same tree with the missing vectors zeroed.
- **check:** pass
- **compare:** significantly faster, 2.895x [2.894 – 2.896]
- **Kept:** yes
- **Notes:** against naive C at that point: about 1.4–1.5x from 16 to 16,384, 1.28x at 65,536 and 262,144, 1.2x at 1,048,576, 1.01x at the two largest. Private micro-benchmark, cycles per 16 bytes in L1: this shape 0.504, eight `cmtst` and eight `sub` into 64-bit accumulators 0.504, `fcmeq #0.0` and `sub` (counting zeros) 0.504, compiler 0.755. All three two-instruction shapes are the same speed, limited by the four vector ports; the tree was chosen because short arrays need fewer instructions to finish. The integer trick that gives count_nonzero_i64 nine instructions per eight vectors (`umaxp` on the raw halves) does not carry over: the sign bit must be ignored first, and `shl #1` on every vector followed by that tree measured 0.535. Checked privately: all-non-zero, all-zero and a mix of awkward bit patterns (-0.0, a lone bit in either half, sign bit plus one low bit, NaNs and infinities of both signs) at every n up to 3,000, each pattern alone at every position, and sizes up to 3,145,728.

## 3. `ldnp` in the main loop

- **Idea:** as in count_nonzero_i64 attempt 4: the non-temporal load pair is free in cache and a few percent faster beyond it.
- **check:** pass
- **compare:** significantly faster, 1.006x [1.005 – 1.007]
- **Kept:** yes
- **Notes:** 1.03x at 1,048,576, 1.04x at 4,194,304, 1.02x at 16,777,216; 0.98x [0.98 – 1.00] at 65,536, nothing elsewhere.

## 4. Prefetching copy of the loop for arrays beyond the L2 cache

- **Idea:** as in count_nonzero_i64 attempt 5: from 2,097,152 elements (16 MB) a copy of the loop that also issues two `prfm pldl2keep` per step, 256 KB ahead.
- **check:** pass
- **compare:** significantly faster, 1.005x [1.005 – 1.007]
- **Kept:** yes
- **Notes:** 1.04x at 4,194,304 and 1.05x at 16,777,216; 0.99x at 64 from the extra test on the way into the loop (removed again by attempt 7).

## 5. Hybrid loop: two more elements per step through general registers

- **Idea:** the vector loop needs sixteen instructions on four vector ports per group, four cycles, while the core can start eight instructions per cycle and a group with its loads and loop control is only 26. Spend the slack on elements that use no vector port: per step, one group of sixteen plus two elements loaded with `ldp` into general registers and counted with `tst` / `cinc`. Eighteen elements in the time of sixteen. Steps counted with `udiv`, leftovers (0..17) through the existing tail. Used from 128 elements and below the prefetching threshold.
- **check:** pass
- **compare:** significantly faster, 1.021x [1.019 – 1.024]
- **Kept:** yes
- **Notes:** 1.08–1.09x at 1,024, 4,096 and 16,384; 0.96x at 256, where the leftovers and the division cost more than the loop saves. Private micro-benchmark, cycles per 16 bytes in L1: plain 0.504, one group plus one pair 0.462, one group plus two pairs 0.49, two groups plus one pair 0.475, two groups plus two, three, four pairs (all the general-register work at the end of the step) 0.49, 0.505, 0.52. More general-register work than one pair per group does not fit in the eight instructions per cycle.

## 6. Hybrid loop only from 512 elements

- **Idea:** fix the loss at 256 by raising the threshold.
- **check:** pass
- **compare:** significantly faster, 1.003x [1.002 – 1.004]
- **Kept:** yes
- **Notes:** 1.04x at 256, nothing else changes.

## 7. Plain loop without block logic for 32 to 511 elements

- **Idea:** those sizes never need a flush and never prefetch, so give them a loop without the per-block setup; the block loop is then only used with prefetch.
- **check:** pass
- **compare:** significantly faster, 1.004x [1.003 – 1.005]
- **Kept:** yes
- **Notes:** 1.05x at 64 (5.9 ns to 5.6 ns), nothing elsewhere.

## 8. Hybrid step of 32 elements (28 vector, 4 general-register) below one block

- **Idea:** a hybrid step that is a power of two needs no division and leaves no leftovers at the sizes that had none, so it can replace both the 18-element loop and the plain loop. Per 32 elements: one whole group, six more vectors through a tree padded with a zero register (13 instructions), and four elements in general registers: 29 vector instructions instead of 32. Used from 32 elements up to one block (262,144); above that the plain block loop with `ldnp`, which is at the L2 rate anyway.
- **check:** pass
- **compare:** significantly faster, 1.008x [1.004 – 1.011]
- **Kept:** yes
- **Notes:** 1.09x at 256, 1.02x at 1,024, 1.01x at 4,096 and 16,384, but 0.99x at 64 and 0.97x [0.96 – 0.99] at 262,144. The order of the instructions inside the step matters a great deal. Private micro-benchmark, cycles per 16 bytes in L1 (a slower session: plain loop 0.518, the 18-element hybrid 0.475): this step with the general-register tests placed after each vector block 0.470, the same instructions with all the general-register tests at the end of the step 0.535, slower than the plain loop. A 32-element step with twelve vectors and eight general-register elements measured 0.51.

## 9. Plain groups below 128 elements, hybrid from there

- **Idea:** at 64 elements a call is limited by its instruction count, not by the vector ports, and the hybrid step has five more instructions than two plain groups; use plain groups below 128.
- **check:** pass
- **compare:** significantly faster, 1.001x [1.000 – 1.001]
- **Kept:** no
- **Notes:** reverted although the judge calls it significant: 1.01x at 64 paid for with 0.99x at 256 (one more taken branch on the way in and out), a net 0.1% for a fourth loop. Superseded by attempt 10, which makes 64 elements one step.

## 10. Hybrid step of 64 elements (56 vector, 8 general-register)

- **Idea:** three whole groups, four vectors whose masks stop at eight 16-bit lanes and are added to the accumulator directly (eight instructions), and eight elements in general registers: 56 vector instructions per 64 elements instead of 58 for two 32-element steps. One 32-element step afterwards if n has that bit set.
- **check:** pass
- **compare:** significantly faster, 1.014x [1.011 – 1.016]
- **Kept:** yes
- **Notes:** 1.02x at 64, 256 and 1,024, 1.04x at 4,096, 1.03x at 16,384; 0.99x at 65,536. Private micro-benchmark in L1: 0.458 cycles per 16 bytes against 0.470 for the 32-element step; a 64-element step with thirty vectors and four general-register elements measured 0.50.

## 11. Hybrid loop only while the array fits in the L1 cache

- **Idea:** at 65,536 elements (512 KB) the loop is limited by the L2 cache and the hybrid step's extra instructions are pure cost; send everything from 32,768 elements to the plain block loop.
- **check:** pass
- **compare:** significantly faster, 1.002x [1.000 – 1.004]
- **Kept:** yes
- **Notes:** 1.01x [1.01 – 1.01] at 65,536, no significant difference anywhere else.

## 12. Fewer than two vector instructions per vector (measured privately or ruled out, never run through `compare`)

- **Idea:** the four vector ports allow two vectors per cycle at two instructions each; anything below two would lift the limit the way the `umaxp` tree does for integers.
- **check:** not run
- **compare:** not run. Micro-benchmark, cycles per 16 bytes in L1: `shl #1` then the integer `umaxp` tree 0.535 (17 instructions per 8 vectors); four `fcmeq` merged by one `tbl` over four table registers, then `sadalp`, 0.504, no better than `uzp1` (the four-register `tbl` costs about three instructions on this core).
- **Kept:** no
- **Notes:** the compare is one instruction per vector, and merging n masks into one countable vector takes n - 1 two-input merges plus the accumulate, so compare-then-merge is two per vector however it is arranged. Merging before the compare needs a two-input instruction that keeps "low 63 bits non-zero" for both inputs. The pairwise ones (`umaxp`, `addp`, `uminp`) treat the two halves of an element alike, so they cannot tell -0.0 (only the top bit of the high half) from a denormal with only the top bit of the low half; the saturating narrows (`uqxtn`, `sqxtn`, `sqxtun` and the shifting forms) each map either -0.0 or every ordinary negative number to the wrong side. A fast path that counts integers and branches to a slow path when a group might hold -0.0 would work on random data but would make the time depend on the data (an ordinary value such as 512.0 looks like a candidate), so it was not built.

## Where it stands

Final `judge run` (30 samples per point, machine under load from other agents' timing runs):

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 1.8 ns | 2.5 ns | 1.39x [1.39 – 1.39] | 181.8 ns | 102x [102 – 102] |
| 64 | 5.6 ns | 8.2 ns | 1.47x [1.47 – 1.47] | 205.0 ns | 36.7x [36.62 – 36.81] |
| 256 | 18.8 ns | 30.6 ns | 1.62x [1.62 – 1.63] | 273.9 ns | 14.5x [14.52 – 14.57] |
| 1,024 | 71.4 ns | 124.2 ns | 1.74x [1.74 – 1.74] | 516.5 ns | 7.23x [7.22 – 7.24] |
| 4,096 | 281.8 ns | 481.7 ns | 1.71x [1.71 – 1.71] | 1.5 us | 5.21x [5.20 – 5.21] |
| 16,384 | 1.1 us | 1.9 us | 1.69x [1.69 – 1.69] | 6.2 us | 5.47x [5.46 – 5.47] |
| 65,536 | 6.1 us | 7.7 us | 1.26x [1.26 – 1.26] | 23.0 us | 3.78x [3.77 – 3.78] |
| 262,144 | 24.0 us | 30.5 us | 1.27x [1.27 – 1.27] | 95.7 us | 3.99x [3.98 – 4.00] |
| 1,048,576 | 99.3 us | 129.4 us | 1.30x [1.30 – 1.31] | 385.9 us | 3.88x [3.86 – 3.90] |
| 4,194,304 | 476.4 us | 568.3 us | 1.19x [1.19 – 1.19] | 1.55 ms | 3.25x [3.24 – 3.26] |
| 16,777,216 | 2.08 ms | 2.25 ms | 1.08x [1.08 – 1.08] | 6.20 ms | 2.99x [2.98 – 2.99] |
| overall | | | 1.41x [1.41 – 1.41] | | 7.79x [7.78 – 7.79] |

Stopped here: in L1 the loop is at about 0.44 cycles per 16 bytes (281.8 ns for 4,096 elements),
against 0.75 for the compiler and a floor of 0.333 set by the loads that only fewer vector
instructions per vector could approach (attempt 12); from 65,536 to 1,048,576 elements it runs at
the L2 rate; beyond that at what memory delivers with `ldnp` and prefetch.

Things a reviewer should know. The hybrid steps are tuned to this core: their gain comes from
filling eight instruction slots per cycle exactly, and attempt 8 shows the same instructions in a
different order losing to the plain loop. The thresholds (32,768 elements for L1, 2,097,152 for
L2) are this machine's cache sizes. Not exercised by the judge's timing: sizes below 16 and, apart
from 16 itself, sizes that are not a multiple of 64, so the leftover code and the single
32-element hybrid step are checked for correctness but never timed; and data with zeros in it (the timing input is normally
distributed doubles, so the count always equals n; the code has no data-dependent branch, so the
timing does not depend on it).

# x86-64

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core), Linux, locked, pinned to one performance core by the judge. The two `count_nonzero` kernels are generated from one source and were written together.

What the compiler does with `naive.c` (`clang -O3 -march=native`): it vectorises the count: 0.039 ns per element at 1,024 and 0.066 at 65,536. NumPy takes 0.39 ns per element at 1,048,576.

## x86-64 1. Compare with zero and subtract into four accumulators, every load aligned

- **Idea:** the design's first candidate. The zeros are counted and subtracted from n. `vcmpeqpd` against zeros (+0.0 and -0.0 compare equal, a NaN does not) gives all ones, which is -1, in the lanes that are zero; subtracting that from an accumulator of 64-bit lanes adds one per zero. Two instructions per four elements, the compare with its load folded in. Sixteen elements per step into four accumulators, then four per step, then the last 0 to 3 one at a time. The loads are brought to a 32-byte boundary first, as in `sum_u8`, by counting up to three elements one at a time.
- **check:** pass
- **compare:** not run (first version). `run`: 0.99x over naive C overall.
- **Kept:** yes, as the starting point
- **Notes:** 0.50x at 16 elements and 0.78x at 64: the head and its loop cost more than a short array's straddling loads.

## x86-64 2. No alignment below 64 elements; vectors from 4 elements

- **Idea:** arrays of fewer than 64 elements skip the head, and the vector steps start at 4 elements, not 16.
- **check:** pass
- **compare:** not run: both kernels were changed in the generator and judged by `run` against the naive C again (table below).
- **Kept:** yes
- **Notes:** the 16-element call went from 3.3 ns to 2.3 ns.

## x86-64: where it stands

Final `judge run`:

`random`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 2.3 ns | 1.7 ns | 0.75x [0.75 – 0.75] | 133.7 ns | 59.0x [58.9 – 59.1] |
| 64 | 4.1 ns | 3.3 ns | 0.80x [0.80 – 0.81] | 159.0 ns | 39.1x [38.8 – 39.4] |
| 256 | 9.5 ns | 9.7 ns | 1.02x [1.02 – 1.02] | 247.2 ns | 25.9x [25.9 – 26.0] |
| 1,024 | 35.7 ns | 39.6 ns | 1.11x [1.11 – 1.11] | 550.1 ns | 15.4x [15.4 – 15.4] |
| 4,096 | 144.7 ns | 150.6 ns | 1.04x [1.04 – 1.04] | 1.8 us | 12.2x [12.2 – 12.2] |
| 16,384 | 650.6 ns | 646.1 ns | 0.99x [0.99 – 0.99] | 6.6 us | 10.2x [10.1 – 10.2] |
| 65,536 | 2.6 us | 4.3 us | 1.65x [1.65 – 1.65] | 25.8 us | 9.86x [9.86 – 9.87] |
| 262,144 | 15.6 us | 20.8 us | 1.33x [1.33 – 1.33] | 103.2 us | 6.61x [6.61 – 6.62] |
| 1,048,576 | 94.2 us | 105.0 us | 1.11x [1.11 – 1.12] | 412.6 us | 4.38x [4.38 – 4.39] |
| 4,194,304 | 797.9 us | 864.7 us | 1.08x [1.08 – 1.09] | 1.95 ms | 2.44x [2.43 – 2.45] |
| 16,777,216 | 4.14 ms | 4.40 ms | 1.06x [1.06 – 1.07] | 8.08 ms | 1.95x [1.95 – 1.96] |
| overall |  |  | 1.06x [1.06 – 1.06] |  | 10.4x [10.4 – 10.4] |

What holds it: memory, at nearly every size. Two 32-byte loads per cycle is eight elements per cycle, 0.024 ns per element, and the judge's timing leaves the first-level cache from about a thousand elements. The compiler's loop is close to the same limit, which is why the gain is what it is.

The candidates the design lists for this family (`docs/x86-64-design.md`):

| Candidate | Outcome |
|---|---|
| A compare against zero and a subtract into an accumulator | Kept |
| `vmovmskpd` and `popcnt` | Not tried: three instructions per four elements and a trip to the general registers, against two |
| For floats the sign is shifted out first | Not needed: `vcmpeqpd` treats the two zeros alike without it, in one instruction with its load folded in |
| Level with NumPy above the second-level cache whatever is done | Close to it: see the table from 1,048,576 up |
| Alignment of data | Kept from 64 elements (attempts 1 and 2) |
| A prefetch, several streams | Not tried. Worth 2% and 6% on `sum_u8`, at sizes where this kernel is already at memory's speed |
| Masked loads for leftover elements | Not tried: open issue 3 of the design |
| Instructions slow on other x86-64 processors | None used |

**Verdict for a person to confirm.** 1.06x over the naive C overall is below the 10% the goal template asks of an assembly kernel. It is never slower from 256 elements up and 1.2x to 1.65x faster at 65,536 and 262,144, where the aligned loads count; it loses at 16 and 64 elements. Removing `x86_64/kernel.s` makes this a C kernel on x86-64, and the figures would support that.
