# cumsum_bool attempts (arm64)

All timings on an Apple M1 Pro (performance core: 128 KB first-level cache, 12 MB second-level cache shared by the cluster), macOS, with the spec locked. Another kernel run was timing on the same machine throughout, so the figures are noisier than a quiet run, mostly from about 100,000 elements up.

What the compiler does with `naive.c` (Apple clang, `-O3`): the scalar chain, one `ldrb`, one `add` and one `str` per element, not unrolled: 0.37 ns per element at every size from 256 to 16,777,216, about 1.2 cycles. NumPy takes 2.5 ns per element at 4,096 and 2.3 from 65,536 up.

The output is eight bytes for every byte read. Sixteen elements are 128 bytes of output, eight 16-byte stores, and this core does two of those per cycle, so four cycles per 16 elements is the floor for any version in the first-level cache; a loop of the stores alone measures 0.08 ns per element.

## 1. Baseline: one element at a time

- **Idea:** `ldrb`, `add`, `str` with post-indexed pointers and a down-counter; the starting point, to get `check` passing.
- **check:** pass
- **compare:** n/a (first version). `run`: 1.00x [1.00 – 1.00] of naive C overall, not significant at any size; 11.9x over NumPy.
- **Kept:** yes (baseline)
- **Notes:** the same instructions the compiler emits for `naive.c`: 0.37 ns per element at every size, in both tables of `run`.

## 2. NEON, sixteen elements per step: byte counts by multiply and `tbl`, widened by `uaddw`

- **Idea:** the running count within a block of 16 as sixteen bytes, P, without the count before the block: `mul` of the 32-bit lanes by 0x01010101 (counts within each group of four), `tbl` and `add` to carry the group totals to the second group of each half, `tbl` and `add` to carry the low half's total to the high half. Four `tbl` spread P four bytes at a time over 32-bit lanes; `uaddw` and `uaddw2` widen those to 64 bits and add T, the count before the block in both lanes, giving the eight output vectors. The next T is T plus byte 15 of P (`tbl`, `add`): one add per 16 elements on the carried chain. 19 vector operations, one load and four `stp` per 16 elements. Under 16 elements and for the last 0 to 15, the loop of attempt 1.
- **check:** pass
- **compare:** significantly faster, 2.954x [2.945 – 2.962]
- **Kept:** yes
- **Notes:** 2.87x at 16, 2.75x at 64, 3.03x to 3.13x from 256 to 1,048,576 (0.122 ns per element at 4,096, about 6.2 cycles per block), 2.91x at 4,194,304, 2.57x at 16,777,216. 19 operations in 6.2 cycles is three per cycle on a core with four vector pipes, so something other than the count limits it.

## 3. The count and the byte counts in general registers, eight elements per group

- **Idea:** a micro-benchmark outside the judge (see "Measurements outside the judge" below) showed that on this core the depth of the chain from the load to the stores costs time even when nothing is carried from block to block, and attempt 2 is seven vector operations deep. So: load the block as two 64-bit numbers and multiply each by 0x0101010101010101, which gives the running count within eight elements as eight bytes and the group's total as the top byte. Keep T in a general register (`add x3, x3, x4, lsr #56`, one cycle per group) and move it to a vector register for each group with `dup`. The products go across with `fmov`; two `tbl` and four `uaddw` per group as before. 12 vector operations, four moves from general to vector registers and two deep after the move, against 19 and seven.
- **check:** pass
- **compare:** significantly faster, 1.071x [1.056 – 1.087]
- **Kept:** yes
- **Notes:** 1.06x at 16 and 64, 1.11x to 1.14x from 256 to 16,384 (0.106 ns per element at 4,096), 1.06x at 1,048,576; not significant at 65,536, 262,144 and 16,777,216, where the intervals were wide with the other run on the machine.

## 4. Load the next block between widening the current one and storing it

- **Idea:** the same instructions in another order. In the micro-benchmark the loop of attempt 3 with one `stp` in place of four takes 4.15 cycles a block and the eight stores alone 4.2, but together they take 5.3, and adding four `nop` or four integer adds changes nothing: the loop is not short of width or of integer ports. What it is short of looks like room for instructions that wait: the stores are issued long before their data is ready. So the loop widens the block it already holds, then loads, multiplies and moves the next one, then stores the current one; the first block is loaded before the loop and the last widened and stored after it.
- **check:** pass
- **compare:** significantly faster, 1.039x [1.028 – 1.049]
- **Kept:** yes
- **Notes:** 1.04x at 64, 1.07x to 1.08x from 256 to 16,384 (0.097 ns per element at 4,096); 16 unchanged, and nothing significant from 65,536 up, where the kernel already runs at the speed of a loop that only stores (see below).

## 5. The 32-bit counts from a table indexed by the eight mask bits

- **Idea:** the x86-64 kernel's lookup, in this kernel's shape. One multiply by 0x0102040810204080 gathers a group's eight mask bits; they index two tables of 256 entries of 16 bytes, the running counts of the group's first and last four elements already spread over 32-bit lanes (8 KB in all). The loads replace the two `fmov` and the four `tbl`, so every vector operation left is a `uaddw` that depends on nothing but a load and T: 8 vector operations and four table loads a block against 12 and two moves. T as in attempt 3, from the multiply by 0x0101010101010101; the order of attempt 4.
- **check:** pass
- **compare:** significantly slower, 0.896x [0.876 – 0.917]
- **Kept:** no
- **Notes:** 0.83x to 0.87x from 16 to 65,536 (0.114 ns per element at 4,096 against 0.098), 0.96x to 1.00x from 262,144 up. Four loads a block whose addresses come out of a multiply cost more here than four `tbl`. An earlier form of the same idea, with T in a vector register and the group totals from the table too (64-byte entries, 16 KB), ran at 0.125 ns per element at 4,096 in the micro-benchmark; with every lookup redirected to entry 0 it ran at 0.103, so it is the dependent loads and not the number of operations.

## 6. The last 1 to 15 elements as groups of eight, the last one overlapping

- **Idea:** the judge times sizes that are multiples of 16, so the one-element loop for what is left after the blocks is never timed, and it is slow: a chain of one cycle per element and a loop exit that is seldom predicted. Take 8 or more left as one group (`ldr`, `mul`, two `tbl`, four `uaddw`), then the last 1 to 7 as the group of eight that ends exactly at the end of the array: the elements it shares with what was done already get the same outputs written again, and their count, a byte of the same product, is taken back out of T first. Sizes of 8 to 15 take the same two steps; under 8 stays one element at a time.
- **check:** pass
- **compare:** no significant difference, 0.999x [0.997 – 1.000]
- **Kept:** no
- **Notes:** `compare` cannot see it, as expected: every size it times is a multiple of 16 and read 1.00x. In the micro-benchmark, per call: 8 elements 4.6 ns to 1.9, 9 elements 5.0 to 2.3, 15 elements 6.8 to 2.3, 24 elements 5.3 to 2.9, 31 elements 7.4 to 3.7, 47 elements 7.7 to 5.2; 16, 64, 100 and 4,096 unchanged within its noise. The differential test below passed with it. Reverted under the rule; whether a gain at sizes the judge does not time is worth keeping is a person's decision.

## Measurements outside the judge

A micro-benchmark in C (best of nine timings of repeated calls on one buffer, half the mask true) and a differential test against the scalar loop: every size from 0 to 700 and 16 sizes up to 5,000,001, the mask ending 0, 5 and 10 bytes before an inaccessible page, the output ending 0 to 56 bytes and 3 bytes before one, five mask densities, and the bytes before and after the output checked: 95,211 runs. Every version that went to the judge passed it. Timings are given in cycles per block of 16 elements at 3.2 GHz; with the other run on the machine the core ran a few percent below that, and the figures moved by about 0.1 from run to run. They decided nothing: every kept change has a `compare` verdict.

- **The floor.** A loop of four `stp` of two vector registers and nothing else: 4.2. Two 16-byte stores a cycle.
- **Depth costs, not only count.** Eight stores and k independent `eor`: 4.4 for k = 8 to 16, 5.6 for k = 20, 6.4 for k = 24, so four vector operations a cycle. The same 16 `eor` arranged two deep from the load: 4.9; four deep: 5.7; as one chain of 16: 6.5, though nothing is carried from block to block. Attempt 2 (19 operations, seven deep) took 6.2; with every `tbl` replaced by `eor` and every `uaddw` by `add`, still 6.2. Attempt 4 is 12 operations, two deep after the moves, at 4.8 to 4.9.
- **Attempt 2 with the byte counts from the general registers** (`ldp`, two `mul` chained through the low half's total, `fmov` and `mov v0.d[1], x5`): 6.2, no better than the five vector operations it replaced. With T still carried in a vector register the gain of attempt 3 is not there.
- **`st2` to do the widening** (32-bit counts in one register, a zero register beside it, one `st2` per four elements; right only below 2^32 elements): 7.35 where attempt 2 took 6.2. A loop of four `st2` alone runs at 4.2, so the cost is in what `st2` does with data that is not ready. `st4` of 16-bit lanes alone: 4.65. Four-register `st1` in place of `stp`: no change.
- **One 16-byte move in place of two `fmov`** (`fmov` and `mov v0.d[1], x5`): 5.8 against 5.3 for attempt 3.
- **Knock-outs of attempt 4** (results wrong, timing only): without the two `dup` of T 4.7; with one `stp` in place of four 4.1. Four `nop` or four integer adds more in the loop of attempt 3: no change.
- **T moved to a vector register once a block** (the second group's bytes continue the first's, `(x5 + total) * 0x0101010101010101`): 5.0, no change. **Once every eight blocks** (the count since then added to every byte with a `madd`, which stays below 256 for 128 elements): 4.9, no change, and slower at 1,024.
- **Three stages** (widen block i, spread block i+1, load block i+2, store block i; two register copies a block): 4.8 to 5.1, no better than attempt 4. Other orders of the same instructions (the next block's load between the `tbl` and the `uaddw`; half the stores before the moves of T): 4.7 to 5.0, the same.
- **Two blocks per iteration** (on attempt 3): 5.4 to 5.5 against 5.3.
- **`stnp` in place of `stp`:** 6.7 at 4,096 against 4.8, and no faster at 1,048,576, 4,194,304 or 16,777,216 in three alternating runs (0.158 against 0.152 ns per element at 4,194,304, 0.176 against 0.176 at 16,777,216).
- **`dc zva` on two output lines 320 bytes ahead of the stores:** 10 cycles a block at 4,096, 20 at 65,536, 70 from 1,048,576 up. Timing only; it was never a candidate as written.
- **Beyond the first-level cache the kernel runs at the speed of the stores alone:** the four-`stp` loop takes 0.107, 0.124, 0.116, 0.154 and 0.154 ns per element at 65,536, 262,144, 1,048,576, 4,194,304 and 16,777,216, the kernel 0.118, 0.119, 0.124, 0.123 and 0.177 in the same session (both moved by 20% between runs with the other job on the machine).
- **A load beside the stores is not free.** Four `stp` and j loads a block: 4.1 with none, 4.4 with one `ldr` or one `ldp` of two general registers, 4.7 with two, 4.7 to 4.9 with four or six; with `ldp` of two vector registers 4.6, 4.5, 5.2 and 5.8 for one, two, four and six. So the floor for a kernel that reads its 16 mask bytes with one load a block is about 4.4, and the table of attempt 5, four more loads a block, starts from 5 or more. Attempt 4 is within a tenth of where the same count of operations two deep lands in the synthetic loop.

## Where it stands

Four kept attempts. In the first-level cache the loop takes about 4.8 cycles per 16 elements (0.097 ns per element at 4,096 in the judge) against 19 for the reference chain; eight 16-byte stores with one load beside them take 4.4. From 65,536 elements up it runs at the speed of the stores alone, and neither `stnp` nor `dc zva` moves that on this machine. 
Final `judge run` (locked, not recorded, with another kernel run timing on the machine): 3.24x [3.23 – 3.29] over naive C overall, 38.5x [38.41 – 39.10] over NumPy.

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 2.3 ns | 7.1 ns | 3.06x [3.06 – 3.06] | 1.2 us | 515x |
| 64 | 7.4 ns | 22.5 ns | 3.04x [3.03 – 3.05] | 1.3 us | 172x |
| 256 | 25.9 ns | 97.0 ns | 3.75x [3.73 – 3.77] | 1.8 us | 69.0x |
| 1,024 | 100.2 ns | 390.0 ns | 3.89x [3.86 – 3.93] | 3.5 us | 35.3x |
| 4,096 | 385.5 ns | 1.5 us | 3.93x [3.92 – 3.95] | 10.2 us | 26.6x |
| 16,384 | 1.5 us | 6.0 us | 3.96x [3.95 – 3.97] | 38.4 us | 25.3x |
| 65,536 | 6.0 us | 23.8 us | 3.96x [3.95 – 3.97] | 148.0 us | 24.6x |
| 262,144 | 25.7 us | 94.8 us | 3.69x [3.68 – 3.70] | 596.6 us | 23.2x |
| 1,048,576 | 150.7 us | 383.2 us | 2.54x [2.54 – 2.55] | 2.40 ms | 16.0x |
| 4,194,304 | 607.1 us | 1.53 ms | 2.52x [2.44 – 2.53] | 9.65 ms | 15.9x |
| 16,777,216 | 2.95 ms | 6.13 ms | 2.08x [2.06 – 2.45] | 38.65 ms | 13.1x |

The run before it, minutes earlier, read 2.90x [2.87 – 2.91] overall: the same figures up to 16,384 and 2.85x, 1.96x, 2.51x, 2.06x and 2.06x from 65,536 up, where the other run shares the second-level cache. The `compare` runs above had 1,048,576 at 122 to 128 us and 4,194,304 at 511 to 620 us. The figures from 65,536 up want a quiet machine.

**The second table, output 16 bytes behind the input within a page.** The kernel is slower there at the two smallest sizes it times, in both runs: 427.4 ns against 408.9 ns at 4,096, 0.96x [0.95 – 0.96] (0.94x [0.93 – 0.96] in the run before), and 1.6 us against 1.5 us at 16,384, 0.98x [0.97 – 0.98] (0.99x). At 4,194,304 and 16,777,216 it read 0.99x and 1.00x; from 65,536 to 1,048,576 its own "apart" column was 20% to 140% off the first table's in these runs (7.3 us, 62.2 us and 319.4 us), so those rows say nothing. The loop already loads the next block ahead of the current block's stores. No attempt was aimed at this table.

Not tried, or not kept, and why:

- **The vector tail of attempt 6:** works, passes `check` and the differential test, and is two to three times faster for the sizes it serves, but `compare` times only multiples of 16 and reads 1.00x. The kernel as kept does up to 15 elements one at a time after the blocks, and every size under 16 that way.
- **A 2-register `tbl` or `tbx` that writes the count's high bytes and a 16-bit low part straight into each output vector**, which would make every output vector one operation from a moved value: it needs a second path for the block in which the low 16 bits wrap, which the judge's correctness cases (5,000 elements at most) never reach, and it saves two of twelve operations if a 2-register `tbl` is one operation on this core, none if it is two.
- **Output brought to a multiple of 64 bytes first:** a 16-byte store only straddles a cache line when `out` is not a multiple of 16, and the mask is a byte per element, so any head length would do; the judge's timed buffers are aligned, so `compare` would not see it either.
- **A shorter path for exactly 16 elements:** about 45 instructions run for one block; a percent or two overall at best.
