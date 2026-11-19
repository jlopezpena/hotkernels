# cumsum_bool attempts (arm64)

All timings on an Apple M1 Pro (performance core: 128 KB first-level cache, 12 MB second-level cache shared by the cluster), macOS, with the spec locked. Another kernel run was timing on the same machine throughout, so the figures are noisier than a quiet run, mostly from about 100,000 elements up.

What the compiler does with `naive.c` (Apple clang, `-O3`): the scalar chain, one `ldrb`, one `add` and one `str` per element, not unrolled: 0.37 ns per element at every size from 256 to 16,777,216, about 1.2 cycles. NumPy takes 2.5 ns per element at 4,096 and 2.3 from 65,536 up.

The output is eight bytes for every byte read. Sixteen elements are 128 bytes of output, eight 16-byte stores, and this core retires about two of those per cycle, so about 4.4 cycles per 16 elements (0.085 ns per element) is the floor for any version in the first-level cache.

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

- **Idea:** the same instructions in another order. With one `stp` in place of four the loop of attempt 3 runs at the speed of eight stores alone, and so do eight stores alone, but together they take a quarter longer, and adding four `nop` or four integer adds changes nothing: the loop is not short of width or of ports. What it is short of looks like room for instructions that wait: the stores are issued long before their data is ready. So the loop widens the block it already holds, then loads, multiplies and moves the next one, then stores the current one; the first block is loaded before the loop and the last widened and stored after it.
- **check:** pass
- **compare:** significantly faster, 1.039x [1.028 – 1.049]
- **Kept:** yes
- **Notes:** 1.04x at 64, 1.07x to 1.08x from 256 to 16,384 (0.097 ns per element at 4,096); 16 unchanged, and nothing significant from 65,536 up, where the kernel already runs at the speed of a loop that only stores (see below).
