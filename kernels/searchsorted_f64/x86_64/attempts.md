# searchsorted_f64 attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, locked, pinned to one performance core by the judge.

What the compiler does with `naive.c` (`clang -O3 -march=native`): one search at a time with a branch per step. The judge times the same needles again and again, so with 64 needles the branches are learnt and the naive C takes 6 to 17 ns per needle up to 262,144 elements; with 1,000 needles they are not, and it takes 34 ns per needle at 1,024 elements and 120 ns at 1,048,576. NumPy takes 78 ns per needle at 1,048,576 with 1,000 needles.

## 1. Baseline: sixteen branch-free searches in step, probes by `vgatherqpd`

- **Idea:** the design's candidate. One search is a fixed number of steps that depends only on n: `half = (len + 1) / 2; if a[lo + half - 1] < needle: lo += half; len -= half`. The answer is always in `lo .. lo + len` and `lo + len` never grows, so every probe is inside `a` whatever the data, sorted or not. Sixteen searches in four registers: per register one `vgatherqpd`, one `vcmpltpd` against the needles, one `vpand` with `half` and one `vpaddq`. A NaN in `a` compares false, which is right for every needle that is not a NaN. **A NaN needle** compares false at every step and gets 0; the needles are tested for NaN as they are read (one `vcmpunordpd` per two registers), and only if one was seen is its answer found, by one more search of `a` for its first NaN, and written over the answers of the NaN needles. One to three needles are searched one at a time in general registers with `cmova`; four to fifteen take one register at a time; a leftover is covered by a last group that overlaps the one before it.
- **check:** pass
- **compare:** not run (first version). `run`: 2.77x over naive C overall (64 needles 1.74x, 1,000 needles 4.40x), 3.82x over NumPy.
- **Kept:** yes
- **Notes:** slower than the instruction count allows: with 1,000 needles and 1,024 elements, 0.38 ns per needle and step, which is 32 cycles for a step of sixteen searches. A gather can start every two to three cycles on this processor (measured: 2.2), but the chain from one step of a search to its next (gather, compare, add) is about thirty cycles long, so four gathers in flight leave most of the time idle. At 4,194,304 and 16,777,216 with 64 needles the kernel was slower than the naive C (0.91x and 0.87x).

## 2. Up to 64 searches in step, their positions kept in `out`

- **Idea:** sixteen gathers in flight in place of four. Sixteen registers cannot hold sixteen vectors of positions and their needles, so the positions live in `out`, which is exactly one 64-bit slot per needle, and the needles stay in `v` as the compare's memory operand. A step loads a vector of positions, gathers, compares, adds and stores it back, for each vector of the chunk in turn. The first step needs no gather: every search probes the same element, which is broadcast. A count that is not a multiple of four is finished by one more vector in registers that ends with the last needle.
- **check:** pass
- **compare:** significantly faster, 1.917x [1.902 – 1.932]
- **Kept:** yes
- **Notes:** 2.0x to 2.4x up to 262,144 elements for both needle counts; 1.35x to 1.4x at 4,194,304 and 16,777,216. With 1,000 needles and 1,024 elements a step now costs 0.9 cycles per needle. The gather writes into its destination under a mask, which makes the destination an input: it is zeroed before each gather so that no probe waits for the one before.

## 3. More searches in step when `a` does not fit the second-level cache

- **Idea:** when the probes miss the caches, the number in flight is the number of searches in step. The chunk becomes 512 needles from 262,144 elements (2 MB).
- **check:** pass
- **compare:** significantly faster, 1.064x [1.061 – 1.068]
- **Kept:** yes
- **Notes:** only calls with more than 64 needles can change. With 1,000: 1.83x at 1,048,576, 1.52x at 4,194,304, 1.44x at 16,777,216, nothing below. Tried first as one chunk size for every n: 256 needles gave 1.72x, 1.50x and 1.31x at those three sizes and 0.97x at 262,144; 1,024 gave 1.89x, 1.43x and 1.55x, and 0.92x and 0.89x at 16,384 and 65,536, where the positions of a thousand searches are walked for every step. Both were 1.12x on the 1,000-needle table. So 64 below the threshold and 512 above it.

## Where it stands

Final `judge run`:

`random`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 51.7 ns | 531.7 ns | 10.3x [10.2 – 10.4] | 1.2 us | 23.8x [23.7 – 23.9] |
| 64 | 74.1 ns | 1.2 us | 15.6x [15.5 – 15.6] | 1.8 us | 24.5x [24.4 – 24.6] |
| 256 | 98.2 ns | 348.2 ns | 3.54x [3.54 – 3.55] | 1.3 us | 13.5x [13.4 – 13.7] |
| 1,024 | 113.4 ns | 409.8 ns | 3.61x [3.61 – 3.62] | 1.0 us | 9.05x [9.03 – 9.05] |
| 4,096 | 135.7 ns | 499.8 ns | 3.68x [3.68 – 3.69] | 1.1 us | 7.95x [7.94 – 7.96] |
| 16,384 | 185.3 ns | 655.5 ns | 3.54x [3.54 – 3.54] | 1.2 us | 6.52x [6.51 – 6.53] |
| 65,536 | 253.2 ns | 827.2 ns | 3.27x [3.27 – 3.27] | 1.3 us | 5.33x [5.32 – 5.33] |
| 262,144 | 324.2 ns | 1.1 us | 3.34x [3.34 – 3.35] | 1.5 us | 4.48x [4.47 – 4.48] |
| 1,048,576 | 584.7 ns | 1.2 us | 2.03x [2.02 – 2.03] | 1.8 us | 3.01x [2.99 – 3.01] |
| 4,194,304 | 1.5 us | 1.9 us | 1.30x [1.30 – 1.31] | 2.6 us | 1.73x [1.73 – 1.73] |
| 16,777,216 | 2.8 us | 3.7 us | 1.32x [1.29 – 1.35] | 4.2 us | 1.49x [1.46 – 1.53] |
| overall |  |  | 3.52x [3.51 – 3.52] |  | 6.36x [6.35 – 6.37] |

`random, scalars=many`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 791.7 ns | 6.1 us | 7.67x [7.53 – 7.84] | 9.9 us | 12.5x [12.4 – 12.5] |
| 64 | 1.2 us | 16.0 us | 13.9x [13.8 – 14.6] | 19.3 us | 16.7x [16.6 – 16.9] |
| 256 | 1.5 us | 24.8 us | 16.5x [16.4 – 17.3] | 27.7 us | 18.5x [18.4 – 18.6] |
| 1,024 | 1.9 us | 33.8 us | 17.7x [17.5 – 18.5] | 36.7 us | 19.2x [19.0 – 19.3] |
| 4,096 | 2.1 us | 10.4 us | 4.93x [4.88 – 4.99] | 32.1 us | 15.2x [15.1 – 15.2] |
| 16,384 | 3.3 us | 24.4 us | 7.32x [6.98 – 7.52] | 44.9 us | 13.5x [13.4 – 13.5] |
| 65,536 | 4.6 us | 48.1 us | 10.4x [10.3 – 10.5] | 55.4 us | 12.0x [11.9 – 12.0] |
| 262,144 | 7.5 us | 62.6 us | 8.38x [8.28 – 8.53] | 66.6 us | 8.91x [8.88 – 8.97] |
| 1,048,576 | 12.6 us | 120.1 us | 9.57x [9.51 – 9.60] | 78.5 us | 6.25x [6.23 – 6.29] |
| 4,194,304 | 30.2 us | 228.8 us | 7.57x [7.47 – 7.66] | 100.2 us | 3.31x [3.29 – 3.34] |
| 16,777,216 | 55.0 us | 431.5 us | 7.85x [7.77 – 7.91] | 142.1 us | 2.58x [2.55 – 2.61] |
| overall |  |  | 9.48x [9.43 – 9.55] |  | 9.87x [9.85 – 9.89] |

All distributions: 5.77x [5.76 – 5.80] over naive C, 7.92x [7.91 – 7.93] over NumPy

With the output buffer 16 bytes behind its input (64 needles; "apart" is the same kernel with the output clear of the input):

`random`:

| n | kernel | apart | speedup [95%] | naive C | speedup [95%] |
|---|---|---|---|---|---|
| 4,096 | 134.0 ns | 133.9 ns | 1.00x [1.00 – 1.00] | 499.7 ns | 3.73x [3.73 – 3.73] |
| 16,384 | 183.9 ns | 184.9 ns | 1.01x [1.00 – 1.01] | 654.8 ns | 3.56x [3.56 – 3.56] |
| 65,536 | 251.9 ns | 249.1 ns | 0.99x [0.99 – 0.99] | 827.3 ns | 3.28x [3.28 – 3.28] |
| 262,144 | 325.6 ns | 324.8 ns | 1.00x [1.00 – 1.00] | 1.1 us | 3.33x [3.33 – 3.33] |
| 1,048,576 | 456.2 ns | 678.0 ns | 1.49x [1.48 – 1.49] | 1.2 us | 2.66x [2.66 – 2.66] |
| 4,194,304 | 1.4 us | 1.7 us | 1.15x [1.14 – 1.15] | 2.0 us | 1.36x [1.35 – 1.36] |
| 16,777,216 | 2.8 us | 2.8 us | 1.00x [0.99 – 1.00] | 3.4 us | 1.23x [1.22 – 1.24] |
| overall |  |  | 1.08x [1.08 – 1.08] |  | 2.52x [2.52 – 2.52] |

No stall; the output here sits behind the needles, which are read as often as it is written.

What holds it:

- **In the caches:** about 0.9 cycles per needle and step: a gather every two to three cycles, and about ten more instructions per four searches for the load, the mask, the compare, the add, the store and the loop.
- **Beyond the second-level cache:** the misses. With 1,000 needles a search costs 12.6 ns at 1,048,576 elements and 55 ns at 16,777,216, against 4.6 ns at 65,536.
- **64 needles against a large array:** 1.3x to 2.0x over the naive C from 1,048,576 up. The judge repeats the same 64 needles, so the naive C's branches are learnt and its probes are in the cache; a caller's needles are new each time.

The candidates the design lists for this family (`docs/x86-64-design.md`):

| Candidate | Outcome |
|---|---|
| `vgatherqpd` loads four probes in one instruction, the compare and the move of the base stay in vectors | Kept (attempts 1 and 2) |
| Scalar loads with `cmov`, sixteen or more searches interleaved | Kept only for one to three needles. Not tried for more: a search step is about six instructions in scalar form with its position in memory, against about four per search with a gather, and the scalar form was not expected to win. Not measured |
| NumPy's fused or unfused multiply-add | `interp_f64` only; see that log |
| Gathers are slow on older AMD processors and on Intel processors with the gather mitigation | Used, in every path from four needles up. Untested elsewhere |
| Masked loads and stores for leftover elements | Not needed: a leftover is one more vector that overlaps the one before |
| Non-temporal stores; alignment of data | Not tried: the output is as small as the needles, and the probes are single elements |
| An output just behind the input | No cost in the judge's close table |
| Alignment of code | The probe loop starts on a 32-byte boundary. Not varied |

**What differs from the ARM64 kernel for a caller.**

- **Fewer than a group's worth of needles** (issue 39, issue 65): there is no padded group here. One to three needles are searched one at a time; four and more go through the same code as a thousand. The judge still times no call with fewer than 64 needles.
- **Unsorted input:** the result is meaningless, as on ARM64, and the kernel stays inside its buffers by construction: `lo + len <= n` holds at every step for any data.
- **`out` is used as working storage** during the call: an element holds the search's current position until its last step.

Left untried: the first two or three steps without a gather (their probes are one of two, then four, known elements); the probe loop unrolled; a prefetch of both possible next probes.

Found here and worth trying on ARM64: nothing that carries over; the ARM64 kernel already chooses its group width by the size of `a`.
