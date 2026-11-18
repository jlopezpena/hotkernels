# cumsum_f64 attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, locked, pinned to one performance core by the judge. Read with the `cumsum_i64` log, where the regrouping and its ceiling on this machine were worked out: beyond the first-level cache a plain copy takes 0.115 ns per `float64`, and no running sum can be faster.

What the compiler does with `naive.c` (`clang -O3 -march=native`): the scalar chain, one `vaddsd` per element at two cycles each: 0.38 ns per element from 1,024 to 1,048,576. NumPy takes 1.7 ns per element at 1,048,576.

## 1. Baseline: window sums from overlapping loads, a magnitude test per chunk

- **Idea:** the regrouping of `cumsum_i64`'s wide version in floating point: `D` = input plus the load one element back, `W` = `D` plus `D` two elements back (one `vperm2f128`), output = previous output plus `W`. The chain is one `vaddpd` per four elements, two cycles, where the reference takes two per element. Before the array every value is -0.0. **The proof that non-finite values land where NumPy has them is the ARM64 kernel's, with the design's candidate in place of the sum of squares:** per chunk of 1,024 elements the largest output magnitude is tracked with `vpand` and `vpmaxud` on the bit patterns, and the chunk is accepted if every output is below 2^1023. A chunk that fails is redone one element at a time under watch, and from there the ARM64 kernel's three cases apply unchanged (store, finish in the plain order, or restart from element 0). The argument is in the `kernel.s` header. Fewer than 8 elements take the plain chain.
- **check:** pass
- **compare:** not run (first version). `run`: 1.78x over naive C overall (2.6x to 3.0x from 1,024 to 65,536), 15.7x over NumPy.
- **Kept:** yes
- **Notes:** **the close-placement table showed a stall:** with the output 16 bytes behind the input, 0.40x at 262,144 and 0.44x at 1,048,576 against the same kernel with the output clear, and nothing at the other sizes. The naive C loses about as much there.

## 2. Loads ahead of the stores, and stores past the caches from 24 MB

- **Idea:** two things from `cumsum_i64`. The `D` of the next vector is loaded before the current output is stored, so that no load follows a store it could be mistaken for a read of. And from 24 MB of output the vectors are stored with `vmovntpd`, after 0 to 3 elements in the reference order that bring `out` to a 32-byte boundary; their sum takes the place of -0.0 as the output before the first vector.
- **check:** pass
- **compare:** no significant difference, 0.995x [0.989 – 1.001]
- **Kept:** no, in this form
- **Notes:** 1.18x at 4,194,304 and 1.21x at 16,777,216, 1.00x in between, and 0.69x at 16 elements: 4.1 ns against 2.8. Neither a last-pair step that avoids two single-vector steps (1.000x overall, 0.73x at 16) nor deciding the kind of store from the registers instead of a flag below the stack pointer (0.996x, 0.67x) moved it. In a micro-benchmark with warm caches the same two versions differ by 13% at 16 elements, not 50%; the judge cycles through several inputs at small sizes.

## 3. The same, with the baseline's loop kept for fewer than 32 elements

- **Idea:** attempt 2 for 32 elements and more, and below that one chunk of the baseline's vectors: nothing loaded ahead and no choice of store.
- **check:** pass. The paths the judge does not reach (the cache-bypassing stores, a chunk redone and followed by the fast path, the restart) were tested outside it: 20,990 runs against the left-to-right sum over 41 sizes and five large ones up to 3,200,005, at random alignments, on random data, small integers, -0.0, one or two infinities or NaN at random places, values near the largest double and near 1e154. Positions of NaN and of each infinity must match exactly, `out[0]` must be a copy of `x[0]`, and finite values must be within the spec's tolerance. No failure.
- **compare:** significantly faster, 1.027x [1.019 – 1.035]
- **Kept:** yes
- **Notes:** 1.20x at 4,194,304, 1.21x at 16,777,216, no significant difference anywhere else. The close-placement stall is gone: 1.00x at 262,144 (table below).

## Where it stands

Final `judge run`:

`random`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 3.0 ns | 3.1 ns | 1.03x [1.02 – 1.04] | 725.8 ns | 242x [238 – 245] |
| 64 | 10.4 ns | 14.9 ns | 1.43x [1.43 – 1.43] | 781.7 ns | 75.2x [75.1 – 75.3] |
| 256 | 37.4 ns | 85.0 ns | 2.27x [2.27 – 2.27] | 1.1 us | 30.5x [30.5 – 30.6] |
| 1,024 | 146.4 ns | 381.3 ns | 2.60x [2.60 – 2.61] | 2.5 us | 16.8x [16.7 – 16.8] |
| 4,096 | 574.4 ns | 1.6 us | 2.74x [2.73 – 2.74] | 7.9 us | 13.8x [13.7 – 13.8] |
| 16,384 | 2.1 us | 6.3 us | 3.01x [3.01 – 3.01] | 28.9 us | 13.8x [13.8 – 13.8] |
| 65,536 | 9.0 us | 25.2 us | 2.81x [2.81 – 2.81] | 113.4 us | 12.6x [12.6 – 12.6] |
| 262,144 | 64.3 us | 101.4 us | 1.58x [1.57 – 1.58] | 449.0 us | 6.98x [6.96 – 6.99] |
| 1,048,576 | 293.8 us | 412.4 us | 1.40x [1.39 – 1.42] | 1.79 ms | 6.10x [6.03 – 6.16] |
| 4,194,304 | 1.95 ms | 2.53 ms | 1.30x [1.29 – 1.32] | 9.08 ms | 4.65x [4.64 – 4.67] |
| 16,777,216 | 7.98 ms | 10.41 ms | 1.30x [1.30 – 1.31] | 36.07 ms | 4.52x [4.51 – 4.54] |
| overall |  |  | 1.83x [1.82 – 1.83] |  | 16.2x [16.2 – 16.2] |

With the output buffer 16 bytes behind its input ("apart" is the same kernel with the output clear of the input):

`random`:

| n | kernel | apart | speedup [95%] | naive C | speedup [95%] |
|---|---|---|---|---|---|
| 4,096 | 528.3 ns | 587.1 ns | 1.11x [1.11 – 1.11] | 1.6 us | 2.98x [2.97 – 2.98] |
| 16,384 | 2.1 us | 2.3 us | 1.08x [1.08 – 1.08] | 6.3 us | 3.02x [3.02 – 3.02] |
| 65,536 | 9.1 us | 9.3 us | 1.02x [1.02 – 1.03] | 25.2 us | 2.78x [2.78 – 2.79] |
| 262,144 | 65.6 us | 65.5 us | 1.00x [1.00 – 1.00] | 464.9 us | 7.09x [7.07 – 7.11] |
| 1,048,576 | 420.0 us | 525.2 us | 1.25x [1.24 – 1.27] | 1.65 ms | 3.93x [3.90 – 3.95] |
| 4,194,304 | 1.85 ms | 1.97 ms | 1.06x [1.06 – 1.07] | 2.46 ms | 1.33x [1.32 – 1.34] |
| 16,777,216 | 7.80 ms | 7.94 ms | 1.02x [1.02 – 1.02] | 10.24 ms | 1.31x [1.31 – 1.32] |
| overall |  |  | 1.08x [1.07 – 1.08] |  | 2.76x [2.76 – 2.76] |

What holds it at each size:

- **16 and 64:** the steps of a call; 16 elements are level with the naive C.
- **256 to 65,536:** 0.13 to 0.15 ns per element, against 0.115 for a plain copy beyond the first-level cache. The vector loop itself needs two cycles per four elements (0.097 ns per element): the chain is one add of two cycles, and the three adds and the permute of a vector fill the two ports that take them.
- **262,144 and 1,048,576:** the third-level cache, 1.4x to 1.6x over the naive C.
- **4,194,304 and 16,777,216:** memory, with a third less traffic than the naive C: 1.3x.

The candidates the design lists for this family (`docs/x86-64-design.md`):

| Candidate | Outcome |
|---|---|
| A prefix sum across the register with an extra permute | Kept (attempt 1): the shift by one element is an overlapping load, and one `vperm2f128` per vector does the rest |
| Non-temporal stores above the third-level cache | Kept (attempt 3) |
| The overflow test as a compare and a `vmovmskpd` instead of a sum of squares | Kept (attempt 1), as an integer maximum of the magnitudes per vector and one compare and `vmovmskpd` per chunk |
| Masked loads and stores for leftover elements | Not tried: open issue 3 of the design |
| An output just behind the input | It stalled the baseline at 262,144 and 1,048,576 (0.40x and 0.44x). Loading ahead of the stores removes it (attempt 3) |
| Alignment of data | Not tried as a general step: rejected on `cumsum_i64` (its attempt 2), and the judge's two buffers cannot both be aligned. Used where the stores need it |
| Alignment of code | The loops start on 64-byte boundaries. Not varied |
| Instructions slow on other x86-64 processors | None used |

**What differs from the ARM64 kernel for a caller.**

- **The fast path reaches further.** The ARM64 kernel's sum of squares limits it to running sums below about 1e154. The magnitude test here accepts everything below 2^1023, about 9e307.
- **The test raises no floating-point flag.** The sum of squares raises overflow on finite data once the squares overflow. The regrouped additions themselves can still raise overflow or invalid where the left-to-right sum does not, so the binding's handling of flags is still needed.
- **Everything else is the same:** not bit-identical to NumPy; adjacent outputs come from separate running sums (four here, two on ARM64), so non-negative input does not give a non-decreasing output; `out` must not overlap `x`.

Left untried: two interleaved chains (the output two vectors back plus the sum of two `W`), which halves the chain but adds half an add per vector to ports that are already full; the adds that are not on the chain done as fused multiply-adds by one to move them to another port (slower on `cumsum_f32`, its attempt 2); the cache-bypassing stores between 2 MB and 24 MB of output (see `cumsum_i64`, attempt 5).

Found here and worth trying on ARM64: the magnitude test in place of the sum of squares. It would lift the ARM64 fast path's limit from 1e154 to 9e307 for `float64` and stop the test itself from raising overflow. Whether an integer maximum per vector is as cheap there as the `fmla` is for the ARM64 run to measure.
