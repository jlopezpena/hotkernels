# cumsum_i64 attempts

This log holds two architectures. ARM64 (Apple M1 Pro) comes first. x86-64 (Intel Core i5-14600KF) follows under [x86-64](#x86-64), with its own numbering.

All timings on an Apple M1 Pro, with `--unlocked` (the spec was not on `main`
yet) and with other agents timing other kernels on the same machine, so the
numbers are noisier than a quiet run.

Reference point: the naive C (`clang -O3`, unrolled eight times by the
compiler, still one `add` per element on a single chain) runs at about
0.31 ns per element from 256 to 1,048,576 elements, which is one cycle per
element at 3.2 GHz. That is the latency of the carried `add`, not the width of
the core.

## 1. Simplest scalar loop

- **Idea:** one `ldr`/`add`/`str` per element with post-indexed pointers and a
  down-counter; the starting point, to get `check` passing.
- **check:** pass
- **compare:** not run (first version). `judge run`: 0.80x [0.80 – 0.81] of
  naive C overall, 0.56x at n=16 and about 0.85x from 4,096 up.
- **Kept:** yes (baseline)
- **Notes:** slower than the C because the C is unrolled eight times and this
  pays the loop branch on every element (0.37 ns against 0.31 ns per element).

## 2. Scalar blocks of four with the running sum off the critical path

- **Idea:** load four elements with two `ldp`, build the block's partial sums
  from its inputs only (a+b, c+d, a+b+c+d), add the running sum when storing,
  and advance the running sum with one add per block. The carried chain drops
  from one add per element to one add per four.
- **check:** pass
- **compare:** significantly faster, 1.96x [1.95 – 1.99]
- **Kept:** yes
- **Notes:** 2.0x at n=16, 2.2x from 256 to 262,144 (0.172 ns per element,
  about 0.55 cycles), 2.09x at 1,048,576, and 1.35x at 4,194,304 and
  16,777,216, where it runs at 0.29 ns per element and is limited by memory.
  The tail of 0 to 3 elements is the scalar loop from attempt 1.

## 3. Two blocks of four per iteration

- **Idea:** the loop of attempt 2 spends 3 of its 15 instructions on pointer
  and counter updates; doing two blocks per iteration halves that. The running
  sum still advances by one add per block (two per iteration).
- **check:** pass
- **compare:** significantly faster, 1.07x [1.07 – 1.08]
- **Kept:** yes
- **Notes:** 1.10x to 1.15x from 16 to 16,384 (0.152 ns per element, about
  0.49 cycles). No gain from 65,536 up (1.01x, 1.02x, then not significant):
  once the two arrays no longer fit in the first-level cache both versions
  run at about 0.177 ns per element, so the instruction count is no longer
  what limits them. One leftover block of four, then the scalar tail.

## 4. NEON, two lanes, carry as [even sum, odd sum]

- **Idea:** blocks of four vectors. Keep `Rc = [sum of even elements, sum of
  odd elements]`; with `R_k = Rc + (v_0 + ... + v_k)`, output vector k is
  `R_k + ext(R_{k-1}, R_k, #8)`, and the next carry is `R_3` (one add per
  block on the carried chain). 15 NEON operations per 8 elements.
- **check:** pass
- **compare:** significantly slower, 0.97x [0.96 – 0.97]
- **Kept:** no
- **Notes:** 0.92x to 0.94x from 256 to 65,536, no difference above. 15 vector
  operations per 8 elements on four vector pipes cannot go below 3.75 cycles;
  the scalar loop of attempt 3 already runs at about 3.9.

## 5. NEON with overlapping loads, vectors on odd elements

- **Idea:** `out[i..i+1] = out[i-2..i-1] + x[i..i+1] + x[i-1..i]`, so the step
  between output vectors is the sum of two loads, the second shifted back one
  element. That removes the `ext`s: 11 NEON adds per 8 elements, at the cost
  of 8 loads. Element 0 was done in scalar code and the vectors started at
  element 1.
- **check:** pass
- **compare:** significantly slower, 0.92x [0.91 – 0.92]
- **Kept:** no
- **Notes:** slower at every size (0.83x to 0.97x). Starting the vectors at
  element 1 puts every 16-byte store 8 bytes off the alignment of `out`, so
  one store in four straddles a cache line. Fixed in attempt 6.

## 6. NEON with overlapping loads, vectors on even elements, 16 per iteration

- **Idea:** the form of attempt 5 with the vectors on even elements, so loads
  and stores keep the 16-byte alignment the arrays usually have. The first
  shifted vector of each block comes from `ext` of the previous input vector
  (zero before the start, which also covers element 0 without reading
  `x[-1]`); the other three are overlapping loads. Two blocks of four vectors
  per iteration; one more block, then the scalar block of four and the scalar
  tail, with the running sum moved out of the vector register only when there
  is a tail.
- **check:** pass
- **compare:** significantly faster, 1.06x [1.06 – 1.06] with a noisy-machine
  warning; rerun: 1.07x [1.07 – 1.08]
- **Kept:** yes
- **Notes:** 1.10x at 16, 1.05x at 64, 1.13x at 256, 1.18x at 1,024, 1.15x at
  4,096, 1.11x at 16,384 (0.130 ns per element, about 3.35 cycles per 8
  elements against 3.9), 1.02x to 1.04x at 65,536, no significant difference
  from 1,048,576 up.
  Measurements behind the design, from a standalone loop timer (not the
  judge; cycles per 8 elements at n=4,096):
  - store only, 4 x 16 bytes: 2.2; load only: 1.4; plain copy: 2.3 to 2.8.
    The copy is the floor for anything that loads and stores 16-byte units.
  - attempt 3 loop: 3.9; the same without its stores: 3.2.
  - synthetic 4-load 4-store loop plus k independent scalar adds: 2.7 (k=4),
    3.1 (8), 3.7 (12), 3.9 (14). Half a cycle per four adds, so the scalar
    loop is limited by the number of micro-operations (about 7.3 per cycle,
    with `ldp` of two x registers counting as two), not by the carried chain.
  - the same with NEON adds: 2.7 (k=8), 3.25 (12), 4.2 (16): four vector
    operations per cycle at most.
  - overlap form, single block per iteration, with 1, 2, 3 or 4 of the shifted
    vectors loaded and the rest from `ext`: 3.78, 3.58, 3.58, 3.56. Two blocks
    per iteration: 3.53 (2 loaded), 3.46 (3 loaded), 3.50 (4 loaded).
  - scalar alternatives that were measured but not taken to the judge because
    this form beat them: attempt 3 unrolled to 16 per iteration, 3.65 to 3.70;
    a 13-add variant with the carried chain three adds deep per 8 elements,
    3.76 to 3.82.

## 7. Software prefetch of the input

- **Idea:** two `prfm pldl1keep` per iteration, 512 bytes ahead of the loads,
  to help once the data no longer fits in the first-level cache.
- **check:** pass
- **compare:** significantly slower, 0.98x [0.97 – 0.99]
- **Kept:** no
- **Notes:** 0.94x to 0.97x from 16 to 4,096 and 0.95x at 65,536 (two extra
  instructions per iteration and nothing gained); no significant difference
  from 1,048,576 up. In the standalone timer, distances of 256, 1,024, 2,048
  and 4,096 bytes and `pldl2keep` were no better at any size, and 2,048 and
  4,096 were far worse at 4,194,304 and above (0.69x and 0.35x). Prefetching
  the output with `pstl1keep` was worse everywhere (0.80x at 65,536, 0.27x at
  16,777,216, measured on the attempt 3 loop). The hardware prefetcher already
  follows both streams.

## 8. A second loop with fewer overlapping loads for large arrays

- **Idea:** from 65,536 elements up the loop of attempt 6 slows from 3.35 to
  about 3.7 cycles per 8 elements, and there the variant that loads two of the
  shifted vectors and builds two with `ext` is faster (it is slower in the
  first-level cache: 3.44 against 3.36). Use the attempt 6 loop below 32,768
  elements and the two-load loop from there up.
- **check:** pass
- **compare:** five runs, because the gain is small and the machine was busy:
  significantly faster 1.008x [1.003 – 1.011]; no significant difference
  1.002x [0.972 – 1.017]; no significant difference 1.008x [0.998 – 1.019];
  significantly faster 1.014x [1.002 – 1.020]; significantly faster 1.007x
  [1.003 – 1.017].
- **Kept:** yes, on three significant runs out of five with the same point
  estimate in all of them. This is a marginal keep.
- **Notes:** the whole effect is at three sizes: 1.04x to 1.05x at 65,536,
  1.03x to 1.04x at 262,144 and about 1.02x at 1,048,576 in the quiet runs.
  Sizes below the threshold run the same code as before and measure 1.00x. At
  4,194,304 and 16,777,216 the intervals were wide in every run (other jobs
  were using memory bandwidth) and the two runs that were not significant
  were the ones where those two sizes were noisiest.

## Screened in the standalone timer only (not taken to the judge)

These were measured with the loop timer described under attempt 6 and were
neutral or worse there, so no judge run was spent on them. Cycles are per 8
elements; the attempt 6 loop is 3.35 at sizes up to 16,384.

- **Four blocks (32 elements) per iteration:** 3.33 to 3.37 up to 4,096, so no
  gain from the smaller loop overhead, and 5.6 at 16,384 and above (0.61x).
  The same collapse appeared when each `ldp`/`stp` of q registers was split
  into two single loads and stores (5.5 at 16,384). Both variants have more
  load instructions per iteration (20 and 14 against 10), which is the only
  thing they share; the cause was not established.
- **One add fewer per block, carried chain two adds deep:** using
  `out vector 3 = out vector 1 + steps 2..3` saves an add but puts two vector
  adds (2 cycles each) on the carried chain. In every block: 4.0 to 4.2
  (0.81x). In one block of two: 3.43 to 3.51 (0.96x).
- **Other mixes of loaded and `ext` shifted vectors:** two loaded in both
  blocks 3.44 to 3.49; three in one block and two in the other 3.43 to 3.48;
  one loaded 3.65 to 3.72. Three loaded (attempt 6) is the best in cache.
- **Loop alignment** to 16, 32 or 64 bytes: 3.33 to 3.37, no change (64 bytes
  cost 10% at n=16).
- **`stnp` in place of `stp`:** no difference at any size, including
  4,194,304 and 16,777,216 (0.193 ns per element with either).
- **`dc zva` on each output line before writing it**, to avoid reading the
  line in: 260 to 300 cycles per line, about 70 times slower. Zeroing four
  lines ahead: 22 cycles per line.
- **`ld2`/`st2`** to de-interleave even and odd elements (pair sums, then a
  prefix over half as many values, then even outputs by subtraction): a copy
  loop with two `ld2` and two `st2` per 64 bytes takes 4.1 cycles against 2.3
  for `ldp`/`stp`, more than the whole current loop.
- **Removing one resource at a time from the attempt 6 loop** (results wrong,
  timing only) shows what it is limited by: one vector add fewer per block
  3.15 (-0.22), the three overlapping loads removed 3.20 (-0.18), the `ext`
  removed 3.26 (-0.12), one of the two `stp` removed 3.14 (-0.24), no stores
  3.08. Vector operations and stores are the expensive parts; the loop is at
  12 vector operations per 8 elements on four vector pipes.

## Where it stands

- Final `judge run` against naive C, on a run without a noise warning: 1.90x
  [1.89 – 1.90] overall. By size: 1.36x at 16, 1.56x at 64, 2.15x at 256,
  2.40x at 1,024, 2.41x at 4,096, 2.39x at 16,384, 1.88x at 65,536, 1.91x at
  262,144, 1.95x at 1,048,576, 1.51x at 4,194,304, 1.70x at 16,777,216.
- In the first-level cache: about 3.35 cycles per 8 elements (0.13 ns per
  element) against 8 cycles for the one-add-per-element chain. A plain copy
  with 16-byte loads and stores takes 2.3 to 2.8, so at most a fifth is left.
- At 4,194,304 and 16,777,216 elements the kernel, the scalar version of
  attempt 3 and a plain copy all run at 0.19 to 0.21 ns per element: memory
  speed, not computation.
- Not tried: blocks of three vectors (the vector operation count per element
  is the same as with four, so no gain was expected), and carrying the
  running sum in a general register with a broadcast per block (three more
  micro-operations per block for a chain that is not the limit).

# x86-64

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, locked, pinned to one performance core by the judge.

What the compiler does with `naive.c` (`clang -O3 -march=native`): the scalar chain, one add and one store per element, one cycle each: 0.19 ns per element at every size up to 262,144. NumPy takes 1.4 ns per element at 1,048,576.

**What there is to win, measured before anything was kept.** A micro-benchmark outside the judge, 16,384 elements (128 KB in, 128 KB out, in the second-level cache): a plain copy loop of 32-byte loads and stores takes 0.115 ns per element when both buffers are aligned and 0.124 when they are not. The naive C takes 0.19. So beyond the first-level cache no running sum can be more than about 1.65x faster than the naive C on this machine; inside it, about 2x was reached (0.094 ns per element). The design expected this family to reach about half its ARM64 figures; the reason is this ceiling, and a scalar chain that costs one cycle here against three on the M1.

## x86-64 1. Baseline: window sums from overlapping loads, one add per vector on the chain

- **Idea:** the ARM64 kernel's regrouping in 32-byte vectors. `D` = the input vector plus the same load one element back. `W` = `D` plus `D` two elements back, which is the upper half of the previous `D` and the lower half of this one: one `vperm2i128`. The output vector is the previous one plus `W`, so the chain is one `vpaddq` per four elements. The design's worry was the extra permute for a prefix sum across the register (`vpslldq` works inside each 128-bit half); the overlapping load avoids all but one. The `D` of later vectors is loaded before the current output is stored, which keeps the kernel right when `out` is `x`. Before the array every value counts as zero.
- **check:** pass
- **compare:** not run (first version). `run`: 1.09x over naive C overall, 12.9x over NumPy.
- **Kept:** yes
- **Notes:** 0.165 ns per element from 1,024 to 65,536 in the judge. In the micro-benchmark the same code takes 0.095 with every buffer aligned inside the first-level cache, and at 16,384 elements 0.134 aligned, 0.161 with x 16 bytes off and 0.172 with both off: a 32-byte load that straddles two cache lines takes a cycle more and a store two, and the load one element back straddles at every second vector whatever the alignment. Knock-out variants (no store, no permute, no chain) all ran at 1.8 to 2 cycles per vector: four vector instructions per vector on three ports, and two 32-byte loads per cycle at most.

## x86-64 2. Align the stores to 32 bytes

- **Idea:** 0 to 3 elements one at a time until `out` is a multiple of 32, then the vectors, with the sum so far as the output before the first.
- **check:** pass
- **compare:** no significant difference, 0.999x [0.991 – 1.007]
- **Kept:** no
- **Notes:** the judge places the output 16 bytes after the input counted within a cache line, so aligning one buffer leaves the other 16 bytes off: with the stores aligned, two of every four loads straddle. 1.06x to 1.07x at 16,384 and 65,536, 0.84x at 16.

## x86-64 3. 16-byte vectors at every size

- **Idea:** two elements per vector: `D` = input plus the load one element back, output = previous output plus `D`. No permute, a chain of one add per two elements, and 16-byte accesses at 8-byte offsets rarely straddle a line. The loads run two vectors ahead of the stores, which also keeps a store 16 bytes behind the input (within a page) from holding up the loads after it.
- **check:** pass
- **compare:** significantly slower, 0.991x [0.988 – 0.995]
- **Kept:** no
- **Notes:** the sizes disagree: 1.05x at 1,024, 1.04x at 4,096, 1.13x at 16,384 and 65,536; 0.87x to 0.92x from 16 to 256, where its twice as many steps count; 0.96x to 0.97x at 4,194,304 and 16,777,216. In the micro-benchmark it takes 0.094 ns per element inside the first-level cache and 0.146 at 16,384, the same at every alignment.

## x86-64 4. 16-byte vectors from 512 to 131,071 elements

- **Idea:** attempts 1 and 3 by size. The wide version for short arrays and for arrays that do not fit the second-level cache together with their output; the narrow one in between.
- **check:** pass
- **compare:** significantly faster, 1.023x [1.018 – 1.028]
- **Kept:** yes
- **Notes:** 1.05x at 1,024, 1.03x at 4,096, 1.13x at 16,384, 1.15x at 65,536, nothing elsewhere except 16 elements, 0.91x: two more compares on the way in. With the wide path reached by fall-through; with a taken branch it was 0.88x there and 1.021x overall.

## x86-64 5. Stores past the caches from 24 MB of output

- **Idea:** the design's candidate. `vmovntdq` does not read the line it is about to overwrite, which removes a third of the memory traffic. It needs a 32-byte-aligned address, so the head of attempt 2 comes back for this path only, and the vectors that are left at the end use ordinary stores after an `sfence`. The threshold is the third-level cache: at a size no cache holds, it costs a caller nothing that the output is in no cache afterwards.
- **check:** pass (after one failure: the narrow path fell through into the new code). The judge's correctness cases stop far below 3,145,728 elements, so this path was tested outside it: every path, 43 sizes up to 3,200,003, eight alignments of x and five of out including one that is not a multiple of 8, against the scalar sum, and with `out` = `x`. No failure.
- **compare:** significantly faster, 1.044x [1.040 – 1.049]
- **Kept:** yes
- **Notes:** 1.26x at 4,194,304 and 1.25x at 16,777,216, nothing elsewhere. In the micro-benchmark the same stores are 1.3x faster already at 262,144 and 1,048,576 elements (0.175 and 0.185 ns per element against 0.227 and 0.243) and slower at 65,536. They are not used there: an output of 2 to 24 MB written with ordinary stores is left in the third-level cache for the caller, and one written this way is not.

## x86-64: where it stands

Final `judge run`:

`random`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 2.7 ns | 3.0 ns | 1.12x [1.11 – 1.13] | 720.6 ns | 271x [271 – 272] |
| 64 | 11.7 ns | 12.7 ns | 1.08x [1.08 – 1.09] | 764.3 ns | 65.1x [65.0 – 65.1] |
| 256 | 40.9 ns | 45.5 ns | 1.11x [1.11 – 1.11] | 1.1 us | 26.9x [26.8 – 26.9] |
| 1,024 | 164.2 ns | 191.4 ns | 1.17x [1.17 – 1.17] | 2.2 us | 13.4x [13.3 – 13.4] |
| 4,096 | 625.4 ns | 778.9 ns | 1.25x [1.25 – 1.25] | 6.6 us | 10.5x [10.5 – 10.5] |
| 16,384 | 2.5 us | 3.1 us | 1.23x [1.23 – 1.23] | 24.1 us | 9.58x [9.58 – 9.58] |
| 65,536 | 10.0 us | 12.4 us | 1.24x [1.24 – 1.24] | 94.3 us | 9.44x [9.44 – 9.46] |
| 262,144 | 60.8 us | 63.9 us | 1.05x [1.05 – 1.05] | 379.5 us | 6.24x [6.22 – 6.25] |
| 1,048,576 | 289.2 us | 268.1 us | 0.93x [0.92 – 0.94] | 1.50 ms | 5.19x [5.16 – 5.26] |
| 4,194,304 | 1.83 ms | 2.45 ms | 1.34x [1.34 – 1.36] | 7.83 ms | 4.28x [4.24 – 4.29] |
| 16,777,216 | 7.37 ms | 10.09 ms | 1.37x [1.37 – 1.38] | 30.80 ms | 4.18x [4.17 – 4.20] |
| overall |  |  | 1.17x [1.16 – 1.17] |  | 13.8x [13.8 – 13.8] |

With the output buffer 16 bytes behind its input ("apart" is the same kernel with the output clear of the input):

`random`:

| n | kernel | apart | speedup [95%] | naive C | speedup [95%] |
|---|---|---|---|---|---|
| 4,096 | 629.0 ns | 653.3 ns | 1.04x [1.04 – 1.04] | 779.3 ns | 1.24x [1.24 – 1.24] |
| 16,384 | 2.5 us | 2.5 us | 1.00x [1.00 – 1.01] | 3.1 us | 1.24x [1.24 – 1.24] |
| 65,536 | 10.0 us | 10.3 us | 1.03x [1.03 – 1.03] | 12.4 us | 1.23x [1.23 – 1.23] |
| 262,144 | 67.0 us | 65.0 us | 0.97x [0.97 – 0.97] | 450.0 us | 6.72x [6.71 – 6.73] |
| 1,048,576 | 393.3 us | 517.4 us | 1.32x [1.30 – 1.33] | 1.45 ms | 3.70x [3.69 – 3.71] |
| 4,194,304 | 1.70 ms | 1.84 ms | 1.08x [1.07 – 1.09] | 2.43 ms | 1.43x [1.42 – 1.44] |
| 16,777,216 | 7.20 ms | 7.38 ms | 1.02x [1.02 – 1.03] | 10.06 ms | 1.40x [1.39 – 1.40] |
| overall |  |  | 1.06x [1.06 – 1.06] |  | 1.91x [1.91 – 1.92] |

No stall.

What holds it at each size:

- **16 to 256:** the steps of a call and the cold start of each; 1.1x over the naive C.
- **1,024 to 65,536:** the second-level cache. 0.15 to 0.16 ns per element against 0.115 to 0.124 for a plain copy, 1.17x to 1.25x over the naive C.
- **262,144 and 1,048,576:** the third-level cache: 1.05x and 0.93x against the naive C.
- **4,194,304 and 16,777,216:** memory, with a third less traffic than the naive C: 1.34x and 1.37x.

The candidates the design lists for this family (`docs/x86-64-design.md`):

| Candidate | Outcome |
|---|---|
| A prefix sum across the register with an extra permute (`vpslldq` works inside each half) | Kept in another form (attempt 1): the shift by one element is an overlapping load, and one `vperm2i128` per vector does the rest. Between 512 and 131,071 elements the 16-byte version without any permute is faster (attempt 4) |
| Non-temporal stores above the third-level cache | Kept (attempt 5) |
| The overflow test as a compare | Does not apply to integers |
| Masked loads and stores for leftover elements | Not tried: open issue 3 of the design. The last elements are done one at a time |
| An output just behind the input | No cost in the judge's close table: the loads run ahead of the stores |
| Alignment of data | Rejected as a general step (attempt 2); used where the stores need it (attempt 5) |
| Alignment of code | The loops start on 64-byte boundaries. Not varied |
| Instructions slow on other x86-64 processors | None used |

**Unlike the float kernels, this one is right when `out` is `x`**, as the ARM64 kernel is; tested outside the judge, which has no such case. A partial overlap is not supported by any vector kernel.

Left untried: the cache-bypassing stores between 2 MB and 24 MB of output (1.3x in the micro-benchmark, at the cost named in attempt 5: a person's call); a prefetch.

Found here and worth trying on ARM64: nothing. The M1's scalar chain is three times slower relative to its vector loop, so its kernel is not near a copy's speed in cache.
