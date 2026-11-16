# cumsum_i64 attempts

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
