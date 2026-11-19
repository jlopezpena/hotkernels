# bincount_i64 attempts (arm64)

All runs on an Apple M1 Pro (macOS, native), locked, while other kernel runs were timing on the same machine, so timings from about 100,000 elements up are noisier than usual.

This kernel was written straight after `bincount_i16` on the same machine, whose log beside its `kernel.s` has the reasoning and the dead ends. Each idea kept there was tried again here and judged on its own.

What the naive C costs here (Apple clang, `-O3`): on codes uniform in [0, 64) 0.35 ns per element from 4,096 up; on four distinct values 0.93 ns per element, where an increment waits for the store of the previous increment of the same counter.

## 1. Baseline: clear 64 bytes per step, eight elements per step

- **Idea:** `bincount_i16` as it stood after its attempt 2, with 8-byte elements read by `ldp`: each compared with `bins` as an unsigned number (a negative one is a huge one), a branch, and a load, add and store of its counter. The counters are cleared with `stp q0, q0`, 64 bytes per step, the last step ending exactly at the end of `counts`.
- **check:** pass
- **compare:** n/a (first version). `run`: 1.03x [1.03 – 1.04] over the naive C overall, 4.01x over NumPy.
- **Kept:** yes
- **Notes:** uniform codes 1.10x–1.13x from 1,024 up (0.31 ns per element against 0.35); four values 0.97x from 1,024 up (0.95 ns against 0.93), where both wait for the counters; 0.89x at 16 elements with 64 counters (18.9 ns against 16.8).

## 2. Four tables of 32-bit counters on the stack, from 8 elements per counter

- **Idea:** attempt 3 of `bincount_i16`: element k of each group of four is counted in table k of `bins` 32-bit counters below the stack pointer (at most 16 KB, 4 to 1,024 counters), the four summed and widened over `counts` at the end. Each element keeps its own test against `bins`. n < 2^32.
- **check:** pass
- **compare:** significantly faster, 1.343x [1.342 – 1.345]
- **Kept:** yes
- **Notes:** four values 2.24x–2.36x with 64 counters from 1,024 up (0.42 ns per element against 0.95), 2.33x–2.53x with 1,000 counters from 16,384 up. Uniform codes 1.07x–1.10x from 65,536 up (0.28 ns per element), but 0.98x at 1,024 with 64 counters and 0.95x at 16,384 with 1,000, where the set-up is not repaid.

## 3. A table of pairs for at most 64 counters, from 8,192 elements

- **Idea:** attempt 4 of `bincount_i16`, for 8-byte elements: four elements from two `ldp`, or'ed together and compared with 64 once (the or is below 64 as an unsigned number only if each is, so a negative element and one with any high bit set both fail), and each pair index one `orr x, a0, a1, lsl #6`. Four that are not all pair codes go one at a time into `counts`; the totals are taken as in `bincount_i16`. First from 4,096 elements, as `bincount_i16` has it: 0.89x at 4,096 uniform codes, so from 8,192.
- **check:** pass
- **compare:** significantly faster, 1.072x [1.071 – 1.072]
- **Kept:** yes
- **Notes:** uniform codes with 64 counters 1.09x at 16,384, 1.25x–1.26x from 1,048,576 up (0.228 ns per element against 0.288); four values 1.23x at 16,384, 1.39x–1.40x from 65,536 up (0.304 ns against 0.423). Less than `bincount_i16` gets (0.139 ns per element there): twice the loads of input and three `orr` for the test.

## 4. Four-table path: one test per four elements

- **Idea:** the counterpart of `bincount_i16` attempt 6. A vector test buys less with 8-byte elements (two per vector, and `umaxv` has no 64-bit form), so the four elements of each two `ldp` are tested with one `cmp` and three `ccmp` (each against `bins`, unsigned, the carry kept set once one fails) and one branch; four that pass are counted with no test of their own, four that do not go one at a time, out of line.
- **check:** pass
- **compare:** significantly faster, 1.003x [1.003 – 1.004]
- **Kept:** yes
- **Notes:** uniform codes with 64 counters 1.05x at 1,024 and 4,096; with 1,000 counters 1.01x from 16,384 up (0.287 ns per element at 1,048,576); four values unchanged. The `bincount_i16` loop gained 12% with 1,000 counters from its vector test, which also replaced a sign extension per element; here the compare per element was already the only test.

## 5. Eight tables where a sample of the data repeats, from 8,192 elements

- **Idea:** attempt 8 of `bincount_i16`: a separate loop with eight tables (up to 32 KB), taken from 8,192 elements when at least 4 of the first 32 elements equal the element four places on (uniform codes over 64 values have about 0.5 such, four values about 8). The sample is 18 vector loads and 16 `cmeq` on 64-bit lanes, no loop. Every free integer register holds a table base; the frame size is kept in `d30`.
- **check:** pass
- **compare:** significantly faster, 1.027x [1.026 – 1.028]
- **Kept:** yes
- **Notes:** four values with 1,000 counters 1.13x at 16,384, 1.22x–1.24x from 65,536 up (0.29 ns per element at 1,048,576 against 0.36). Every other point 1.00x.
