# bincount_i16 attempts (arm64)

All runs on an Apple M1 Pro (macOS, native), locked, while other kernel runs were timing on the same machine, so timings from about 100,000 elements up are noisier than usual.

What the naive C costs here (Apple clang, `-O3`): on codes uniform in [0, 64) 0.41 ns per element from 4,096 up; on four distinct values 1.35 ns per element, about 4.3 cycles: an increment waits for the store of the previous increment of the same counter, and with four values that is most elements. With 1,000 counters the call costs 97 ns at 16 elements, nearly all of it clearing 8 KB of counters.

## 1. Baseline: one element per step

- **Idea:** the reference by hand. Clear the counters one 8-byte store at a time; per element a sign-extending load, an unsigned compare with `bins` (a negative element is a huge one), a branch, and a load, add and store of its counter.
- **check:** pass
- **compare:** n/a (first version). `run`: 0.87x [0.87 – 0.87] of the naive C overall, 3.01x over NumPy.
- **Kept:** yes
- **Notes:** 1.09x–1.10x over the C on uniform codes from 4,096 up (0.38 ns per element against 0.41), 1.01x on four values (both wait for the counters). Slow where clearing counts: 1,000 counters at 16 elements 349 ns against 97 (one store per counter), 0.28x.

## 2. Clear with 64-byte steps; eight elements per step from two 8-byte loads

- **Idea:** the two things attempt 1 lost: clear the counters with `stp q0, q0` 64 bytes per step, the last step ending exactly at the end of `counts` (one store each below eight counters); and read eight elements per step with one `ldp` of two words, each element taken out with `sxth`/`sbfx`/`asr`, then compared and counted with its own branch.
- **check:** pass
- **compare:** significantly faster, 1.588x [1.585 – 1.591]
- **Kept:** yes
- **Notes:** uniform codes 1.35x from 4,096 up (0.28 ns per element), four values 1.41x (0.96 ns per element: the wait for the counters is less, but still more than three times the uniform cost), 1,000 counters at 16 elements 3.76x (91 ns, level with the C's `memset`).

## 3. Four tables of 32-bit counters on the stack, from 8 elements per counter

- **Idea:** as `bincount_u8`: element k of each group of four is counted in table k, so neighbouring equal elements do not wait for each other's store. A table is `bins` 32-bit counters rounded up to 16 bytes, all four below the stack pointer (at most 16 KB, for 4 to 1,024 counters); `counts` is not cleared on this path, the sum of the four tables, widened, is stored over it four counters per step with the last step overlapping. Taken when n >= 8 * bins and n < 2^32, so no 32-bit counter can overflow.
- **check:** pass
- **compare:** significantly faster, 1.381x [1.381 – 1.382]
- **Kept:** yes
- **Notes:** four values 2.43x–2.65x from 1,024 up with 64 counters (0.35 ns per element against 0.94), 2.81x–2.94x from 65,536 up with 1,000 counters (0.32 ns). Uniform codes 1.03x–1.04x from 4,096 up, but 0.95x at 1,024 with 64 counters, where clearing and summing the tables is not repaid.

## 4. A table of pairs for at most 64 counters, from 8,192 elements

- **Idea:** from the x86-64 kernel: with codes of six bits, count two elements with one increment, in a table of 64 x 64 32-bit counters (16 KB) below the stack pointer. One `tst` of each 8-byte word against `0xffc0` in every element says whether all four are in 0..63; `orr x, w, w, lsr #10` puts both pair indexes in the register, `and`/`ubfx` take them out. A word with an element outside 0..63 goes one element at a time into `counts` (cleared first on this path), each with its own test against `bins`. At the end one pass over the table gives the 64 column totals in sixteen vector accumulators and each row total by `addv`; `counts[v]` gets both for v < bins, so a pair code at or above `bins` is dropped there. n < 2^32 keeps the counters from overflowing.
- **check:** pass
- **compare:** significantly faster, 1.159x [1.158 – 1.160]
- **Kept:** yes
- **Notes:** uniform codes, 64 counters: 1.66x at 16,384, 1.93x at 1,048,576 (0.139 ns per element against 0.27). Four values: 1.49x–1.64x (0.216 ns per element against 0.36): with four values there are sixteen pairs, and consecutive increments of the same pair still wait for each other. 1,000 counters unchanged.

## 5. Two pair tables, the second pair of each word in the second

- **Idea:** with four values, consecutive pairs hit the same counter one time in sixteen; give the two pairs of each word tables of their own (32 KB), the second added into the first before the totals are taken.
- **check:** pass
- **compare:** significantly faster, 1.011x [1.010 – 1.011]
- **Kept:** no: it wins one table by losing another
- **Notes:** four values 1.16x from 1,048,576 up (0.195 ns per element against 0.225), but uniform codes 0.87x at 16,384 and 0.96x at 65,536: clearing 32 KB and adding 16 KB more costs about 0.4 us a call, and uniform codes gain nothing from the second table.

## 6. Four-table path: one vector test per sixteen elements

- **Idea:** the four-table loop spent a compare and a branch on every element. Load sixteen elements also as two vectors, `cmhs` against `bins` as unsigned 16-bit numbers (a negative element is at least 32,768, `bins` at most 1,024 here), `orr`, `umaxv`, one branch; when all sixteen are inside, each is taken out of its word with `and`/`ubfx`/`lsr` and counted with no test. Sixteen with any element outside go one at a time with the full test, out of line.
- **check:** pass
- **compare:** significantly faster, 1.030x [1.021 – 1.039]
- **Kept:** yes
- **Notes:** uniform codes with 1,000 counters 1.10x–1.13x from 16,384 up (0.246 ns per element against 0.276); 64 counters 1.14x–1.15x at 1,024 and 4,096 (below the pair path). Four values unchanged within noise: there the wait for the counters, not the instructions, sets the speed.

## 7. Eight tables instead of four on the table path

- **Idea:** with four values, element k and element k + 4 share a table and often a counter; eight tables put the same counter eight elements apart. Up to 32 KB of stack for 1,024 counters. Eight table bases need every free integer register, so `bins` lives in `v31` during the loop and the stack frame size in `d30`.
- **check:** pass
- **compare:** significantly faster, 1.014x [1.013 – 1.015]
- **Kept:** no: it wins one table by losing another
- **Notes:** four values with 1,000 counters 1.11x–1.17x from 16,384 up (0.27 ns per element against 0.31), but uniform codes with 1,000 counters 0.95x–0.98x and 64 counters 0.95x at 1,024 (clearing and summing twice as much).

## 8. Eight tables only where a sample of the data repeats

- **Idea:** attempt 7 only when at least 32 of the first 256 elements equal the element four places on (the one that shares their table of four), from 8,192 elements: uniform codes over 64 values have about 4 such in 256, four values about 64. Taken from the x86-64 kernel's sample. First written as one loop for both, with tables 4 to 7 the same memory as 0 to 3 on data that does not repeat: uniform codes lost 3% with 1,000 counters and 4% at 1,024 elements with 64 (0.97x and 0.96x), from the loop's changed registers alone. Kept as a separate loop, the four-table loop exactly as before.
- **check:** pass
- **compare:** significantly faster, 1.019x [1.016 – 1.022]
- **Kept:** yes
- **Notes:** four values with 1,000 counters 1.11x–1.17x from 16,384 up (0.27 ns per element at 1,048,576 against 0.31); every other table 1.00x.
