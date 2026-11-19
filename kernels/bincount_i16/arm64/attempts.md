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
