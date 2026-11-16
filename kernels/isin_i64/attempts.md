# isin_i64 attempts

All timings were taken with `--unlocked` (the spec was not on `main` yet) while
other agents were timing other kernels on the same machine, so intervals are
wider than usual.

## 1. Scalar baseline

- **Idea:** simplest correct version: the 8 test values in registers, one element at a time, `cmp` followed by seven `ccmp`, `cset`, `strb`.
- **check:** pass
- **compare:** n/a (first version). `run`: 0.54x [0.54 – 0.54] vs naive C, the same at every size.
- **Kept:** yes
- **Notes:** 1.2 ns per element. Naive C runs at 0.64 ns per element: clang vectorises it, 16 elements per iteration, with 64 `cmeq`, 56 `uzp1`, 7 `orr` and an `and` (128 vector instructions per 16 elements).

## 2. NEON, 16 elements per iteration, compare and OR at 64 bits

- **Idea:** broadcast the 8 test values (`ld1r`); per vector of two elements, 8 `cmeq` and a 7-`orr` tree; narrow the eight lane-mask vectors to 16 bytes with `uzp1` (4 + 2 + 1), `ushr #7` to get 0/1, one 16-byte store. Leftover by an overlapping final block; n < 16 stays scalar.
- **check:** pass
- **compare:** significantly faster, 1.84x [1.84 – 1.85]
- **Kept:** yes
- **Notes:** 0.644 ns per element, which is exactly what naive C does (0.64): 128 vector instructions per 16 elements at four per cycle. Doing the same work as the compiler gives the same speed; a win needs fewer instructions per element.

## 3. Narrow the elements to 16-bit keys when the test values are close together

- **Idea:** if all 8 test values lie within 32,767 of `test[0]`, subtract `base = test[0] - 0x7FFF` from every element and narrow 64 to 32 to 16 bits with unsigned saturation (`uqxtn`, `uqxtn2`). Values below 0xFFFF stay exact, everything else becomes 0xFFFF, and no test key is 0xFFFF, so membership is exactly "16-bit key equals a test key". The 8 `cmeq` + 7 `orr` then run on vectors of eight keys instead of two elements: 52 vector instructions per 16 elements (8 `sub`, 12 narrowing, 30 compare/OR, 2 to finish) instead of 128. Any other test values take the 64-bit loop of attempt 2, chosen once per call by narrowing the test values the same way and looking for a 0xFFFF.
- **check:** pass (plus 2,000,000 random cases in a private fuzzer: test values at spreads of exactly 32,767 and 32,768, near the int64 wrap, with duplicates, and elements just outside the key range: no failure)
- **compare:** significantly faster, 2.30x [2.29 – 2.30] (NOISY MACHINE warning on this run; every size from 256 up was 2.37x–2.43x, n = 16 1.60x, n = 64 2.14x)
- **Kept:** yes
- **Notes:** 0.265 ns per element. The speedup exists only for test values that fit a 16-bit window; unrelated 64-bit test values run at the speed of attempt 2, which is the speed of naive C. The benchmark's only distribution draws test values from [0, 1000), so it always takes the narrow path.

## 4. Skip the subtraction when the test values are already below 0xFFFF

- **Idea:** try base 0 first: if every test value, read as unsigned, is below 0xFFFF, the elements can be narrowed as they are and the 8 `sub` per 16 elements disappear (44 vector instructions instead of 52). The biased loop of attempt 3 stays for other close-together sets (negative values, large values), the 64-bit loop for the rest.
- **check:** pass (private fuzzer extended with test values in [0, 66000) and around 65,534: 2,000,000 cases, no failure)
- **compare:** significantly faster, 1.17x [1.17 – 1.17]; 1.32x at 16, 1.14x–1.16x from 256 up
- **Kept:** yes
- **Notes:** 0.229 ns per element, 11.8 cycles per 16 elements: still four vector instructions per cycle, so the count is what matters.

## 5. Nibble tables instead of 8 compares, for n >= 96

- **Idea:** match the 16-bit keys with table lookups. Split the 16 keys into low bytes and high bytes (`uzp1`, `uzp2`), each byte into two nibbles (`and`, `ushr`), and look each nibble up with `tbl` in a 16-entry table for its position; the table byte has one bit per test key, set when that test key has the same nibble there. AND the four lookups: bit j survives exactly when the key equals test key j, so `umin` with 1 gives the answer. 14 instructions for 16 keys instead of 32, so 26 per 16 elements with the narrowing (34 with a base). The tables are built without memory: for each nibble position and each test key, `dup` its nibble, `cmeq` against 0..15, and `sli` the result in at bit j (98 instructions in all), so the tables are only used from n = 96; below that the compare loops of attempt 4 run.
- **check:** pass (private fuzzer, 2,000,000 cases with the threshold lowered to 16 so the table path sees every size: no failure)
- **compare:** significantly faster, 1.50x [1.49 – 1.50]: 1.37x at 256, 1.60x–1.73x from 1,024 up, unchanged at 16 and 64
- **Kept:** yes
- **Notes:** 0.131 ns per element in the table loop, 6.7 cycles per 16 elements. Building the tables takes about 8.7 ns; measured with the threshold at 16, n = 16 took 12.1 ns against 5.4 and n = 64 17.6 against 16.0, and the break-even is near 90 elements.
