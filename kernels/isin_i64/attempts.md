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
