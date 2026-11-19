# bincount_i16 attempts (arm64)

All runs on an Apple M1 Pro (macOS, native), locked, while other kernel runs were timing on the same machine, so timings from about 100,000 elements up are noisier than usual.

What the naive C costs here (Apple clang, `-O3`): on codes uniform in [0, 64) 0.41 ns per element from 4,096 up; on four distinct values 1.35 ns per element, about 4.3 cycles: an increment waits for the store of the previous increment of the same counter, and with four values that is most elements. With 1,000 counters the call costs 97 ns at 16 elements, nearly all of it clearing 8 KB of counters.

## 1. Baseline: one element per step

- **Idea:** the reference by hand. Clear the counters one 8-byte store at a time; per element a sign-extending load, an unsigned compare with `bins` (a negative element is a huge one), a branch, and a load, add and store of its counter.
- **check:** pass
- **compare:** n/a (first version). `run`: 0.87x [0.87 – 0.87] of the naive C overall, 3.01x over NumPy.
- **Kept:** yes
- **Notes:** 1.09x–1.10x over the C on uniform codes from 4,096 up (0.38 ns per element against 0.41), 1.01x on four values (both wait for the counters). Slow where clearing counts: 1,000 counters at 16 elements 349 ns against 97 (one store per counter), 0.28x.
