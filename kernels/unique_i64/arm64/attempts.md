# unique_i64 attempts (arm64)

All runs on an Apple M1 Pro (macOS, native), with `--unlocked`: the spec and naive C were new on this branch and not yet on `main`, and the branch's `pyproject.toml` and `uv.lock` carry the applications study's dependency group. Other agents were timing on the same machine, at a load average of about 2.5 to 3.5, so figures from about 100,000 elements up are noisier than usual.

What the naive C costs here (Apple clang, `-O3`): about 0.24 ns per element from 1,024 elements up, on two values and on ten, with ten or 256 values of range: one store per element, unrolled eight times with a test and branch each. With a range of 256 the table of 2 KB is cleared by `bzero` and read back, about 110 ns at 16 elements.

## 1. A 64-bit set for at most 64 values, a byte table inside `out` above that

- **Idea:** for `span <= 64`, value `lo + d` is bit `d` of a set: NEON makes `1 << d` with `ushl` for two elements per vector, or'ed into four sets, and ors `d` itself into four more; over a chunk of 64 elements the or of the `d` is below 64 exactly when every `d` is, and a chunk where it is not is read again one element at a time with the exact test (`ushl` reads only the low byte of `d`). Bits from `span` to 63 are values outside the range and are masked off at the end; the set is read out with `rbit` and `clz`. Fewer than 64 elements go one at a time. For `span > 64`, the last `span` bytes of `out` are a table of one byte per value, marked with `strb` four elements per step, and read out eight bytes at a time, skipping words of zeros; the values written to the front never reach a table byte not yet read (proof in the header of `kernel.s`). A first version used `out` itself as a table of int64 flags, with a branch-free compaction: 0.62x to 0.91x of the naive C with a range of 256, since it cleared and read eight times the bytes.
- **check:** pass
- **compare:** n/a (first version). `run`: 1.31x [1.31 – 1.31] over the naive C overall, 37.8x over NumPy.
- **Kept:** yes
- **Notes:** the set: 1.27x to 1.30x from 64 elements up (0.19 ns per element against 0.24 at a million), 1.20x at 16 elements of two values and 0.88x at 16 of ten. The byte table with a range of 256: 3.7x to 3.9x at 16 elements (30 ns against 112), 1.08x at 4,096, 0.98x to 0.99x from a million up, where both are one store per element.
