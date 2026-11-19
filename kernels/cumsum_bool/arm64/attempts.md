# cumsum_bool attempts (arm64)

All timings on an Apple M1 Pro (performance core: 128 KB first-level cache, 12 MB second-level cache shared by the cluster), macOS, with the spec locked. Another kernel run was timing on the same machine throughout, so the figures are noisier than a quiet run, mostly from about 100,000 elements up.

What the compiler does with `naive.c` (Apple clang, `-O3`): the scalar chain, one `ldrb`, one `add` and one `str` per element, not unrolled: 0.37 ns per element at every size from 256 to 16,777,216, about 1.2 cycles. NumPy takes 2.5 ns per element at 4,096 and 2.3 from 65,536 up.

The output is eight bytes for every byte read. Sixteen elements are 128 bytes of output, eight 16-byte stores, and this core retires about two of those per cycle, so about 4.4 cycles per 16 elements (0.085 ns per element) is the floor for any version in the first-level cache.

## 1. Baseline: one element at a time

- **Idea:** `ldrb`, `add`, `str` with post-indexed pointers and a down-counter; the starting point, to get `check` passing.
- **check:** pass
- **compare:** n/a (first version). `run`: 1.00x [1.00 – 1.00] of naive C overall, not significant at any size; 11.9x over NumPy.
- **Kept:** yes (baseline)
- **Notes:** the same instructions the compiler emits for `naive.c`: 0.37 ns per element at every size, in both tables of `run`.
