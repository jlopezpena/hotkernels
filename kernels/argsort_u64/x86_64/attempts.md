# argsort_u64 attempts (x86_64)

There is no `kernel.s` beside this log: on x86-64 the kernel ships as its naive C, the same C as on ARM64.

**Verdict: ships as its naive C, untimed.** The kernel was written on an Apple M1 Pro and no x86-64 machine was at hand. Nothing here is a measurement on x86-64.

## 1. The C as it ships on ARM64

- **Idea:** the gain over NumPy's stable sort is the algorithm, a radix sort where NumPy merges, and it does not depend on the processor. The C has no instruction of either architecture in it.
- **check:** not run on x86-64. On ARM64 `uv run judge check argsort_u64 --unlocked` passes
- **compare:** n/a: nothing to compare with, and no x86-64 machine to time it on
- **Kept:** yes
- **Notes:** three choices in the C were made by timing on the Apple M1 Pro and may sit elsewhere on x86-64: the digit of 11 bits (2,048 places a pass scatters to), the 13 bits a first pass may take, and the 2,048 elements below which digits are 8 bits. The first x86-64 run should time those before trying assembly; the ARM64 log, [`kernels/argsort_f64/arm64/attempts.md`](../../argsort_f64/arm64/attempts.md), says what each was weighed against.
