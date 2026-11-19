# argsort_i64 attempts (arm64)

There is no `kernel.s` beside this log: on ARM64 the kernel ships as its naive C.

**Verdict: ships as its naive C.** It is `argsort_f64`'s sort with another key, and that kernel's log, [`kernels/argsort_f64/arm64/attempts.md`](../../argsort_f64/arm64/attempts.md), holds the attempts that shaped it and the assembly tried against its hot loop.

## 1. The float64 kernel's sort with the key of int64

- **Idea:** everything after the key is the same for every 64-bit element type: the window of bits that differ, the packed words, the counts taken once, the last pass that writes indices. The key of an int64 is the element with its sign bit inverted.
- **check:** pass
- **compare:** n/a: first version
- **Kept:** yes
- **Notes:** integers show both ends of the kernel. Values in a small range have a window of a few bits and are sorted in one or two passes. Values that differ in all 64 bits take the pair path from about 8,000 elements, where an index and 64 key bits no longer fit one word.

## Result

`uv run judge run argsort_i64 --unlocked`, on an Apple M1 Pro. Unlocked because the kernel is not yet on `main`. NumPy is timed from Python.

RESULT_TABLE
