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

| Elements | full_range: C | NumPy | faster by | small_range: C | NumPy | faster by |
|---|---|---|---|---|---|---|
| 16 | 48.6 ns | 1.0 us | 21.0x | 48.5 ns | 1.0 us | 20.7x |
| 64 | 1.4 us | 1.6 us | 1.12x | 195.2 ns | 1.6 us | 8.02x |
| 256 | 2.6 us | 5.7 us | 2.19x | 608.7 ns | 5.6 us | 9.27x |
| 1,024 | 7.5 us | 27.6 us | 3.69x | 2.2 us | 27.2 us | 12.2x |
| 4,096 | 24.7 us | 158.1 us | 6.39x | 5.2 us | 150.3 us | 28.8x |
| 16,384 | 108.0 us | 832.9 us | 7.71x | 21.1 us | 812.1 us | 38.6x |
| 65,536 | 433.2 us | 4.48 ms | 10.3x | 83.1 us | 4.34 ms | 52.3x |
| 262,144 | 1.95 ms | 22.78 ms | 11.7x | 380.7 us | 23.44 ms | 61.6x |
| 1,048,576 | 8.55 ms | 147.59 ms | 17.3x | 1.84 ms | 119.67 ms | 64.9x |
| 4,194,304 | 39.69 ms | 1209.89 ms | 30.5x | 7.76 ms | 1005.40 ms | 130x |
| 16,777,216 | 181.45 ms | 7929.03 ms | 43.7x | 31.14 ms | 7531.18 ms | 242x |
| overall | | | 8.85x [8.77 – 8.95] | | | 36.1x [35.84 – 36.92] |

Over every distribution: 17.9x [17.79 – 18.11]. Load average 12.26 when the run started, with other work running, so the figures are indicative.
