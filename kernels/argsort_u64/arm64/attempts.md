# argsort_u64 attempts (arm64)

There is no `kernel.s` beside this log: on ARM64 the kernel ships as its naive C.

**Verdict: ships as its naive C.** It is `argsort_f64`'s sort with another key, and that kernel's log, [`kernels/argsort_f64/arm64/attempts.md`](../../argsort_f64/arm64/attempts.md), holds the attempts that shaped it and the assembly tried against its hot loop.

## 1. The float64 kernel's sort with the key of uint64

- **Idea:** everything after the key is the same for every 64-bit element type: the window of bits that differ, the packed words, the counts taken once, the last pass that writes indices. The key of a uint64 is the element.
- **check:** pass
- **compare:** n/a: first version
- **Kept:** yes
- **Notes:** integers show both ends of the kernel. Values in a small range have a window of a few bits and are sorted in one or two passes. Values that differ in all 64 bits take the pair path from about 8,000 elements, where an index and 64 key bits no longer fit one word.

## Result

`uv run judge run argsort_u64 --unlocked`, on an Apple M1 Pro. Unlocked because the kernel is not yet on `main`. NumPy is timed from Python.

| Elements | full_range: C | NumPy | faster by | small_range: C | NumPy | faster by |
|---|---|---|---|---|---|---|
| 16 | 47.3 ns | 1.0 us | 21.2x | 46.4 ns | 978.1 ns | 21.1x |
| 64 | 1.4 us | 1.5 us | 1.10x | 182.3 ns | 1.5 us | 8.20x |
| 256 | 2.5 us | 5.6 us | 2.22x | 576.5 ns | 5.6 us | 9.74x |
| 1,024 | 7.4 us | 27.6 us | 3.72x | 2.1 us | 27.3 us | 13.0x |
| 4,096 | 24.0 us | 158.2 us | 6.58x | 4.0 us | 153.6 us | 37.9x |
| 16,384 | 108.6 us | 838.5 us | 7.72x | 18.2 us | 822.5 us | 45.2x |
| 65,536 | 417.6 us | 4.44 ms | 10.6x | 71.0 us | 4.38 ms | 61.8x |
| 262,144 | 1.90 ms | 22.97 ms | 12.1x | 344.6 us | 21.99 ms | 63.8x |
| 1,048,576 | 7.90 ms | 116.03 ms | 14.7x | 1.66 ms | 129.03 ms | 78.0x |
| 4,194,304 | 38.60 ms | 1109.23 ms | 28.7x | 7.10 ms | 1139.68 ms | 160x |
| 16,777,216 | 177.09 ms | 7571.14 ms | 42.8x | 28.58 ms | 7703.10 ms | 270x |
| overall | | | 8.74x [8.67 – 8.82] | | | 40.6x [40.20 – 41.82] |

Over every distribution: 18.8x [18.71 – 19.14]. Load average 12.76 when the run started, with other work running, so the figures are indicative.
