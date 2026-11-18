# bincount_u8 attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core), Linux, locked, pinned to one performance core by the judge.

What the compiler does with `naive.c` (`clang -O3 -march=native`): a call to `memset` and a loop unrolled eight times of a byte load and an `inc` on memory, 0.185 ns per byte, one cycle, up to 16,384 bytes. That is the limit of the scheme: one counter is loaded, incremented and stored per byte, and this processor makes one store per cycle unless two go to the same cache line. NumPy takes 0.8 ns per byte at 1,048,576.

**The naive C's time is not steady here.** Between runs of the judge it varied by up to 2x at single sizes (4,096 bytes: 759 ns in two runs, 1.7 us in a third; 64 bytes: 20 to 29 ns). Its counters are the output buffer and its loads follow its stores; which sizes are hit differs from process to process. The kernel's own times did not move.

## 1. Eight bytes per load, four tables from 32,768 bytes

- **Idea:** the design's candidate, the ARM64 scheme with an increment that addresses memory directly. One 8-byte load feeds eight `inc`, the byte taken with `movzx` from the low two bytes of the register and a shift by 16 between pairs. From 32,768 bytes the counts go into four tables of 256 counters below the stack pointer, byte i into table i mod 4, added up at the end, so that equal neighbouring bytes update different counters. Below that the counts go straight into the output, cleared with `rep stosb`.
- **check:** pass
- **compare:** not run (first version). Against the naive C in `run`: 1.07x overall in the final run (table below), 1.08x and 1.2x in two earlier ones, the difference being the naive C's (see above).
- **Kept:** yes
- **Notes:** three things were settled on the way to this version, by `run` and not by `compare`, there being no committed version yet. **Tables from 2,048 bytes:** 0.91x to 0.96x from 2,048 to 16,384, where clearing and adding four tables costs more than they save; they pay from 65,536 (1.16x to 1.27x). **Clearing 2 KB with a loop of 32-byte stores:** 22.8 ns for a call with 16 bytes against the naive C's 12; the first increments read counters whose zeros are still in the store buffer as parts of wider stores. `rep stosq` 15.4 ns, `rep stosb` 16.8 ns; `memset` in the C library does the same job in about 10. **A byte loop below the threshold** was 5% to 13% slower than the C's unrolled loop; the eight-bytes-per-load loop is level with it.

## Where it stands

Final `judge run`:

`random`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 16.8 ns | 11.4 ns | 0.68x [0.67 – 0.68] | 474.6 ns | 28.3x [28.2 – 28.3] |
| 64 | 25.3 ns | 21.3 ns | 0.84x [0.83 – 0.84] | 515.0 ns | 20.3x [20.3 – 20.4] |
| 256 | 63.4 ns | 110.7 ns | 1.75x [1.74 – 1.75] | 682.9 ns | 10.8x [10.8 – 10.8] |
| 1,024 | 200.0 ns | 198.6 ns | 0.99x [0.99 – 0.99] | 1.3 us | 6.28x [6.27 – 6.28] |
| 4,096 | 758.5 ns | 760.1 ns | 1.00x [1.00 – 1.00] | 3.8 us | 4.97x [4.97 – 4.98] |
| 16,384 | 3.0 us | 3.0 us | 1.00x [1.00 – 1.00] | 13.2 us | 4.37x [4.37 – 4.37] |
| 65,536 | 12.8 us | 15.3 us | 1.20x [1.20 – 1.20] | 50.8 us | 3.97x [3.97 – 3.98] |
| 262,144 | 50.6 us | 60.4 us | 1.19x [1.19 – 1.19] | 208.2 us | 4.11x [4.11 – 4.12] |
| 1,048,576 | 202.9 us | 229.5 us | 1.13x [1.13 – 1.13] | 850.5 us | 4.19x [4.19 – 4.19] |
| 4,194,304 | 832.7 us | 931.0 us | 1.12x [1.12 – 1.12] | 6.32 ms | 7.59x [7.58 – 7.62] |
| 16,777,216 | 3.32 ms | 3.81 ms | 1.15x [1.15 – 1.15] | 26.17 ms | 7.89x [7.87 – 7.92] |
| overall |  |  | 1.07x [1.07 – 1.07] |  | 7.37x [7.36 – 7.37] |

With the output buffer 16 bytes behind its input ("apart" is the same kernel with the output clear of the input):

`random`:

| n | kernel | apart | speedup [95%] | naive C | speedup [95%] |
|---|---|---|---|---|---|
| 4,096 | 757.4 ns | 758.8 ns | 1.00x [1.00 – 1.00] | 758.9 ns | 1.00x [1.00 – 1.00] |
| 16,384 | 3.0 us | 3.0 us | 1.00x [1.00 – 1.00] | 3.0 us | 1.00x [1.00 – 1.00] |
| 65,536 | 12.8 us | 12.7 us | 0.99x [0.99 – 0.99] | 16.1 us | 1.26x [1.26 – 1.26] |
| 262,144 | 50.6 us | 50.7 us | 1.00x [1.00 – 1.00] | 63.1 us | 1.25x [1.25 – 1.25] |
| 1,048,576 | 202.9 us | 202.5 us | 1.00x [1.00 – 1.00] | 232.1 us | 1.14x [1.14 – 1.14] |
| 4,194,304 | 811.4 us | 811.1 us | 1.00x [1.00 – 1.00] | 923.7 us | 1.14x [1.14 – 1.14] |
| 16,777,216 | 3.32 ms | 3.33 ms | 1.00x [1.00 – 1.01] | 3.78 ms | 1.14x [1.14 – 1.14] |
| overall |  |  | 1.00x [1.00 – 1.00] n.s. |  | 1.13x [1.13 – 1.13] |

What holds it: one counter update per byte, at one store per cycle. From 1,024 to 16,384 bytes the kernel and the naive C are the same loop at the same speed. At 16 and 64 bytes the kernel loses to the C library's `memset` (0.68x and 0.84x). From 65,536 the naive C slows to 0.22 to 0.23 ns per byte and the four tables do not.

**What the judge does not time: bytes that repeat.** A micro-benchmark outside the judge, 1,048,576 bytes, ns per byte, naive C then kernel: uniform random 0.220 and 0.194; 16 distinct values 0.206 and 0.194; runs of 1 to 8 equal bytes 0.286 and 0.204; all bytes equal **1.390 and 0.351**. In one table every update of the same counter waits for the one before it to be stored; in four tables it is every fourth. Images and text are closer to the last rows than to the first.

The candidates the design lists for this kernel (`docs/x86-64-design.md`):

| Candidate | Outcome |
|---|---|
| The same scheme with an increment that addresses memory directly | Kept |
| `rep stosq` to clear the tables | Used for the output on the short path (as `rep stosb`, see above); the four tables are cleared with a loop of 32-byte stores, which costs nothing there because the first increments come long after |
| The number of tables is retested | Four from 32,768 bytes, one below. Two and eight were not tried |
| Alignment of data, non-temporal stores, masked loads | Not tried: an 8-byte load per step and a 2 KB output |
| An output just behind the input | No cost for the kernel in the judge's close table |
| Instructions slow on other x86-64 processors | None used |

**Verdict for a person to confirm:** assembly, by a small margin on the judge's data (1.07x to 1.2x overall, 1.12x to 1.2x from 65,536 bytes) and a large one on repeated bytes, which the judge does not time. Shipping the C here would be defensible on the judge's figures alone.

Left untried: 32-bit counters, which halve the tables; a pair table of 65,536 entries (slower on ARM64).
