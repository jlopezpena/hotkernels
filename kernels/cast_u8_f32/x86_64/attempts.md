# cast_u8_f32 attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core), Linux, locked, pinned to one performance core by the judge.

**What the design expected: nothing to install.** NumPy has an AVX2 loop for this conversion and the compiler vectorises `naive.c` (`clang -O3 -march=native`); before any assembly, the naive C was level with NumPy from 16,384 elements to 4,194,304 (0.98x to 1.13x) and 1.5x ahead at 16,777,216.

## 1. `vpmovzxbd` and `vcvtdq2ps`, aligned stores, stores past the caches from 24 MB

- **Idea:** the design's two instructions, eight bytes per vector straight from memory, and around them what paid on `clip` and `cumsum`: stores aligned to 32 bytes (an element can be converted twice without harm, so alignment is one overlapping vector and the leftover is a block that ends with the last element), and `vmovntps` from 24 MB of output.
- **check:** pass
- **compare:** not run (first version). `run`: 1.23x over naive C and 6.10x over NumPy overall.
- **Kept:** yes
- **Notes:** the gain is at the two ends. 2.1x over the naive C at 16 elements and 1.2x to 1.3x up to 256, where the C has more to set up; 1.2x at 65,536 and 262,144; 2.14x at 16,777,216, which is the only size with an output of 24 MB or more. Level with it from 1,024 to 16,384 and at 1,048,576.

## Where it stands

Final `judge run`:

`random`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 1.8 ns | 3.6 ns | 2.05x [2.03 – 2.09] | 213.8 ns | 121x [120 – 123] |
| 64 | 2.1 ns | 2.8 ns | 1.31x [1.31 – 1.31] | 220.2 ns | 104x [104 – 104] |
| 256 | 6.4 ns | 7.8 ns | 1.23x [1.23 – 1.23] | 235.6 ns | 37.0x [37.0 – 37.1] |
| 1,024 | 24.8 ns | 27.0 ns | 1.09x [1.09 – 1.09] | 351.1 ns | 14.1x [14.1 – 14.3] |
| 4,096 | 97.7 ns | 96.7 ns | 0.99x [0.99 – 0.99] | 509.9 ns | 5.22x [5.20 – 5.23] |
| 16,384 | 849.6 ns | 852.1 ns | 1.00x [1.00 – 1.00] | 1.3 us | 1.47x [1.47 – 1.47] |
| 65,536 | 3.4 us | 4.2 us | 1.22x [1.22 – 1.23] | 4.1 us | 1.20x [1.20 – 1.20] |
| 262,144 | 14.5 us | 17.2 us | 1.19x [1.19 – 1.19] | 16.5 us | 1.14x [1.14 – 1.14] |
| 1,048,576 | 88.5 us | 89.0 us | 1.01x [1.01 – 1.01] | 93.1 us | 1.05x [1.05 – 1.05] |
| 4,194,304 | 490.4 us | 418.6 us | 0.85x [0.84 – 0.86] | 884.0 us | 1.80x [1.78 – 1.83] |
| 16,777,216 | 1.70 ms | 3.63 ms | 2.14x [2.12 – 2.15] | 5.69 ms | 3.35x [3.32 – 3.39] |
| overall |  |  | 1.23x [1.23 – 1.23] |  | 6.10x [6.09 – 6.12] |

With the output buffer 16 bytes behind its input ("apart" is the same kernel with the output clear of the input):

`random`:

| n | kernel | apart | speedup [95%] | naive C | speedup [95%] |
|---|---|---|---|---|---|
| 4,096 | 96.9 ns | 96.9 ns | 1.00x [1.00 – 1.00] | 161.8 ns | 1.67x [1.67 – 1.67] |
| 16,384 | 855.5 ns | 862.6 ns | 1.01x [1.01 – 1.01] | 850.5 ns | 0.99x [0.99 – 0.99] |
| 65,536 | 3.4 us | 3.4 us | 1.01x [1.01 – 1.01] | 4.1 us | 1.21x [1.21 – 1.22] |
| 262,144 | 13.7 us | 14.6 us | 1.06x [1.06 – 1.07] | 13.6 us | 0.99x [0.99 – 0.99] |
| 1,048,576 | 89.1 us | 89.0 us | 1.00x [1.00 – 1.00] | 89.0 us | 1.00x [1.00 – 1.00] |
| 4,194,304 | 619.5 us | 811.2 us | 1.31x [1.30 – 1.32] | 511.9 us | 0.83x [0.82 – 0.83] |
| 16,777,216 | 1.50 ms | 1.64 ms | 1.09x [1.07 – 1.11] | 3.55 ms | 2.36x [2.32 – 2.39] |
| overall |  |  | 1.06x [1.06 – 1.07] |  | 1.21x [1.21 – 1.22] |

The candidates the design lists for this kernel (`docs/x86-64-design.md`):

| Candidate | Outcome |
|---|---|
| `vpmovzxbd` from memory and `vcvtdq2ps` | Kept |
| NumPy ties the C here; expected: nothing to install on x86-64 | The C does tie NumPy. The assembly is 1.23x over the C overall, and against NumPy 1.2x at 65,536, 1.05x at 1,048,576 and 3.35x at 16,777,216 |
| Non-temporal stores above the third-level cache | Kept: all of the gain at 16,777,216 |
| An output just behind the input | No cost in the judge's close table |
| Instructions slow on other x86-64 processors | None used |

**Verdict for a person to confirm.** The design expected this kernel to be dropped on x86-64, and between 1,024 and 1,048,576 elements it is level with NumPy to within a fifth. What it adds is short arrays, which NumPy's call overhead decides anyway once there is a binding, and arrays whose output is 24 MB or more, where it is 3.35x over NumPy. Installing it is defensible; so is not installing it. On ARM64 it ships as C.

Left untried: a prefetch; loads a block ahead of the stores.
