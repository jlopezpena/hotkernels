# count_nonzero_i64 attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core), Linux, locked, pinned to one performance core by the judge. The two `count_nonzero` kernels are generated from one source and were written together.

What the compiler does with `naive.c` (`clang -O3 -march=native`): it vectorises the count: 0.055 ns per element at 1,024 and 0.057 at 65,536. NumPy takes 0.11 ns per element at 1,048,576.

## 1. Compare with zero and subtract into four accumulators, every load aligned

- **Idea:** the design's first candidate. The zeros are counted and subtracted from n. `vpcmpeqq` against zeros gives all ones, which is -1, in the lanes that are zero; subtracting that from an accumulator of 64-bit lanes adds one per zero. Two instructions per four elements, the compare with its load folded in. Sixteen elements per step into four accumulators, then four per step, then the last 0 to 3 one at a time. The loads are brought to a 32-byte boundary first, as in `sum_u8`, by counting up to three elements one at a time.
- **check:** pass
- **compare:** not run (first version). `run`: 1.17x over naive C overall.
- **Kept:** yes, as the starting point
- **Notes:** 0.59x at 16 elements: the head and its loop cost more than a short array's straddling loads.

## 2. No alignment below 64 elements; vectors from 4 elements

- **Idea:** arrays of fewer than 64 elements skip the head, and the vector steps start at 4 elements, not 16.
- **check:** pass
- **compare:** not run: both kernels were changed in the generator and judged by `run` against the naive C again (table below).
- **Kept:** yes
- **Notes:** the 16-element call went from 3.3 ns to 2.3 ns.

## Where it stands

Final `judge run`:

`random`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 2.3 ns | 2.1 ns | 0.92x [0.92 – 0.92] | 133.5 ns | 58.9x [58.9 – 58.9] |
| 64 | 4.0 ns | 4.5 ns | 1.12x [1.12 – 1.12] | 138.3 ns | 34.6x [34.6 – 34.7] |
| 256 | 11.2 ns | 14.8 ns | 1.33x [1.33 – 1.33] | 165.7 ns | 14.9x [14.8 – 14.9] |
| 1,024 | 41.7 ns | 54.5 ns | 1.31x [1.31 – 1.31] | 275.8 ns | 6.61x [6.61 – 6.62] |
| 4,096 | 161.4 ns | 205.0 ns | 1.27x [1.27 – 1.27] | 593.4 ns | 3.68x [3.67 – 3.68] |
| 16,384 | 853.1 ns | 1.4 us | 1.65x [1.65 – 1.65] | 1.8 us | 2.12x [2.12 – 2.12] |
| 65,536 | 3.4 us | 3.7 us | 1.07x [1.07 – 1.07] | 6.5 us | 1.89x [1.89 – 1.89] |
| 262,144 | 17.7 us | 25.1 us | 1.42x [1.42 – 1.42] | 27.1 us | 1.53x [1.53 – 1.53] |
| 1,048,576 | 93.7 us | 111.1 us | 1.19x [1.19 – 1.19] | 117.7 us | 1.26x [1.26 – 1.26] |
| 4,194,304 | 781.6 us | 897.4 us | 1.15x [1.15 – 1.15] | 1.01 ms | 1.29x [1.29 – 1.29] |
| 16,777,216 | 4.03 ms | 4.40 ms | 1.09x [1.09 – 1.10] | 4.82 ms | 1.20x [1.19 – 1.20] |
| overall |  |  | 1.21x [1.21 – 1.21] |  | 4.28x [4.28 – 4.28] |

What holds it: memory, at nearly every size. Two 32-byte loads per cycle is eight elements per cycle, 0.024 ns per element, and the judge's timing leaves the first-level cache from about a thousand elements. The compiler's loop is close to the same limit, which is why the gain is what it is.

The candidates the design lists for this family (`docs/x86-64-design.md`):

| Candidate | Outcome |
|---|---|
| A compare against zero and a subtract into an accumulator | Kept |
| `vmovmskpd` and `popcnt` | Not tried: three instructions per four elements and a trip to the general registers, against two |
| For floats the sign is shifted out first | Does not apply to integers |
| Level with NumPy above the second-level cache whatever is done | Close to it: see the table from 1,048,576 up |
| Alignment of data | Kept from 64 elements (attempts 1 and 2) |
| A prefetch, several streams | Not tried. Worth 2% and 6% on `sum_u8`, at sizes where this kernel is already at memory's speed |
| Masked loads for leftover elements | Not tried: open issue 3 of the design |
| Instructions slow on other x86-64 processors | None used |

**Verdict:** assembly, 1.2x over the naive C overall, as the design expected (1.1x to 1.3x over NumPy at 1,048,576; it is 1.26x).
