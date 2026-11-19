# count_nonzero_f64 attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core), Linux, locked, pinned to one performance core by the judge. The two `count_nonzero` kernels are generated from one source and were written together.

What the compiler does with `naive.c` (`clang -O3 -march=native`): it vectorises the count: 0.039 ns per element at 1,024 and 0.066 at 65,536. NumPy takes 0.39 ns per element at 1,048,576.

## 1. Compare with zero and subtract into four accumulators, every load aligned

- **Idea:** the design's first candidate. The zeros are counted and subtracted from n. `vcmpeqpd` against zeros (+0.0 and -0.0 compare equal, a NaN does not) gives all ones, which is -1, in the lanes that are zero; subtracting that from an accumulator of 64-bit lanes adds one per zero. Two instructions per four elements, the compare with its load folded in. Sixteen elements per step into four accumulators, then four per step, then the last 0 to 3 one at a time. The loads are brought to a 32-byte boundary first, as in `sum_u8`, by counting up to three elements one at a time.
- **check:** pass
- **compare:** not run (first version). `run`: 0.99x over naive C overall.
- **Kept:** yes, as the starting point
- **Notes:** 0.50x at 16 elements and 0.78x at 64: the head and its loop cost more than a short array's straddling loads.

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
| 16 | 2.3 ns | 1.7 ns | 0.75x [0.75 – 0.75] | 133.7 ns | 59.0x [58.9 – 59.1] |
| 64 | 4.1 ns | 3.3 ns | 0.80x [0.80 – 0.81] | 159.0 ns | 39.1x [38.8 – 39.4] |
| 256 | 9.5 ns | 9.7 ns | 1.02x [1.02 – 1.02] | 247.2 ns | 25.9x [25.9 – 26.0] |
| 1,024 | 35.7 ns | 39.6 ns | 1.11x [1.11 – 1.11] | 550.1 ns | 15.4x [15.4 – 15.4] |
| 4,096 | 144.7 ns | 150.6 ns | 1.04x [1.04 – 1.04] | 1.8 us | 12.2x [12.2 – 12.2] |
| 16,384 | 650.6 ns | 646.1 ns | 0.99x [0.99 – 0.99] | 6.6 us | 10.2x [10.1 – 10.2] |
| 65,536 | 2.6 us | 4.3 us | 1.65x [1.65 – 1.65] | 25.8 us | 9.86x [9.86 – 9.87] |
| 262,144 | 15.6 us | 20.8 us | 1.33x [1.33 – 1.33] | 103.2 us | 6.61x [6.61 – 6.62] |
| 1,048,576 | 94.2 us | 105.0 us | 1.11x [1.11 – 1.12] | 412.6 us | 4.38x [4.38 – 4.39] |
| 4,194,304 | 797.9 us | 864.7 us | 1.08x [1.08 – 1.09] | 1.95 ms | 2.44x [2.43 – 2.45] |
| 16,777,216 | 4.14 ms | 4.40 ms | 1.06x [1.06 – 1.07] | 8.08 ms | 1.95x [1.95 – 1.96] |
| overall |  |  | 1.06x [1.06 – 1.06] |  | 10.4x [10.4 – 10.4] |

What holds it: memory, at nearly every size. Two 32-byte loads per cycle is eight elements per cycle, 0.024 ns per element, and the judge's timing leaves the first-level cache from about a thousand elements. The compiler's loop is close to the same limit, which is why the gain is what it is.

The candidates the design lists for this family (`docs/x86-64-design.md`):

| Candidate | Outcome |
|---|---|
| A compare against zero and a subtract into an accumulator | Kept |
| `vmovmskpd` and `popcnt` | Not tried: three instructions per four elements and a trip to the general registers, against two |
| For floats the sign is shifted out first | Not needed: `vcmpeqpd` treats the two zeros alike without it, in one instruction with its load folded in |
| Level with NumPy above the second-level cache whatever is done | Close to it: see the table from 1,048,576 up |
| Alignment of data | Kept from 64 elements (attempts 1 and 2) |
| A prefetch, several streams | Not tried. Worth 2% and 6% on `sum_u8`, at sizes where this kernel is already at memory's speed |
| Masked loads for leftover elements | Not tried: open issue 3 of the design |
| Instructions slow on other x86-64 processors | None used |

**Verdict for a person to confirm.** 1.06x over the naive C overall is below the 10% the goal template asks of an assembly kernel. It is never slower from 256 elements up and 1.2x to 1.65x faster at 65,536 and 262,144, where the aligned loads count; it loses at 16 and 64 elements. Removing `x86_64/kernel.s` makes this a C kernel on x86-64, and the figures would support that.
