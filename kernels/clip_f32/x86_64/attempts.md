# clip_f32 attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, locked, pinned to one performance core by the judge. `clip_f32` and `clip_f64` are generated from one source and were written together; this log has the reasoning.

**The opponent and the baseline.** NumPy has an AVX2 loop for `clip` here: 0.096 ns per `float32` at 65,536. The compiler vectorises `naive.c` (`clang -O3 -march=native`), but not into the two instructions the design expected: it emits `vminps`, a `vcmpltps` and a `vblendvps` per vector, and `vblendvps` runs once per cycle on this processor. 0.044 ns per element at 1,024, 0.069 at 65,536.

## 1. `vmaxps` then `vminps` with the bound first; aligned stores; loads a block ahead

- **Idea:** the design's candidate. `vmaxps a, b` computes `a > b ? a : b`: it returns its second operand when the two are unordered and when they are equal, zeros of either sign included. With the bound first and the data second, `t = vmaxps lo, v` is the reference's `v < lo ? lo : v` and `out = vminps hi, t` is its `t > hi ? hi : t`, bit for bit, for every pair of bounds with `lo <= hi`: a NaN element stays, and a zero element keeps its sign against a zero bound. **One path, whatever the bounds and the data.** Around those two instructions: the stores are aligned to 32 bytes (an element can be clamped twice without harm, so alignment costs one overlapping vector and the leftover is a block that ends with the last element); the loads run a block of four vectors ahead of the stores, so that a store just behind the input does not hold up the loads after it; and from 24 MB of output the blocks are stored with `vmovntps`.
- **check:** pass, first time, both kernels (53,489 cases each: the spec's twelve pairs of bounds, with zeros of both signs as bounds and as data). Outside the judge: 3,936 runs per kernel against the reference expression, with the same twelve pairs of bounds, data that is one tenth each of +0.0, -0.0, NaN, both infinities and the bounds themselves, 36 sizes up to 6,300,001 so that the cache-bypassing stores are reached, nine alignments of `out`, and with `out` = `x`. No mismatch.
- **compare:** not run (first version). `run`: 1.25x over naive C and 12.7x over NumPy overall.
- **Kept:** yes
- **Notes:** 0.025 ns per element at 1,024, eight elements per cycle, which is what two ports for `vmaxps` and `vminps` allow. The six tables of the judge (three kinds of data, two pairs of bounds) agree to within a few percent: 1.22x to 1.32x over the naive C. The ARM64 kernel has to time them separately because its speed depends on them.

## Where it stands

Final `judge run`:

`random`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 1.7 ns | 3.5 ns | 2.02x [2.00 – 2.06] | 1.1 us | 608x [602 – 614] |
| 64 | 3.0 ns | 3.3 ns | 1.10x [1.10 – 1.11] | 1.1 us | 369x [368 – 369] |
| 256 | 7.0 ns | 11.1 ns | 1.58x [1.58 – 1.59] | 1.1 us | 160x [160 – 161] |
| 1,024 | 25.5 ns | 45.3 ns | 1.78x [1.78 – 1.78] | 1.3 us | 50.5x [50.5 – 50.6] |
| 4,096 | 100.3 ns | 174.6 ns | 1.74x [1.74 – 1.74] | 1.6 us | 15.6x [15.6 – 15.6] |
| 16,384 | 948.9 ns | 1.1 us | 1.20x [1.20 – 1.20] | 2.5 us | 2.66x [2.66 – 2.67] |
| 65,536 | 3.8 us | 4.5 us | 1.20x [1.20 – 1.20] | 6.3 us | 1.69x [1.69 – 1.69] |
| 262,144 | 21.2 us | 21.0 us | 0.99x [0.99 – 1.00] | 25.5 us | 1.20x [1.20 – 1.20] |
| 1,048,576 | 120.6 us | 121.0 us | 1.00x [1.00 – 1.00] | 126.6 us | 1.05x [1.05 – 1.05] |
| 4,194,304 | 961.2 us | 981.8 us | 1.02x [1.02 – 1.04] | 1.10 ms | 1.14x [1.13 – 1.15] |
| 16,777,216 | 3.83 ms | 4.95 ms | 1.29x [1.28 – 1.30] | 7.66 ms | 2.00x [1.98 – 2.02] |
| overall |  |  | 1.32x [1.32 – 1.32] |  | 11.3x [11.2 – 11.3] |

`random, scalars=zero_lo`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 1.9 ns | 3.6 ns | 1.89x [1.84 – 1.90] | 1.1 us | 567x [566 – 567] |
| 64 | 2.8 ns | 3.3 ns | 1.17x [1.16 – 1.17] | 1.1 us | 388x [387 – 390] |
| 256 | 7.0 ns | 11.1 ns | 1.58x [1.58 – 1.59] | 1.1 us | 161x [160 – 161] |
| 1,024 | 25.5 ns | 45.2 ns | 1.77x [1.77 – 1.77] | 1.3 us | 50.5x [50.4 – 50.5] |
| 4,096 | 100.5 ns | 175.3 ns | 1.74x [1.74 – 1.75] | 1.6 us | 15.7x [15.7 – 15.7] |
| 16,384 | 948.3 ns | 1.1 us | 1.20x [1.20 – 1.20] | 2.5 us | 2.67x [2.66 – 2.67] |
| 65,536 | 3.8 us | 4.5 us | 1.20x [1.20 – 1.20] | 6.4 us | 1.69x [1.69 – 1.69] |
| 262,144 | 41.4 us | 28.5 us | 0.69x [0.69 – 0.69] | 320.3 us | 7.74x [7.73 – 7.75] |
| 1,048,576 | 185.5 us | 131.8 us | 0.71x [0.71 – 0.71] | 705.5 us | 3.80x [3.80 – 3.81] |
| 4,194,304 | 950.7 us | 973.4 us | 1.02x [1.02 – 1.03] | 1.08 ms | 1.14x [1.13 – 1.15] |
| 16,777,216 | 3.92 ms | 5.10 ms | 1.30x [1.29 – 1.31] | 7.51 ms | 1.92x [1.90 – 1.94] |
| overall |  |  | 1.23x [1.23 – 1.24] |  | 14.9x [14.9 – 14.9] |

`sparse_negative_zeros`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 1.8 ns | 3.5 ns | 2.01x [1.97 – 2.04] | 1.1 us | 605x [599 – 610] |
| 64 | 2.8 ns | 3.3 ns | 1.17x [1.17 – 1.17] | 1.1 us | 393x [392 – 394] |
| 256 | 7.0 ns | 11.1 ns | 1.58x [1.58 – 1.59] | 1.1 us | 160x [160 – 160] |
| 1,024 | 25.5 ns | 45.5 ns | 1.78x [1.78 – 1.79] | 1.3 us | 50.3x [50.3 – 50.4] |
| 4,096 | 100.1 ns | 174.5 ns | 1.74x [1.74 – 1.74] | 1.6 us | 15.7x [15.6 – 15.7] |
| 16,384 | 948.9 ns | 1.1 us | 1.21x [1.21 – 1.21] | 2.5 us | 2.66x [2.66 – 2.66] |
| 65,536 | 3.8 us | 4.5 us | 1.20x [1.20 – 1.20] | 6.3 us | 1.69x [1.68 – 1.69] |
| 262,144 | 41.3 us | 28.5 us | 0.69x [0.69 – 0.69] | 320.3 us | 7.75x [7.74 – 7.75] |
| 1,048,576 | 195.7 us | 129.8 us | 0.66x [0.66 – 0.66] | 1.04 ms | 5.29x [5.28 – 5.29] |
| 4,194,304 | 947.2 us | 962.6 us | 1.02x [1.01 – 1.02] | 1.05 ms | 1.11x [1.10 – 1.12] |
| 16,777,216 | 3.91 ms | 5.01 ms | 1.28x [1.28 – 1.29] | 7.51 ms | 1.92x [1.92 – 1.93] |
| overall |  |  | 1.23x [1.23 – 1.23] |  | 15.4x [15.4 – 15.4] |

`sparse_negative_zeros, scalars=zero_lo`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 1.8 ns | 3.6 ns | 2.02x [2.00 – 2.05] | 1.1 us | 600x [593 – 607] |
| 64 | 2.9 ns | 3.5 ns | 1.18x [1.18 – 1.18] | 1.1 us | 373x [373 – 374] |
| 256 | 7.0 ns | 11.1 ns | 1.58x [1.58 – 1.59] | 1.1 us | 161x [161 – 161] |
| 1,024 | 25.6 ns | 45.6 ns | 1.78x [1.78 – 1.79] | 1.3 us | 50.3x [50.3 – 50.4] |
| 4,096 | 100.6 ns | 184.2 ns | 1.83x [1.83 – 1.83] | 1.6 us | 15.6x [15.6 – 15.7] |
| 16,384 | 949.3 ns | 1.1 us | 1.20x [1.20 – 1.20] | 2.5 us | 2.66x [2.66 – 2.67] |
| 65,536 | 3.8 us | 4.5 us | 1.20x [1.20 – 1.20] | 6.3 us | 1.69x [1.69 – 1.69] |
| 262,144 | 41.3 us | 28.5 us | 0.69x [0.69 – 0.69] | 320.3 us | 7.75x [7.74 – 7.75] |
| 1,048,576 | 195.3 us | 129.7 us | 0.66x [0.66 – 0.66] | 1.25 ms | 6.40x [6.39 – 6.40] |
| 4,194,304 | 950.2 us | 979.9 us | 1.03x [1.03 – 1.04] | 1.05 ms | 1.11x [1.10 – 1.12] |
| 16,777,216 | 3.98 ms | 5.18 ms | 1.30x [1.29 – 1.31] | 7.53 ms | 1.89x [1.88 – 1.91] |
| overall |  |  | 1.24x [1.24 – 1.25] |  | 15.6x [15.6 – 15.6] |

`rounded`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 1.8 ns | 3.5 ns | 1.97x [1.94 – 2.00] | 1.1 us | 599x [595 – 604] |
| 64 | 3.1 ns | 3.3 ns | 1.06x [1.05 – 1.06] | 1.1 us | 349x [349 – 350] |
| 256 | 7.0 ns | 11.1 ns | 1.58x [1.58 – 1.59] | 1.1 us | 160x [160 – 160] |
| 1,024 | 25.5 ns | 45.6 ns | 1.78x [1.78 – 1.79] | 1.3 us | 50.5x [50.4 – 50.5] |
| 4,096 | 100.7 ns | 175.3 ns | 1.74x [1.74 – 1.74] | 1.6 us | 15.7x [15.6 – 15.7] |
| 16,384 | 948.2 ns | 1.1 us | 1.20x [1.20 – 1.20] | 2.5 us | 2.67x [2.67 – 2.67] |
| 65,536 | 3.8 us | 4.5 us | 1.20x [1.20 – 1.20] | 6.4 us | 1.69x [1.69 – 1.69] |
| 262,144 | 41.1 us | 26.5 us | 0.64x [0.64 – 0.64] | 23.8 us | 0.58x [0.58 – 0.58] |
| 1,048,576 | 197.0 us | 128.7 us | 0.65x [0.65 – 0.65] | 124.4 us | 0.63x [0.63 – 0.63] |
| 4,194,304 | 965.5 us | 996.5 us | 1.03x [1.03 – 1.04] | 1.06 ms | 1.10x [1.10 – 1.11] |
| 16,777,216 | 3.46 ms | 4.58 ms | 1.32x [1.32 – 1.33] | 7.47 ms | 2.16x [2.14 – 2.17] |
| overall |  |  | 1.22x [1.21 – 1.22] |  | 10.0x [10.0 – 10.0] |

`rounded, scalars=zero_lo`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 1.8 ns | 3.6 ns | 2.02x [1.99 – 2.04] | 1.1 us | 602x [597 – 606] |
| 64 | 2.7 ns | 3.3 ns | 1.21x [1.21 – 1.21] | 1.1 us | 408x [407 – 409] |
| 256 | 7.0 ns | 11.1 ns | 1.58x [1.58 – 1.58] | 1.1 us | 162x [162 – 162] |
| 1,024 | 25.7 ns | 45.8 ns | 1.78x [1.78 – 1.79] | 1.3 us | 50.1x [50.0 – 50.1] |
| 4,096 | 101.0 ns | 175.2 ns | 1.73x [1.73 – 1.74] | 1.6 us | 15.6x [15.5 – 15.6] |
| 16,384 | 948.4 ns | 1.1 us | 1.21x [1.21 – 1.21] | 2.5 us | 2.68x [2.68 – 2.68] |
| 65,536 | 3.8 us | 4.5 us | 1.20x [1.20 – 1.20] | 6.3 us | 1.69x [1.69 – 1.69] |
| 262,144 | 41.2 us | 26.5 us | 0.64x [0.64 – 0.64] | 23.9 us | 0.58x [0.58 – 0.58] |
| 1,048,576 | 197.1 us | 128.7 us | 0.65x [0.65 – 0.65] | 124.3 us | 0.63x [0.63 – 0.63] |
| 4,194,304 | 962.4 us | 1.01 ms | 1.04x [1.04 – 1.06] | 1.05 ms | 1.09x [1.09 – 1.10] |
| 16,777,216 | 3.50 ms | 4.60 ms | 1.31x [1.31 – 1.32] | 7.59 ms | 2.17x [2.15 – 2.18] |
| overall |  |  | 1.23x [1.23 – 1.23] |  | 10.2x [10.2 – 10.2] |

All distributions: 1.25x [1.24 – 1.25] over naive C, 12.7x [12.7 – 12.7] over NumPy

With the output buffer 16 bytes behind its input ("apart" is the same kernel with the output clear of the input):

`random`:

| n | kernel | apart | speedup [95%] | naive C | speedup [95%] |
|---|---|---|---|---|---|
| 4,096 | 100.0 ns | 100.0 ns | 1.00x [1.00 – 1.00] | 175.1 ns | 1.75x [1.75 – 1.75] |
| 16,384 | 960.4 ns | 957.8 ns | 1.00x [1.00 – 1.00] | 1.1 us | 1.11x [1.11 – 1.11] |
| 65,536 | 3.8 us | 3.7 us | 0.98x [0.98 – 0.98] | 4.2 us | 1.11x [1.11 – 1.11] |
| 262,144 | 15.9 us | 46.8 us | 2.94x [2.93 – 2.95] | 51.8 us | 3.25x [3.25 – 3.26] |
| 1,048,576 | 126.4 us | 181.8 us | 1.44x [1.44 – 1.44] | 201.5 us | 1.59x [1.59 – 1.60] |
| 4,194,304 | 985.9 us | 1.21 ms | 1.23x [1.22 – 1.24] | 1.02 ms | 1.04x [1.04 – 1.05] |
| 16,777,216 | 3.25 ms | 3.79 ms | 1.17x [1.16 – 1.18] | 4.69 ms | 1.44x [1.44 – 1.45] |
| overall |  |  | 1.29x [1.29 – 1.29] |  | 1.50x [1.49 – 1.50] |

No stall. The survey found NumPy 8x slower in this placement and the naive C 1.5x; with the loads a block ahead, the kernel is not slower in it at any size.

What holds it:

- **Up to 1,024:** two instructions per vector on two ports: eight elements per cycle.
- **4,096 to 4,194,304:** the caches and memory. The kernel moves the same bytes as the naive C and NumPy, and is level with both at 1,048,576.
- **16,777,216:** the cache-bypassing stores: 1.3x over the naive C, 2.0x over NumPy.

The candidates the design lists for this family (`docs/x86-64-design.md`):

| Candidate | Outcome |
|---|---|
| `vmaxps` and `vminps` with the bound first and the data second compute the reference bit for bit for every ordered pair of bounds | Kept, and confirmed by the judge's twelve pairs of bounds and by the test above |
| All three `clip` caveats of the ARM64 kernel would not exist on x86-64 | **They do not exist in this kernel; whether the documents and the binding say so is a person's decision.** See below |
| The compiler very likely emits the same two instructions, so the kernel would ship as C | It does not: it emits a minimum, a compare and a blend. The assembly is 1.25x over it (`float32`) and 1.08x (`float64`) |
| Non-temporal stores above the third-level cache | Kept |
| An output just behind the input: a path that does not stall, perhaps the only place a kernel wins | No stall, by loading ahead; and it is not the only place it wins |
| Alignment of data | Stores aligned; loads as they come |
| Masked loads for leftover elements | Not needed: the leftover is a block that overlaps the one before |
| Instructions slow on other x86-64 processors | None used |

**For a person to decide: the caller's constraints on x86-64.** The table "What a caller must know" has, for the ARM64 `clip` kernels: `x` and `out` must not overlap; reads and writes FPSR; with `lo = +0.0`, data with a negative zero in its first few thousand elements runs at the speed of the C. None of the three is true of this kernel:

- **No path depends on the bounds or the data,** and negative zeros cost nothing.
- **No status register is read.** One flag is still written: `vmaxps` and `vminps` set the invalid flag in MXCSR whenever an operand is a NaN, quiet or signalling, on every path and at every size. So the binding's clearing of flags after `clip` is needed on x86-64 as on ARM64, and for every NaN, where the ARM64 kernel sets the flag for a quiet NaN only on its compare-and-select path.
- **`out` = `x` gives the right result,** because every element is computed from its own input and clamping twice changes nothing. Tested outside the judge at every size above. A partial overlap is still wrong, as for any vector kernel.

The kernel is not allowed to remove a caveat on its own authority, and nothing in the binding is changed here. The binding sends in-place `clip` to NumPy today (issue 7); on x86-64 it would not have to.

Left untried: a prefetch; the cache-bypassing stores between 2 MB and 24 MB of output (see `cumsum_i64`, x86-64 attempt 5).

Found here and worth knowing on ARM64: nothing that carries over; the difference is in what the two instruction sets' maximum and minimum do with zeros and NaN.
