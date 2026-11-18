# fmax_reduce_f64 attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, pinned to one performance core by the judge. The runs are `--unlocked`: `spec.py` and `naive.c` are new in the same pull request and not on `main` yet. Neither was changed after the first line of assembly was written. Other judges were running on other cores of the machine during these runs, so the sizes from 1,048,576 up are noisy.

**What NumPy does, found by experiment before `naive.c` was written** (NumPy 2.5.5 on this machine, whose loop for this reduction is the AVX2 one, four float64 to a vector). For every length from 1 to 44, every pair of positions was given `+0.0` and `-0.0` among elements of -1 (and among NaNs), which orders the positions by whose zero wins; an array of NaNs with a different payload in every element shows which one is returned. Neither depends on where the array starts in memory (eight offsets), and `np.fmax.reduceat` gives a run the bits `np.fmax.reduce` gives the same elements.

- The order of the positions is that of a loop with four running maxima, one per lane, all starting at `x[0]`, taking `x[1..4]`, `x[5..8]` and so on, then combining the lanes, then taking the `(n - 1) mod 4` leftover elements one at a time. At 14 elements the zero that wins is the first present of 12, 8, 4, 0, 10, 6, 2, 11, 7, 3, 9, 5, 1, 13: lane 3 from its end back to `x[0]`, then lanes 1, 2 and 0 from their ends, then the leftover element. Up to four elements it is the first zero of the array.
- In a lane, and between lanes, the later of two equal values is kept (the lanes are combined 0 with 2, 1 with 3, then those two). Among the leftover elements the earlier is kept: they go through the C library's `fmax` with the element as its first argument.
- An array of NaNs gives `x[0]`, sign and payload, when `(n - 1) mod 4` is 0, and its last element otherwise.
- A model of that loop in Python agreed with NumPy on 5,820 arrays of 1 to 100,003 elements holding zeros of both signs, quiet NaNs of four payloads and both signs, minus infinity and negative values; `naive.c`, which is the same loop, on 60,000 arrays of 1 to 20,040 elements outside the judge, at eight placements, with infinities, denormals and ordinary values as well.

This is the answer of NumPy's AVX2 loop. A NumPy that runs its AVX-512 loop (eight to a vector) or its NEON loop will order the positions differently; nothing here was run on such a machine.

What the compiler does with `naive.c` (`clang -O3`): it turns the four lanes into vector code, 0.29 to 0.45 ns per element, and the leftover loop into branches on the data, which random data mispredicts: called on runs of 2, 3 and 4 elements it takes 5, 7.5 and 10.7 ns, on runs of 5 it takes 2.2 ns.

Figures "per call on runs of L" are from a driver outside the judge: 8,192 elements in the first-level cache, cut into consecutive runs of L, the kernel called once per run as a binding of `np.fmax.reduceat` would call it, the best of 400 passes. The judge's smallest timed size is 16.

## 1. Baseline: NumPy's loop step for step, NaN to minus infinity, a chain of one `vmaxpd`

- **Idea:** `vmaxpd a, b` is "a if a > b, otherwise b", so with the running value as a it keeps the later of two equal values, which is the rule of NumPy's lanes; `vmaxsd v, r` keeps the earlier, the rule of the leftover elements. Neither ignores a NaN, so every element first goes through a `vmaxpd` against minus infinity. The result is then NumPy's unless every element is a NaN; a result of minus infinity is looked at again, one element at a time, and an array of NaNs returns `x[0]` or its last element. One element returns at once.
- **check:** pass. Outside the judge: 60,000 arrays against NumPy, no difference.
- **compare:** n/a (first version). `run`: 2.38x [2.38 – 2.39] over naive C on `random`, 2.37x [2.36 – 2.37] on `some_nan`; 2.26x and 2.54x over NumPy.
- **Kept:** yes
- **Notes:** 2.9 ns at 16 (1.80x over the naive C), 8.3 ns at 64, 0.18 ns per element from 4,096 to 1,048,576, which is the chain: one cycle per element. NumPy takes 0.096 ns per element at 16,384 to 262,144, so this is at 0.5x of it there. Per call on runs of 2, 3, 4, 8, 16, 32: 1.1, 1.2, 1.5, 1.8, 2.3, 3.2 ns (naive C: 5.2, 7.5, 10.7, 3.9, 5.6, 9.7).

## 2. Blocks of four vectors, combined with one another before they meet the running maximum

- **Idea:** the chain of attempt 1 is four cycles per vector. Combining is associative as long as the earlier value is the a of each `vmaxpd` and the later one the b (the later of two equal values survives either way), which is how NumPy's own loop goes through eight vectors at a time. Four cleaned vectors are combined in pairs, then the pairs, then the result with the running maximum: one `vmaxpd` on the chain per 16 elements. Whole blocks first, then single vectors, then the leftover elements.
- **check:** pass. Outside the judge: 60,000 arrays against NumPy, no difference.
- **compare:** significantly faster, 1.974x [1.968 – 1.980]
- **Kept:** yes
- **Notes:** both distributions alike: 1.91x at 64, 2.0x at 256, 2.8x at 1,024, 3.66x at 4,096, 2.9x at 16,384 and 65,536, 2.4x at 262,144, 1.9x at 1,048,576, 1.12x at 4,194,304, nothing at 16 and at 16,777,216. 0.049 ns per element at 4,096: eight `vmaxpd` per block on two ports, four cycles per 16 elements, which is also the chain. Per call on runs of 8, 16, 32, 64: 1.8, 2.4, 2.9, 4.9 ns.

## 3. No branch on n mod 4: groups taken twice, the leftover elements as one vector, a masked load up to four elements

- **Idea:** called once per run of a `reduceat`, the kernel sees a different length every call, and attempt 2 has three loops whose trip counts follow the length: on runs of mixed lengths (65,536 runs of random length, so that the predictor cannot learn them) it takes 5.2 ns per call for lengths 1 to 4, 8.1 for 5 to 16 and 11.9 for 2 to 64, against 1.2 to 2.4 ns on runs of one length. Keeping the later of two equal values is associative, and a group may be combined twice as long as no earlier group follows it. So: for one to three groups (5 to 16 elements), three loads, the first group, the second or the first again, and the last; from four groups, blocks of four, the last block being the last four groups and overlapping the one before. The leftover elements are the tail of the vector of the last four elements: the lanes before them are set to minus infinity (`vminpd` against a vector of infinities picked by the count), and the vector is reduced keeping the earlier of two equal values. Up to four elements there is no group, and the result is the first of the equal maxima: one `vmaskmovpd` load of the first n elements, reduced the same way.
- **check:** pass. Outside the judge: 60,000 arrays against NumPy, no difference.
- **compare:** significantly faster, 1.023x [1.019 – 1.026]
- **Kept:** yes
- **Notes:** 1.38x and 1.40x at 16 (2.8 ns to 2.1 ns); 0.91x at 256 on both distributions (12.9 ns to 14.2 ns: 255 elements after the first are 63 groups, which was 15 blocks and three vectors and is now 16 blocks); nothing elsewhere. What `compare` does not see, per call: runs of 2, 3, 4: 1.0 ns each (1.2, 1.3, 1.5 before); of 5 and 8: 1.9 (1.3, 1.8); of 16: 1.9 (2.4); of 32 and 64: 2.8 and 4.9 (as before). Mixed lengths 1 to 4: 1.0 ns (5.2 before); 5 to 16: 1.9 (8.1); 2 to 8: 3.5 (7.0); 2 to 32: 5.0 (10.4); 2 to 64: 6.1 (11.9); 17 to 64: 6.1 (11.7). What is left there is the branch between the paths and the count of the block loop. `vmaskmovpd` is the one masked load: fast on this machine, slow on some other x86-64 processors, and slow anywhere if a lane it masks off lies in a page that cannot be read, which for this load means an array of at most three elements that ends within 24 bytes of the end of mapped memory.

## 4. A quick way in front: one `vmaxpd` per vector in any order, NumPy's way only for a zero with a sign bit about

- **Idea:** attempt 3 spends two `vmaxpd` per vector, one to clean and one to combine, and both of this processor's ports for it are full: 42 of them in a call on 64 elements. The value of the maximum needs neither NumPy's order nor the cleaning: `vmaxpd v, m` is m when v is a NaN, so with the maximum so far as m, starting at minus infinity, one instruction per vector ignores NaN, and the vectors may overlap freely and end with the array. Only the sign of a zero and the NaN of an array of NaNs need NumPy's order. So the quick way also ors the bits of every element together (`vorpd`, which has a third port), and its result stands unless it is a zero and some element has its sign bit set (then attempt 3's code runs, from the start), or it is minus infinity (then the array is looked at again for a number). A zero maximum with no sign bit in the array is the positive zero: all-zero and non-negative data never leave the quick way. 5 to 16 elements are four vectors, two at the ends and two between; from 17, blocks of four vectors into four maxima, the last block ending with the array. Up to four elements stay as they were. The second look at an array of NaNs, which was one element at a time, is a vector at a time from 5 elements.
- **check:** pass. Outside the judge: 200,000 arrays against NumPy, no difference.
- **compare:** significantly faster, 1.117x [1.112 – 1.121]; a first run, before the second look was made a vector loop: 1.120x [1.120 – 1.121].
- **Kept:** yes
- **Notes:** both distributions alike: 1.25x to 1.28x at 16 (2.1 ns to 1.6 ns), 1.29x at 64 (4.4 to 3.4), 1.30x at 256, 1.16x at 1,024, 1.03x at 4,096, 1.37x at 16,384, nothing from 65,536 to 1,048,576, 0.97x and 0.98x at the two largest sizes in both runs (4.47 ms to 4.53 ms at 16,777,216). From 1,024 the four maxima are a chain of four cycles per 16 elements, as attempt 3's blocks were. Per call on runs of 5, 8, 16, 32, 64: 1.5, 1.5, 1.5, 2.1, 3.6 ns (1.9, 1.9, 1.9, 2.8, 4.9 before); mixed lengths 5 to 16: 1.6 ns (1.9), 2 to 64: 5.4 (6.1). **What it costs, which `compare` does not see:** data whose maximum is a zero while a negative value, a negative zero or a NaN with its sign bit set is present goes through the array twice. On runs that are negative values with one zero in eight, per call on runs of 8, 16, 32, 64: 2.9, 2.9, 4.4, 7.2 ns, and 0.095 ns per element at 1,024, twice the quick way and 1.5 times attempt 3. Runs of positive zeros only: 1.5, 1.5, 2.0, 3.6 ns, as random data. Runs of NaNs only: 2.1, 2.5, 3.7, 7.0 ns.

## 5. Eight maxima from 128 elements

- **Idea:** the four maxima of attempt 4 are a chain of four cycles per 16 elements, and a call on 256 elements or more is long enough to be held by it. From 128 elements: blocks of eight vectors into eight maxima, the last block ending with the array; the bits are or-ed into the same two registers.
- **check:** pass. Outside the judge: 200,000 arrays against NumPy, no difference.
- **compare:** significantly faster, 1.109x [1.108 – 1.109]. A first version that took this path from 64 elements: significantly faster, 1.110x [1.099 – 1.121], but 0.92x and 0.93x at 64 (3.4 ns to 3.7 ns): setting up eight maxima and merging them is a dozen instructions more. From 128 the row at 64 is 1.02x.
- **Kept:** yes
- **Notes:** 1.07x at 256, 1.20x at 1,024, 1.46x on `random` and 1.22x on `some_nan` at 4,096, 1.22x at 16,384, 1.10x at 65,536, 1.04x to 1.06x at 262,144, 0.99x and 1.00x at 1,048,576, 1.13x and 1.12x at the two largest sizes; 16 unchanged. 0.032 ns per element at 4,096 on `random`: 5.5 cycles per block, which is its 16 `vmaxpd` and `vorpd` on three ports. The `some_nan` row at 4,096 is 159 ns against 133 ns in both runs, and equal to `random` at 1,024 and 16,384; not explained.

## 6. Three instructions fewer on the way through 5 to 16 elements

- **Idea:** the third of the four vectors was placed with four instructions (32 bytes before the last, or the first if that is before the array). As far before the last vector as the second is after the first covers the same elements and is a subtraction. And the address of the table of infinities is only needed up to four elements and on NumPy's way, so it is taken there, and the quick way reads minus infinity relative to `rip`.
- **check:** pass. Outside the judge: 200,000 arrays against NumPy, no difference.
- **compare:** significantly faster, 1.004x [1.002 – 1.006]
- **Kept:** yes
- **Notes:** 1.05x [1.04 – 1.05] at 16 on both distributions (1.6 ns to 1.5 ns), which is the only path changed; the other rows are 0.98x to 1.02x, most of them not significant. Kept for the row at 16, the size nearest the runs this kernel is for.

## 7. The blocks of eight vectors on 32-byte boundaries

- **Idea:** the unexplained row of attempt 5 is the placement of that input, not its NaNs: in the final `run` before this attempt the two distributions had changed places at 4,096 (157 ns and 134 ns). In the driver, runs of 1,024 elements take 42 ns when x is at a multiple of 32 bytes and 62 ns at the three other multiples of 8: half of the loads then straddle two cache lines, and a block's eight loads need twelve of the two load slots a cycle. The quick way may take elements twice, so the first block is taken at x and the following ones from the last multiple of 32 bytes inside it. An x that is not at a multiple of 8 has no such boundary between elements and is left as it is. The first block now starts the eight maxima itself, in place of eight copies of minus infinity.
- **check:** pass. Outside the judge: 300,000 arrays against NumPy at eight placements, no difference.
- **compare:** significantly faster, 1.081x [1.073 – 1.089]
- **Kept:** yes
- **Notes:** 1.50x at 65,536 on both distributions (3.9 us to 2.6 us), 1.19x at 262,144, 1.09x and 1.10x at 1,024, 1.15x at 4,096 on `some_nan` and 0.99x (not significant) on `random`, 1.04x and 1.05x at the two largest sizes, nothing at 16, 64, 256 and 16,384. Which rows gain is which inputs the judge placed off a multiple of 32. In the driver, runs of 1,024 at the three other placements: 62 ns to 44 ns; at a multiple of 32: 42 ns to 44 ns, and runs of 256 there 12.0 ns to 13.0 ns, which the judge's row at 256 (10.0 ns before and after) does not show. The blocks of four vectors (17 to 127 elements) are not placed.

## Where it stands

Final `judge run` (unlocked): 6.28x [6.27 – 6.29] over naive C on `random`, 6.35x [6.35 – 6.35] on `some_nan`, 6.31x [6.31 – 6.32] over the two; 6.08x and 6.69x over NumPy, which is timed from Python with its per-call overhead.

`random`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 1.5 ns | 5.2 ns | 3.40x [3.40 – 3.40] | 284.0 ns | 184x [184 – 184] |
| 64 | 3.4 ns | 18.3 ns | 5.45x [5.45 – 5.45] | 289.2 ns | 86.1x [85.99 – 86.20] |
| 256 | 10.1 ns | 97.3 ns | 9.60x [9.59 – 9.60] | 307.3 ns | 30.3x [30.28 – 30.33] |
| 1,024 | 34.5 ns | 447.1 ns | 13.0x [12.95 – 12.96] | 419.8 ns | 12.2x [12.14 – 12.18] |
| 4,096 | 136.1 ns | 1.8 us | 13.6x [13.54 – 13.56] | 703.5 ns | 5.17x [5.17 – 5.17] |
| 16,384 | 651.3 ns | 7.4 us | 11.4x [11.42 – 11.42] | 1.8 us | 2.78x [2.78 – 2.79] |
| 65,536 | 2.6 us | 29.8 us | 11.3x [11.35 – 11.35] | 6.3 us | 2.40x [2.40 – 2.40] |
| 262,144 | 17.0 us | 119.4 us | 7.01x [7.00 – 7.02] | 26.0 us | 1.52x [1.52 – 1.53] |
| 1,048,576 | 134.3 us | 477.9 us | 3.56x [3.56 – 3.56] | 134.3 us | 1.00x [1.00 – 1.00] n.s. |
| 4,194,304 | 789.1 us | 2.10 ms | 2.66x [2.64 – 2.72] | 921.6 us | 1.17x [1.16 – 1.18] |
| 16,777,216 | 3.82 ms | 8.48 ms | 2.22x [2.21 – 2.23] | 4.43 ms | 1.16x [1.15 – 1.16] |
| overall |  |  | 6.28x [6.27 – 6.29] |  | 6.08x [6.07 – 6.08] |

`some_nan`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 1.5 ns | 5.2 ns | 3.37x [3.37 – 3.37] | 341.9 ns | 223x [223 – 223] |
| 64 | 3.4 ns | 18.6 ns | 5.46x [5.46 – 5.47] | 393.1 ns | 115x [115 – 115] |
| 256 | 10.1 ns | 97.5 ns | 9.68x [9.67 – 9.69] | 408.9 ns | 40.6x [40.52 – 40.62] |
| 1,024 | 34.1 ns | 447.1 ns | 13.1x [13.09 – 13.10] | 477.5 ns | 14.0x [13.95 – 14.02] |
| 4,096 | 134.5 ns | 1.8 us | 13.7x [13.69 – 13.71] | 770.7 ns | 5.73x [5.72 – 5.73] |
| 16,384 | 651.2 ns | 7.4 us | 11.4x [11.42 – 11.42] | 1.9 us | 2.88x [2.88 – 2.88] |
| 65,536 | 2.6 us | 29.8 us | 11.4x [11.35 – 11.36] | 6.4 us | 2.42x [2.42 – 2.43] |
| 262,144 | 16.4 us | 119.3 us | 7.27x [7.27 – 7.27] | 25.5 us | 1.55x [1.55 – 1.56] |
| 1,048,576 | 129.7 us | 477.7 us | 3.68x [3.68 – 3.69] | 129.5 us | 1.00x [1.00 – 1.00] |
| 4,194,304 | 746.2 us | 2.08 ms | 2.78x [2.77 – 2.79] | 864.9 us | 1.16x [1.16 – 1.16] |
| 16,777,216 | 3.85 ms | 8.44 ms | 2.19x [2.19 – 2.20] | 4.41 ms | 1.14x [1.14 – 1.15] |
| overall |  |  | 6.35x [6.35 – 6.35] |  | 6.69x [6.69 – 6.70] |

The kernel has no output buffer, so `run` prints no second table.

**Per call on runs, from the driver** (ns per call, x at a multiple of 32 bytes; the naive C column is for random data):

| runs of | random | naive C | 5% NaN | positive zeros only | negative values, a zero in eight | NaN only |
|---|---|---|---|---|---|---|
| 2 | 1.0 | 5.6 | 1.1 | 1.0 | 1.0 | 1.5 |
| 4 | 1.0 | 10.5 | 1.0 | 1.0 | 1.0 | 1.8 |
| 5 | 1.5 | 1.9 | 1.5 | 1.7 | 2.4 | 2.0 |
| 8 | 1.5 | 3.6 | 1.5 | 1.4 | 2.9 | 1.9 |
| 16 | 1.5 | 5.7 | 1.5 | 1.4 | 2.9 | 2.3 |
| 32 | 2.0 | 9.9 | 2.0 | 2.0 | 4.2 | 3.5 |
| 64 | 3.6 | 18.4 | 3.6 | 3.5 | 6.7 | 7.7 |
| 128 | 6.7 | 39.9 | 6.7 | 6.7 | 12.5 | 11.9 |
| 1,024 | 44.2 | 437.6 | 44.4 | 44.3 | 93.2 | 113.7 |
| mixed 1 to 4 | 1.0 | 10.3 | 1.2 | 1.0 | 1.0 | 5.7 |
| mixed 5 to 16 | 1.6 | 10.2 | 1.6 | 1.5 | 3.7 | 6.4 |
| mixed 2 to 8 | 3.4 | 11.1 | 3.3 | 3.3 | 5.2 | 5.9 |
| mixed 2 to 32 | 4.1 | 11.6 | 4.2 | 4.2 | 6.5 | 9.4 |
| mixed 2 to 64 | 5.6 | 13.7 | 5.5 | 5.6 | 8.6 | 12.1 |
| mixed 17 to 64 | 5.6 | 14.4 | 5.6 | 5.6 | 9.0 | 12.7 |

"Mixed" is 65,536 runs of random length in the range, which the branch predictor cannot learn; a binding of `reduceat` calls like that. Inside the ranges 1 to 4 and 5 to 16 the kernel has no branch on the length and the mixed figure is the fixed one. Across them it pays for the branch between the paths, and from 33 elements for the count of the block loop: 3.4 to 5.6 ns against 1.0 to 3.6 ns. The driver ran on a core whose sibling other work may have used, and its long runs are slower than the judge's (44 ns against 34.5 ns at 1,024).

**Three kinds of input that are slower than the tables**, none of which the judge times:

- **A zero maximum with a sign bit somewhere** (negative values, a negative zero or a NaN with its sign bit set, and no positive value): the quick way cannot tell whether a negative zero is present, and NumPy's order is gone through as well. Twice the time of the quick way: 2.9 ns on runs of 16, 0.09 ns per element on long ones. Non-negative data, all-zero runs included, stays on the quick way.
- **Every element a NaN** (or minus infinity the maximum): a second pass a vector at a time. 2.3 ns on runs of 16, 0.11 ns per element at 1,024. With one to four elements the second look is one element at a time with a branch each: 1.5 to 1.8 ns on fixed lengths, 5.7 ns on mixed ones.
- **x not at a multiple of 32 bytes, 17 to 127 elements:** those blocks are not placed (attempt 7 places the blocks from 128 elements). Not measured separately.

**What the judge's cases exercise.** The `zeros` class (both signs at random, every length to 65 and around every power of two) goes from the quick way to NumPy's order and checks the sign of the result at every count of groups and leftover elements, and up to four elements through the masked load. `nan_all` reaches the second look, but its NaNs all have the same bits, so it cannot tell x[0] from the last element: that rule, NaNs among zeros, and minus infinity among NaNs were checked outside the judge only (300,000 arrays against NumPy at eight placements, NaNs of four payloads and both signs). Nothing in the judge has a signaling NaN.

**What the kernel requires of its caller:**

- n is at least 1.
- No element is a signaling NaN. The kernel ignores one as it ignores any NaN; NumPy's leftover elements go through the C library's `fmax`, which does not. What NumPy returns for one was not looked at.
- The bits promised are those of a NumPy that runs its AVX2 loop for this reduction. Where NumPy runs its AVX-512 loop the two can differ in the sign of a zero maximum with zeros of both signs present, and in which NaN an array of NaNs returns. That was not run; it is a person's decision whether the binding may serve such a machine.
- `vmaxpd` and `vcmpordpd` are given quiet NaNs. If that leaves the invalid flag set in MXCSR, a binding that sits inside NumPy's floating-point error check has to clear it, as for `cumfmax_f64`. Not looked into here.

What holds it at each size:

- **Up to 16:** the count of instructions: about 30 for 5 to 16 elements. The calls of a run are independent, so they overlap in the processor and the length of the chain through a result does not show.
- **17 to 127:** twelve instructions per 16 elements and the loop's count.
- **128 to 4,096:** the ports: 16 `vmaxpd` and `vorpd` per 64 elements on three ports, 5.5 cycles, 0.033 ns per element.
- **16,384 and 65,536:** the second-level cache, 0.040 ns per element. NumPy is at about 0.10 there.
- **From 262,144:** the third-level cache and memory; equal to NumPy at 1,048,576, 1.14x to 1.17x over it above.

Not tried:

- **Leaving the `vorpd` out of the quick way** and or-ing the bits in a pass of its own when the maximum is a zero. By count that is 2 vectors a cycle in place of 1.5 from 128 elements inside the first-level cache, and three instructions fewer up to 16; it makes all-zero runs, which sparse data is full of, pay for a second pass.
- **Telling a negative zero from a negative value in the quick way** (`vpcmpgtq` against the smallest integer, or-ed), so that only a zero maximum with a negative zero present takes NumPy's order. Two instructions per vector in place of one.
- **NumPy's order for a zero maximum by looking for the zeros** from the end of each lane, in place of the second full pass.
- **One path for 5 to 32 or to 64 elements without a branch on the length** (eight or sixteen loads at clamped offsets). It would take the branch between paths out of mixed lengths and cost short runs instructions; `compare` times one length at a time and cannot see the gain.
- **Placing the blocks of four vectors** on 32-byte boundaries, as attempt 7 does for the blocks of eight.
- **The lanes of NumPy's order from loads at multiples of 32 bytes.** Its groups start at x[1], so for an x at a multiple of 32 every group load is 8 bytes off and every other one straddles two cache lines.
- **Arrays beyond the second-level cache** (prefetch, another loop). The kernel is for short runs.
