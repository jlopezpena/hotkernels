# argmax_f32 attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core), Linux, locked, pinned to one performance core by the judge. The four kernels of this family (`argmax` and `argmin`, `float32` and `float64`) are generated from one source and were written together; this log has the reasoning, and the other three have their tables.

**The opponent.** On ARM64 it was a scalar loop in NumPy. Here NumPy has an AVX2 loop: 0.083 ns per `float32` at 1,048,576 and 0.163 ns per `float64`. The compiler does not vectorise `naive.c` (`clang -O3 -march=native`): the comparison and the two early exits stay a scalar chain of 0.77 ns per element. So the column to read is NumPy's.

## 1. Block reduction with one rare branch, then the position from the remembered block

- **Idea:** the ARM64 kernel's shape (reduce blocks, find the position afterwards) with the design's candidates, in one pass over the array.
  - **M** is the extreme seen so far, in every lane of a register. A block of 64 `float32` (eight vectors) is reduced with `vmaxps` into four accumulators that are never reset, so a lane of an accumulator is beyond M exactly when the block holds a new extreme. One compare of the reduced accumulators with M says so.
  - **NaN.** `vmaxps` returns its second operand when either is a NaN, so a NaN can be lost from an accumulator, as the design says. `vcmpunordps` of two vectors is true where either holds a NaN, so four compares cover the eight vectors of a block.
  - **One branch per block** on "a new extreme or a NaN". It is taken about as often as the logarithm of the number of blocks on random data. When it is taken the block is looked at again: the first NaN in it ends the call; otherwise M becomes the block's extreme and the block is remembered.
  - **The position** is the first element equal to M in the remembered block: `vcmpeqps`, `vmovmskps` and `tzcnt`, with no scalar scan. The same three find the first NaN.
  - Fewer than 64 elements take one vector per step, compared with M directly; fewer than 8 take the reference loop. The leftover of a longer array is one more block that ends with the last element.
- **check:** pass, first time, all four kernels (20,313 cases each: NaN at every position class, ties, signed zeros, the extreme at chosen places)
- **compare:** not run (first version). `run`: 13.1x over naive C and 7.49x over NumPy overall; 1.91x over NumPy at 1,048,576.
- **Kept:** yes
- **Notes:** 0.030 ns per element from 4,096 to 65,536. A block is seventeen instructions for the two ports that take a maximum or a compare (eight maxima, four NaN compares, three to combine, the test against M and the mask), 8.5 cycles per 64 elements, and that is the speed measured.

## 2. An answer inside the array whatever happens to the data

- **Idea:** the final search is bounded by the length of the remembered range. If it finds nothing, which cannot happen unless another thread changed the array between the two looks, the kernel returned the range's end plus 32. It now returns the range's first index.
- **check:** pass
- **compare:** not run: three instructions after a loop that always exits early
- **Kept:** yes
- **Notes:** issue 56 asks for exactly this of the ARM64 kernels, whose second scan has no bound. Here both second looks are bounded by construction, so nothing outside the array is read in any case, and with this change the result is always a valid index.

## Where it stands

Final `judge run`:

`random`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 4.3 ns | 10.4 ns | 2.39x [2.39 – 2.40] | 529.1 ns | 122x [122 – 123] |
| 64 | 7.6 ns | 39.7 ns | 5.24x [5.23 – 5.24] | 554.9 ns | 73.3x [73.1 – 73.4] |
| 256 | 15.7 ns | 188.0 ns | 12.0x [12.0 – 12.0] | 560.4 ns | 35.6x [35.6 – 35.7] |
| 1,024 | 43.3 ns | 778.7 ns | 18.0x [18.0 – 18.0] | 617.9 ns | 14.3x [14.3 – 14.3] |
| 4,096 | 123.3 ns | 3.1 us | 25.5x [25.5 – 25.5] | 859.9 ns | 6.97x [6.97 – 6.98] |
| 16,384 | 487.5 ns | 12.6 us | 25.9x [25.9 – 25.9] | 1.9 us | 3.87x [3.87 – 3.88] |
| 65,536 | 1.9 us | 50.4 us | 26.6x [26.6 – 26.6] | 5.8 us | 3.07x [3.07 – 3.07] |
| 262,144 | 9.4 us | 201.8 us | 21.5x [21.5 – 21.5] | 22.0 us | 2.35x [2.35 – 2.35] |
| 1,048,576 | 45.6 us | 812.1 us | 17.8x [17.8 – 17.8] | 87.3 us | 1.92x [1.92 – 1.92] |
| 4,194,304 | 189.2 us | 3.23 ms | 17.1x [16.9 – 17.2] | 361.5 us | 1.91x [1.89 – 1.92] |
| 16,777,216 | 1.97 ms | 12.89 ms | 6.53x [6.52 – 6.57] | 2.56 ms | 1.30x [1.29 – 1.31] |
| overall |  |  | 13.1x [13.1 – 13.2] |  | 7.50x [7.49 – 7.51] |

What holds it:

- **16 to 1,024:** the steps of a call and its events. With sixteen blocks about three or four take the branch, each a mispredicted jump and a second look at 64 elements.
- **4,096 to 65,536:** the two ports for maxima and compares, 0.030 ns per element; 3x to 7x over NumPy, whose own cost per element is 0.08 ns at every size.
- **262,144 to 4,194,304:** the caches: 0.036 to 0.045 ns per element, 1.9x to 2.35x over NumPy.
- **16,777,216:** memory. 0.117 ns per element against NumPy's 0.153: 1.3x.

The candidates the design lists for this family (`docs/x86-64-design.md`):

| Candidate | Outcome |
|---|---|
| `vmaxps` loses a NaN and needs a test of its own; one `vcmpunordps` covers two vectors | Kept |
| The first position equal to the maximum is `vcmpeqps`, `vmovmskps` and `tzcnt`, with no scalar scan | Kept |
| The same pair finds the first NaN | Kept |
| Whether a block scheme beats NumPy's AVX2 loop outside the first-level cache | It does: 3x in the second-level cache, 1.9x to 2.35x in the third, 1.3x from memory |
| Masked loads for leftover elements | Not needed: the leftover is one more block that overlaps the one before |
| Alignment of data, a prefetch | Not tried. From 262,144 elements the kernel is held by the caches, which is where `sum_u8` gained 1.5x from aligned loads; here the loads are four per block with a memory operand each on top, and it is the first thing a second run should try |
| An output just behind the input | Does not apply: no output buffer |
| Instructions slow on other x86-64 processors | None used |

**What differs from the ARM64 kernels for a caller.** The ARM64 row in "What a caller must know" says the kernels read the array twice, that the search for a NaN the first pass saw has no bound, and that a concurrent writer can make them return `n` or read past the end. Here the second look is at one block, both looks are bounded, nothing outside the array is read, and the result is always an index inside it. A concurrent writer can still make the answer wrong, as it can for NumPy.

**One thing is new for a caller on x86-64: the invalid flag.** `vmaxps` and `vminps` set the invalid flag in MXCSR when an operand is a NaN, quiet or signalling, so these four kernels leave it set after a call on an array that holds a NaN. The ARM64 `fmax` does not for a quiet NaN. Whether NumPy looks at the flag after `argmax` and turns it into a warning has to be established by `uv run delivery check` when the package is built for x86-64; if it does, the binding clears the flag after these kernels as it does after `clip`. (Noted while writing the `clip` kernels, after this log's pull request was opened.)

Left untried: aligned loads and a prefetch (above); the NaN test as a running sum on the third vector port instead of compares on the two busy ones (it would raise floating-point flags); blocks of 128.

Found here and worth trying on ARM64: accumulators that are never reset and one branch per block for "new extreme or NaN", which make it one pass with a bounded second look. That would settle issue 56 there as well.
