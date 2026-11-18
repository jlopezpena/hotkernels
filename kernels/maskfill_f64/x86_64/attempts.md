# maskfill_f64 attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, locked, pinned to one performance core by the judge.

What the compiler does with `naive.c` (`clang -O3 -march=native`): a branch per element. With half selected at random it is mispredicted half the time, 2.4 ns per element at 65,536; with 5% selected it is 0.12 ns per element. NumPy takes 3.0 ns per element at 1,048,576 with half selected.

## 1. Baseline: widen four mask bytes, blend and store back

- **Idea:** the ARM64 kernel's scheme, four elements per step. `vpmovsxbq` widens four mask bytes to four 64-bit lanes, `vpcmpeqq` against zero makes the lane mask, and the four elements are loaded, combined with the value by `vpand`, `vpandn` and `vpor`, and stored back. An element that is not selected is rewritten with its own bits. The value's bits are only ever copied. A mask byte is true when it is not zero. The last 1 to 3 elements take a scalar loop with a branch.
- **check:** pass
- **compare:** not run (first version). `run`: 2.80x over naive C overall (half_selected 5.84x, few_selected 1.34x), 11.6x over NumPy.
- **Kept:** yes
- **Notes:** 0.10 ns per element from 1,024 to 65,536, whatever the mask. `vblendvpd` was not used for the blend: measured on this core it runs once per cycle, and three logical instructions run three per cycle.

## 2. Blocks with at most four selected elements are stored one by one, with no load

- **Idea:** the mask bits of 32 elements come from one `vpcmpeqb` and one `vpmovmskb`. A block with at most four set bits is not blended: `tzcnt` and `blsr` take the positions out, and the value is stored to each, a fixed four stores. `tzcnt` sets the carry flag when no bit is left, and a `cmovnc` then leaves the store address where it was, so the extra stores write the value to a selected element a second time, or, in a block with no bit at all, to eight bytes below the stack pointer. No element is loaded, and a cache line with no selected element is not touched.
- **check:** pass
- **compare:** significantly faster, 1.366x [1.349 – 1.384]
- **Kept:** yes
- **Notes:** few_selected 1.92x (2.05x at 64, 2.4x at 1,024, 2.0x at 65,536, 1.6x at 1,048,576, 1.44x at 16,777,216). half_selected 0.97x: 0.93x to 0.96x up to 1,024 for the count and the branch each block now pays, nothing from 65,536. In sparse blocks no unselected element is written, but dense blocks still rewrite them, so the caller's constraint is unchanged.

## 3. The design's candidate: a masked store in place of the blend (`vmaskmovpd`)

- **Idea:** `vpmovzxbq`, a subtraction from zero to put the selection in the sign bit, and `vmaskmovpd`. Three instructions per four elements, no load of x, and unselected elements are not written at all.
- **check:** pass
- **compare:** significantly faster, 1.062x [1.058 – 1.065]
- **Kept:** no. Not mine to keep: it removes a caveat the ARM64 kernel has (decision 1 in `docs/kernel-findings.md`, the rewrite of unselected elements) and would make the binding's constraint differ by architecture. Reported for a person to decide.
- **Notes:** the two halves of the result point opposite ways. half_selected: 1.55x at 64, 1.64x at 256, 1.63x at 1,024, 1.58x at 4,096, then **0.86x at 65,536, 0.87x at 1,048,576** and 0.94x at 16,777,216. few_selected: 1.00x to 1.03x, since sparse blocks do not reach this code. Without the load of x nothing announces to the processor which lines the stores are about to need, so beyond the first-level cache each masked store waits for its line. A prefetch for the store's lines was not tried, the variant not being a candidate to keep. Masked stores are also reported slow on some AMD processors (open issue 4 of the design), untested here.

## 4. Lane masks of a dense block from its byte mask, stored below the stack pointer

- **Idea:** the block's `vpcmpeqb` result is 0xff where an element stays. Stored below the stack pointer and read back four bytes at a time with `vpmovsxbq`, it gives the lane mask in one vector instruction and not two.
- **check:** pass
- **compare:** no significant difference, 1.001x [0.988 – 1.013]
- **Kept:** no
- **Notes:** half_selected 0.91x at 4,096 and 0.93x at 65,536, 1.22x at 16.

## 5. Lane masks of a dense block from a 16-entry table

- **Idea:** the dense block's mask bits are already in a general register. Four bits index a table of 16 lane masks (512 bytes), so the mask is one load with no vector instruction, and a group is `vpandn` with the elements, `vpand` with the value, `vpor` and the store.
- **check:** pass (after one failure: the table's base register was also the sparse path's store address)
- **compare:** significantly faster, 1.021x [1.020 – 1.023]
- **Kept:** yes
- **Notes:** half_selected 1.05x to 1.08x up to 1,024, 1.04x at 65,536, nothing at 4,096 or from 1,048,576. No size slower.

## 6. Bring the blocks to a 32-byte boundary

- **Idea:** every load and every store of the blend straddles two cache lines at every second group when x starts 16 bytes into a line, which it usually does. A micro-benchmark outside the judge puts that at 15% (0.094 against 0.111 ns per element at 1,024 and at 65,536). Up to three elements are done one at a time with `cmovne` first.
- **check:** pass
- **compare:** no significant difference, 1.007x [0.998 – 1.016]
- **Kept:** no
- **Notes:** half_selected 1.05x (1.21x at 65,536, 1.06x at 1,024, nothing at 4,096 or 1,048,576), few_selected 0.97x (0.82x at 64, 0.90x at 256): the sparse path has no load to align and pays for the head.

## Where it stands

Final `judge run`:

`half_selected`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 3.4 ns | 6.8 ns | 2.01x [1.99 – 2.02] | 313.7 ns | 92.3x [91.7 – 93.0] |
| 64 | 7.3 ns | 22.5 ns | 3.10x [3.09 – 3.11] | 358.5 ns | 49.4x [49.3 – 49.4] |
| 256 | 26.2 ns | 84.1 ns | 3.21x [3.20 – 3.21] | 540.6 ns | 20.6x [20.6 – 20.6] |
| 1,024 | 100.9 ns | 333.4 ns | 3.30x [3.30 – 3.31] | 1.3 us | 13.3x [13.3 – 13.4] |
| 4,096 | 369.1 ns | 1.3 us | 3.55x [3.55 – 3.55] | 4.3 us | 11.6x [11.6 – 11.7] |
| 16,384 | 2.0 us | 5.9 us | 2.97x [2.96 – 2.99] | 40.5 us | 20.4x [20.4 – 20.4] |
| 65,536 | 6.3 us | 156.9 us | 24.7x [24.7 – 24.8] | 195.3 us | 30.8x [30.7 – 30.8] |
| 262,144 | 34.9 us | 657.5 us | 18.8x [18.8 – 18.9] | 798.8 us | 22.9x [22.9 – 23.0] |
| 1,048,576 | 174.5 us | 2.66 ms | 15.2x [15.0 – 15.5] | 3.21 ms | 18.4x [18.2 – 18.7] |
| 4,194,304 | 1.26 ms | 10.66 ms | 8.48x [8.41 – 8.55] | 12.89 ms | 10.2x [10.2 – 10.3] |
| 16,777,216 | 6.31 ms | 42.64 ms | 6.76x [6.74 – 6.79] | 51.89 ms | 8.23x [8.20 – 8.26] |
| overall |  |  | 5.87x [5.86 – 5.88] |  | 20.9x [20.8 – 20.9] |

`few_selected`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 3.3 ns | 3.7 ns | 1.10x [1.08 – 1.12] | 259.0 ns | 77.5x [77.2 – 77.7] |
| 64 | 3.6 ns | 9.4 ns | 2.61x [2.60 – 2.62] | 305.2 ns | 84.8x [84.8 – 84.9] |
| 256 | 11.1 ns | 30.6 ns | 2.74x [2.72 – 2.75] | 352.3 ns | 31.6x [31.6 – 31.6] |
| 1,024 | 37.7 ns | 122.5 ns | 3.25x [3.25 – 3.26] | 593.3 ns | 15.7x [15.7 – 15.8] |
| 4,096 | 216.1 ns | 469.1 ns | 2.17x [2.17 – 2.17] | 1.4 us | 6.60x [6.58 – 6.61] |
| 16,384 | 808.8 ns | 2.0 us | 2.41x [2.41 – 2.42] | 4.6 us | 5.73x [5.73 – 5.73] |
| 65,536 | 3.0 us | 7.9 us | 2.67x [2.65 – 2.69] | 32.5 us | 11.0x [10.9 – 11.1] |
| 262,144 | 12.6 us | 99.7 us | 7.88x [7.84 – 7.93] | 183.3 us | 14.5x [14.4 – 14.5] |
| 1,048,576 | 103.0 us | 451.6 us | 4.38x [4.38 – 4.39] | 752.8 us | 7.31x [7.29 – 7.32] |
| 4,194,304 | 739.1 us | 1.82 ms | 2.46x [2.45 – 2.47] | 3.06 ms | 4.13x [4.12 – 4.15] |
| 16,777,216 | 4.42 ms | 7.25 ms | 1.64x [1.64 – 1.64] | 12.30 ms | 2.78x [2.78 – 2.79] |
| overall |  |  | 2.67x [2.67 – 2.68] |  | 12.9x [12.9 – 12.9] |

All distributions: 3.96x [3.95 – 3.97] over naive C, 16.4x [16.4 – 16.4] over NumPy

The kernel writes no buffer other than x, so the judge has no close-placement table for it.

What holds it:

- **half_selected: 0.09 to 0.12 ns per element up to 65,536,** about two cycles per four elements, and 0.13 to 0.17 at 262,144 and 1,048,576. Knock-out variants in a micro-benchmark at 1,024 elements (no store, no load of x, no widening of the mask) each ran at 1.6 to 2.0 cycles, so no single part is the limit. The naive C is so far behind here (a mispredicted branch every second element) that the speedup over it is 15x to 25x from 65,536 to 1,048,576. Below that the naive C's branch is predicted from the repeated timing inputs, and the speedup is 2x to 4x.
- **few_selected:** the sparse block, 0.035 to 0.05 ns per element up to 262,144, 2x to 8x over the naive C from 64 elements up.
- **16,777,216:** memory. Every line of x is read and written back with half selected.

The candidates the design lists for this kernel (`docs/x86-64-design.md`):

| Candidate | Outcome |
|---|---|
| `vmaskmovpd` stores only the selected elements | Measured, not kept (attempt 3): 1.6x faster up to 4,096 elements with half selected, 0.86x at 65,536 and 1,048,576. It removes the rewrite caveat, so the decision is a person's |
| The same load, blend and store | Kept (attempts 1 and 5), with the lane mask from a table |
| Masked loads and stores for leftover elements | Not tried: open issue 3 of the design. The last 1 to 3 elements take a scalar loop |
| Non-temporal stores | Not tried: the kernel stores to the buffer it has just read |
| Alignment of data | Rejected (attempt 6) |
| Alignment of code | The block loop starts on a 64-byte boundary. Not varied |
| An output just behind the input | Does not apply: one buffer |
| Instructions slow on other x86-64 processors | None in the kept kernel. Attempt 3's `vmaskmovpd` is one |

Left untried: a prefetch of the store's lines for attempt 3; a second sparse tier for blocks with five to eight selected elements (the judge times 5% and 50%, so `compare` could not see it).

Found here and worth trying on ARM64: nothing new; the store-only sparse block is the ARM64 kernel's own attempt 7.
