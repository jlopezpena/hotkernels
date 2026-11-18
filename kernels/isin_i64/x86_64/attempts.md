# isin_i64 attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core), Linux, locked, pinned to one performance core by the judge.

What the compiler does with `naive.c` (`clang -O3 -march=native`): it vectorises. Four elements per register, eight `vpcmpeqq` against the broadcast test values, an OR tree, and a narrowing of the masks to bytes: 0.30 ns per element, 1.6 cycles, at every size up to 1,048,576 and for any values. NumPy takes 8.4 ns per element at 1,048,576 on small values and 1.1 ns on values spread over the int64 range.

So the design's question for the general case, whether the compiler already emits the 64-bit compare, is answered: it does.

## 1. Narrow keys in a window around the test values; 64-bit compares from the stack otherwise

- **Idea:** sixteen elements per block. When the test values lie within 32,766 of each other, the smallest is subtracted from every element; if all sixteen differences of a block are below 2^15 they are narrowed to 16-bit keys with three `vpackusdw` and compared with eight `vpcmpeqw`, and otherwise the block is compared at 64 bits against test values kept broadcast on the stack.
- **check:** fail, then pass. The failure was a register: the index of the block was also the scratch register of the code that builds the keys, so the first load was at a garbage address. Found with a differential test outside the judge.
- **compare:** not run (never committed). `run`: **0.77x against the naive C**, on both distributions.
- **Kept:** no
- **Notes:** two mistakes of design, not of detail. **(1) The window started at the smallest test value.** The judge's `mostly_misses` data is uniform from 0 to 999 with test values from the same range, so a tenth of the elements were below the window and nearly every block of sixteen held one: every block took the wide path after paying for the narrow attempt. **(2) The wide path's compares had memory operands**, eight loads per four elements where the processor does two 32-byte loads per cycle; 0.41 ns per element against the compiler's 0.30.

## 2. Keys for test values from 0 to 32,767; 64-bit compares from registers otherwise

- **Idea:** the window is 0 to 32,767, fixed, and a call uses the narrow path when every test value is in it: one OR of the eight values and a shift decide. Nothing is subtracted. A call with any other test values keeps them broadcast in eight registers and compares at 64 bits. A block of a narrow call that holds an element outside the window (it cannot be a member, and it cannot be narrowed) is compared at 64 bits with the test values broadcast from memory one at a time, which is slow and needs no register and no stack.
- **check:** pass. Differential test outside the judge against the reference loop: 400,000 arrays of up to 600 elements, test values in windows of 3 to 65,535 and unrestricted, at small and at random bases, with repeats as the binding pads them, and elements drawn from the test values, from just outside the window, from test values plus a high bit, and at random. No mismatch.
- **compare:** not run (first committed version). `run`: 1.45x over naive C overall (`mostly_misses` 1.94x, `wide_values` 1.08x), 40.7x over NumPy.
- **Kept:** yes
- **Notes:** the narrow path takes 0.124 ns per element up to 65,536, 0.64 cycles, against 1.6 for the compiler's loop: 2.4x. The wide path is the compiler's loop written by hand and gains what a serial OR and a cheaper narrowing to bytes give: 1.05x to 1.14x.

## Where it stands

Final `judge run`:

`mostly_misses`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 3.7 ns | 5.8 ns | 1.58x [1.58 – 1.60] | 7.8 us | 2123x [2121 – 2126] |
| 64 | 10.4 ns | 20.7 ns | 1.99x [1.99 – 1.99] | 7.8 us | 754x [753 – 754] |
| 256 | 32.0 ns | 80.3 ns | 2.51x [2.51 – 2.51] | 7.7 us | 239x [239 – 239] |
| 1,024 | 127.1 ns | 310.1 ns | 2.44x [2.44 – 2.44] | 9.5 us | 75.0x [74.9 – 75.0] |
| 4,096 | 473.9 ns | 1.2 us | 2.59x [2.59 – 2.59] | 19.0 us | 40.1x [40.1 – 40.1] |
| 16,384 | 2.0 us | 4.9 us | 2.43x [2.43 – 2.43] | 79.9 us | 39.5x [39.5 – 39.5] |
| 65,536 | 8.1 us | 19.6 us | 2.43x [2.43 – 2.43] | 354.3 us | 43.9x [43.8 – 43.9] |
| 262,144 | 33.1 us | 78.5 us | 2.37x [2.37 – 2.37] | 1.35 ms | 40.9x [40.9 – 41.0] |
| 1,048,576 | 229.7 us | 318.3 us | 1.39x [1.37 – 1.40] | 8.79 ms | 38.2x [37.9 – 38.7] |
| 4,194,304 | 1.62 ms | 1.92 ms | 1.18x [1.17 – 1.26] | 37.47 ms | 23.1x [23.0 – 23.4] |
| 16,777,216 | 5.91 ms | 7.63 ms | 1.29x [1.29 – 1.30] | 94.02 ms | 15.9x [15.9 – 16.0] |
| overall |  |  | 1.94x [1.94 – 1.95] |  | 82.1x [82.0 – 82.3] |

`wide_values`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 5.0 ns | 5.7 ns | 1.14x [1.13 – 1.14] | 7.9 us | 1588x [1587 – 1590] |
| 64 | 19.2 ns | 20.8 ns | 1.09x [1.09 – 1.09] | 7.9 us | 413x [413 – 413] |
| 256 | 73.0 ns | 80.3 ns | 1.10x [1.10 – 1.10] | 8.1 us | 110x [110 – 110] |
| 1,024 | 292.2 ns | 310.6 ns | 1.06x [1.06 – 1.06] | 9.4 us | 32.1x [32.1 – 32.1] |
| 4,096 | 1.2 us | 1.2 us | 1.06x [1.06 – 1.06] | 12.0 us | 10.4x [10.4 – 10.4] |
| 16,384 | 4.6 us | 4.9 us | 1.06x [1.06 – 1.06] | 22.3 us | 4.82x [4.82 – 4.82] |
| 65,536 | 18.5 us | 19.6 us | 1.06x [1.06 – 1.06] | 62.4 us | 3.37x [3.37 – 3.37] |
| 262,144 | 74.0 us | 78.4 us | 1.06x [1.06 – 1.06] | 255.2 us | 3.45x [3.45 – 3.45] |
| 1,048,576 | 297.5 us | 313.2 us | 1.05x [1.05 – 1.05] | 1.13 ms | 3.79x [3.79 – 3.79] |
| 4,194,304 | 1.65 ms | 1.84 ms | 1.12x [1.12 – 1.12] | 10.48 ms | 6.36x [6.33 – 6.45] |
| 16,777,216 | 6.82 ms | 7.58 ms | 1.11x [1.11 – 1.12] | 47.76 ms | 7.00x [6.98 – 7.03] |
| overall |  |  | 1.08x [1.08 – 1.08] |  | 20.2x [20.2 – 20.2] |

All distributions: 1.45x [1.45 – 1.45] over naive C, 40.7x [40.7 – 40.8] over NumPy

With the output buffer 16 bytes behind its input (`mostly_misses`; "apart" is the same kernel with the output clear of the input):

`mostly_misses`:

| n | kernel | apart | speedup [95%] | naive C | speedup [95%] |
|---|---|---|---|---|---|
| 4,096 | 473.6 ns | 505.6 ns | 1.07x [1.07 – 1.07] | 1.2 us | 2.59x [2.59 – 2.59] |
| 16,384 | 2.0 us | 2.0 us | 1.00x [1.00 – 1.00] | 4.9 us | 2.43x [2.43 – 2.43] |
| 65,536 | 8.1 us | 8.1 us | 1.00x [1.00 – 1.00] | 19.6 us | 2.43x [2.43 – 2.43] |
| 262,144 | 31.0 us | 30.8 us | 0.99x [0.99 – 0.99] | 78.4 us | 2.53x [2.53 – 2.53] |
| 1,048,576 | 153.4 us | 147.0 us | 0.96x [0.95 – 0.97] | 320.7 us | 2.09x [2.08 – 2.11] |
| 4,194,304 | 1.29 ms | 1.45 ms | 1.12x [1.11 – 1.14] | 1.81 ms | 1.40x [1.39 – 1.42] |
| 16,777,216 | 5.70 ms | 5.77 ms | 1.01x [1.00 – 1.02] n.s. | 7.56 ms | 1.33x [1.32 – 1.33] |
| overall |  |  | 1.02x [1.02 – 1.02] |  | 2.05x [2.04 – 2.05] |

What holds it:

- **Narrow path, up to 65,536:** about thirty vector instructions per sixteen elements on three ports: four loads, three ORs and a `vptest` for the window, three packs, eight compares and seven ORs, four to turn masks into bytes.
- **Narrow path, from 1,048,576:** memory. 9 bytes per element move, and the kernel is 1.2x to 1.4x over the naive C.
- **Wide path:** eight 64-bit compares per four elements, on the two ports that take them: four cycles per four elements before anything else. The compiler's loop is at the same limit.

The candidates the design lists for this kernel (`docs/x86-64-design.md`):

| Candidate | Outcome |
|---|---|
| `vpshufb` is the same nibble lookup as the ARM64 `tbl` | Not tried. With sixteen 16-bit keys in one register, eight compares and seven ORs do the membership test in fifteen instructions; four nibble extractions, four lookups, three ANDs and a compare are as many. The ARM64 kernel needs the tables because its registers hold eight keys |
| `vpackssdw` narrows with saturation, within each 128-bit half | Used without the saturation, as `vpackusdw` on blocks already known to be in the window: 64-bit to 16-bit saturation takes two more instructions per vector here than one test per block does. The packs' order within the halves is undone once, on the sixteen result bytes |
| `vpcmpeqq` compares four elements at once | Kept for the general case (attempt 2), where it ties the compiler: 1.08x |
| Whether the compiler already emits it | It does |
| Masked loads and stores for leftover elements | Not needed: the leftover is one more block that overlaps the one before |
| Non-temporal stores, alignment of data | Not tried: the output is one byte per element |
| An output just behind the input | No cost in the judge's close table |
| Instructions slow on other x86-64 processors | None used |

**What differs from the ARM64 kernel for a caller.** The narrow path here needs every test value between 0 and 32,767. The ARM64 kernel also takes any eight values within 32,767 of the first one, wherever they are. A set of negative values, or of large values close together (years as nanoseconds, identifiers from a high range), gets the wide path here: correct, and about as fast as the naive C.

Left untried: a window that is not at zero (attempt 1 had one and placed it badly; centred on the test values it would serve the sets above, at four more instructions per block); a second narrow path with 8-bit keys for test values within 255 of each other.

Found here and worth knowing on ARM64: nothing that carries over.
