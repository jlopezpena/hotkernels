# isin_i64 attempts

This log holds two architectures. ARM64 (Apple M1 Pro) comes first. x86-64 (Intel Core i5-14600KF) follows under [x86-64](#x86-64), with its own numbering.

All timings were taken with `--unlocked` (the spec was not on `main` yet) while
other agents were timing other kernels on the same machine, so intervals are
wider than usual.

## 1. Scalar baseline

- **Idea:** simplest correct version: the 8 test values in registers, one element at a time, `cmp` followed by seven `ccmp`, `cset`, `strb`.
- **check:** pass
- **compare:** n/a (first version). `run`: 0.54x [0.54 – 0.54] vs naive C, the same at every size.
- **Kept:** yes
- **Notes:** 1.2 ns per element. Naive C runs at 0.64 ns per element: clang vectorises it, 16 elements per iteration, with 64 `cmeq`, 56 `uzp1`, 7 `orr` and an `and` (128 vector instructions per 16 elements).

## 2. NEON, 16 elements per iteration, compare and OR at 64 bits

- **Idea:** broadcast the 8 test values (`ld1r`); per vector of two elements, 8 `cmeq` and a 7-`orr` tree; narrow the eight lane-mask vectors to 16 bytes with `uzp1` (4 + 2 + 1), `ushr #7` to get 0/1, one 16-byte store. Leftover by an overlapping final block; n < 16 stays scalar.
- **check:** pass
- **compare:** significantly faster, 1.84x [1.84 – 1.85]
- **Kept:** yes
- **Notes:** 0.644 ns per element, which is exactly what naive C does (0.64): 128 vector instructions per 16 elements at four per cycle. Doing the same work as the compiler gives the same speed; a win needs fewer instructions per element.

## 3. Narrow the elements to 16-bit keys when the test values are close together

- **Idea:** if all 8 test values lie within 32,767 of `test[0]`, subtract `base = test[0] - 0x7FFF` from every element and narrow 64 to 32 to 16 bits with unsigned saturation (`uqxtn`, `uqxtn2`). Values below 0xFFFF stay exact, everything else becomes 0xFFFF, and no test key is 0xFFFF, so membership is exactly "16-bit key equals a test key". The 8 `cmeq` + 7 `orr` then run on vectors of eight keys instead of two elements: 52 vector instructions per 16 elements (8 `sub`, 12 narrowing, 30 compare/OR, 2 to finish) instead of 128. Any other test values take the 64-bit loop of attempt 2, chosen once per call by narrowing the test values the same way and looking for a 0xFFFF.
- **check:** pass (plus 2,000,000 random cases in a private fuzzer: test values at spreads of exactly 32,767 and 32,768, near the int64 wrap, with duplicates, and elements just outside the key range: no failure)
- **compare:** significantly faster, 2.30x [2.29 – 2.30] (NOISY MACHINE warning on this run; every size from 256 up was 2.37x–2.43x, n = 16 1.60x, n = 64 2.14x)
- **Kept:** yes
- **Notes:** 0.265 ns per element. The speedup exists only for test values that fit a 16-bit window; unrelated 64-bit test values run at the speed of attempt 2, which is the speed of naive C. The benchmark's only distribution draws test values from [0, 1000), so it always takes the narrow path.

## 4. Skip the subtraction when the test values are already below 0xFFFF

- **Idea:** try base 0 first: if every test value, read as unsigned, is below 0xFFFF, the elements can be narrowed as they are and the 8 `sub` per 16 elements disappear (44 vector instructions instead of 52). The biased loop of attempt 3 stays for other close-together sets (negative values, large values), the 64-bit loop for the rest.
- **check:** pass (private fuzzer extended with test values in [0, 66000) and around 65,534: 2,000,000 cases, no failure)
- **compare:** significantly faster, 1.17x [1.17 – 1.17]; 1.32x at 16, 1.14x–1.16x from 256 up
- **Kept:** yes
- **Notes:** 0.229 ns per element, 11.8 cycles per 16 elements: still four vector instructions per cycle, so the count is what matters.

## 5. Nibble tables instead of 8 compares, for n >= 96

- **Idea:** match the 16-bit keys with table lookups. Split the 16 keys into low bytes and high bytes (`uzp1`, `uzp2`), each byte into two nibbles (`and`, `ushr`), and look each nibble up with `tbl` in a 16-entry table for its position; the table byte has one bit per test key, set when that test key has the same nibble there. AND the four lookups: bit j survives exactly when the key equals test key j, so `umin` with 1 gives the answer. 14 instructions for 16 keys instead of 32, so 26 per 16 elements with the narrowing (34 with a base). The tables are built without memory: for each nibble position and each test key, `dup` its nibble, `cmeq` against 0..15, and `sli` the result in at bit j (98 instructions in all), so the tables are only used from n = 96; below that the compare loops of attempt 4 run.
- **check:** pass (private fuzzer, 2,000,000 cases with the threshold lowered to 16 so the table path sees every size: no failure)
- **compare:** significantly faster, 1.50x [1.49 – 1.50]: 1.37x at 256, 1.60x–1.73x from 1,024 up, unchanged at 16 and 64
- **Kept:** yes
- **Notes:** 0.131 ns per element in the table loop, 6.7 cycles per 16 elements. Building the tables takes about 8.7 ns; measured with the threshold at 16, n = 16 took 12.1 ns against 5.4 and n = 64 17.6 against 16.0, and the break-even is near 90 elements.

## 6. `ld4` to split the elements into 16-bit pieces while loading

- **Idea:** in the base-0 table loop, `ld4 {v0.8h-v3.8h}` loads eight elements already split into their four 16-bit pieces, so the low piece is the key and the saturating narrowing (12 instructions per 16 elements) is replaced by OR-ing the three upper pieces and masking the result (8 instructions).
- **check:** not run through the judge (private fuzzer: no failure)
- **compare:** not run through the judge. Private harness: 0.196 ns per element at 65,536 against 0.131 for the committed loop.
- **Kept:** no
- **Notes:** 50% slower. `ld4` costs far more on this core than the two loads and the narrowing it replaces.

## 7. Three lookups when every test value is below 4096

- **Idea:** with base 0 and all test values below 4096, the high byte of every test key is below 16, so it can index a table directly: one `tbl` on the whole high byte replaces the `and`, `ushr`, two `tbl` and one `and` of its two nibbles (a key whose high byte is 16 or more finds nothing, which is right). 22 vector instructions per 16 elements instead of 26, and three tables to build instead of four, so this variant is used from n = 64 (four tables stay at 96).
- **check:** pass (private fuzzer with both thresholds at 16: 2,000,000 cases, no failure)
- **compare:** significantly faster, 1.13x [1.12 – 1.13]: 1.08x at 64, 1.17x–1.21x from 256 to 1M, 1.01x–1.02x at 4M and 16M, unchanged at 16
- **Kept:** yes
- **Notes:** 0.108–0.115 ns per element up to 1M. At 4M and 16M the loop now runs at 0.15–0.17 ns per element: reading `x` (8 bytes per element) has become the limit there. Only the base-0 loop has this variant; with a base of `test[0] - 0x7FFF` the keys sit near 0x7FFF and never qualify.

## 8. Software prefetch of `x` in the table loop

- **Idea:** from 4M elements the three-table loop is slower per element than in cache; `prfm` two lines of `x` ahead of the loads.
- **check:** not run (see notes)
- **compare:** not run through the judge. Private harness, ns per element at 65,536 / 1M / 4M / 16M: no prefetch 0.122 / 0.117 / 0.137 / 0.141; `pldl1keep` at 512 bytes 0.127 / 0.114 / 0.136 / 0.141; at 1,024 bytes 0.132 / 0.109 / 0.137 / 0.142; at 2,048 bytes 0.121 / 0.111 / 0.172 / 0.186; at 4,096 bytes 0.121 / 0.113 / 0.380 / 0.537; `pldl2keep` at 4,096 bytes 0.124 / 0.109 / 0.136 / 0.143.
- **Kept:** no
- **Notes:** nothing at 4M and 16M, where it was meant to help, and the differences at 1M are within the spread between runs on this shared machine. Long distances are much worse.

## 9. Build each table with a tree of inserts instead of a chain

- **Idea:** the table build inserts the 8 per-key vectors one after another with `sli`, a chain of seven dependent instructions per table. Pair them instead (keys 0,1 / 2,3 / 4,5 / 6,7, then pairs, then halves): same instruction count, depth three, to shorten the set-up that dominates n = 64 and n = 256.
- **check:** pass (private fuzzer at both thresholds: no failure)
- **compare:** no significant difference, 1.00x [1.00 – 1.00]; n = 64 1.00x, n = 256 0.99x
- **Kept:** no
- **Notes:** the build is limited by how many vector instructions run per cycle (about 70 for three tables), not by the length of the chain.

## Which path runs for which test values, and how fast each is

Every path gives the exact answer for any 8 int64 test values; the choice only changes the speed. It is made once per call by narrowing the test values the same way as the elements and looking for a saturated key, so a set that does not fit simply falls through to the next path, ending at the 64-bit loop, which has no conditions. The judge's check uses arbitrary test values (it mostly exercises the 64-bit loop) and a private fuzzer covered the boundaries of each condition (spreads of exactly 32,767 and 32,768, values at 65,534 and 65,535, the int64 wrap, duplicates), with the table thresholds lowered so the table loops saw every size.

The judge times one distribution, `mostly_misses`, whose test values are drawn from [0, 1000). It therefore only ever times the first row below. The other rows were timed in a private harness against the same naive C build (ns per element at n = 65,536; ratio to naive C at 65,536 / 64 / 16):

| Test values | Path | ns per element | vs naive C (65,536 / 64 / 16) |
|---|---|---|---|
| all in [0, 4096) | base 0, three tables (compare loop below n = 64) | 0.112 | 5.8x / 2.9x / 2.2x |
| all in [0, 65535), not all below 4096 | base 0, four tables (compare loop below n = 96) | 0.134 | 4.9x / 2.6x / 2.1x |
| all within 32,767 of `test[0]`, e.g. negative or large values close together | base `test[0] - 0x7FFF`, four tables (compare loop below n = 96) | 0.172–0.179 | 3.7x / 2.1x / 1.6x |
| anything else, e.g. unrelated 64-bit values | 64-bit compares | 0.666 | 0.99x / 0.97x / 0.87x |

So for test values that do not fit a 16-bit window the kernel is no faster than naive C, and at n = 16 it is about 13% slower, because it has tried and rejected the two narrow set-ups before starting. A set that fits a 16-bit window but is not within 32,767 of its first value (for example {0, 60000} shifted negative) also takes the 64-bit loop.

## Final result

`uv run judge run isin_i64 --unlocked`, check PASS (17,124 cases): 4.46x [4.45 – 4.49] over naive C, 108x [108 – 109] over NumPy. Per size vs naive C: 16: 2.12x, 64: 2.83x, 256: 4.56x, 1,024: 5.39x, 4,096: 5.66x, 16,384: 5.75x, 65,536: 5.74x, 262,144: 5.69x, 1M: 5.36x, 4M: 4.10x, 16M: 4.07x.

## Not tried, and why

- **Choosing the base as the smallest test value** instead of `test[0] - 0x7FFF`. It would accept any set that fits a 16-bit window and let close-together sets away from zero use the three-table loop. The judge's distribution never takes the biased path, so `compare` could not have shown a gain.
- **A 32-bit key path** for sets that fit a 32-bit window but not a 16-bit one (8 compares on vectors of four keys, about 80 instructions per 16 elements against 128). Same reason.
- **`eor3` instead of the OR tree** in the compare loops (four instructions instead of seven per vector). It needs the test keys arranged so that no value appears an even number of times among seven of them; the check costs more than it saves at n = 16, the only timed size that still uses a compare loop.
- **A cheaper table build.** Several formulations were worked out on paper (bit-matrix transpose of one-hot rows, building two entries per compare, 64-bit variable shifts); all came to about 23 instructions per table, the same as the current one.

# x86-64

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core), Linux, locked, pinned to one performance core by the judge.

What the compiler does with `naive.c` (`clang -O3 -march=native`): it vectorises. Four elements per register, eight `vpcmpeqq` against the broadcast test values, an OR tree, and a narrowing of the masks to bytes: 0.30 ns per element, 1.6 cycles, at every size up to 1,048,576 and for any values. NumPy takes 8.4 ns per element at 1,048,576 on small values and 1.1 ns on values spread over the int64 range.

So the design's question for the general case, whether the compiler already emits the 64-bit compare, is answered: it does.

## x86-64 1. Narrow keys in a window around the test values; 64-bit compares from the stack otherwise

- **Idea:** sixteen elements per block. When the test values lie within 32,766 of each other, the smallest is subtracted from every element; if all sixteen differences of a block are below 2^15 they are narrowed to 16-bit keys with three `vpackusdw` and compared with eight `vpcmpeqw`, and otherwise the block is compared at 64 bits against test values kept broadcast on the stack.
- **check:** fail, then pass. The failure was a register: the index of the block was also the scratch register of the code that builds the keys, so the first load was at a garbage address. Found with a differential test outside the judge.
- **compare:** not run (never committed). `run`: **0.77x against the naive C**, on both distributions.
- **Kept:** no
- **Notes:** two mistakes of design, not of detail. **(1) The window started at the smallest test value.** The judge's `mostly_misses` data is uniform from 0 to 999 with test values from the same range, so a tenth of the elements were below the window and nearly every block of sixteen held one: every block took the wide path after paying for the narrow attempt. **(2) The wide path's compares had memory operands**, eight loads per four elements where the processor does two 32-byte loads per cycle; 0.41 ns per element against the compiler's 0.30.

## x86-64 2. Keys for test values from 0 to 32,767; 64-bit compares from registers otherwise

- **Idea:** the window is 0 to 32,767, fixed, and a call uses the narrow path when every test value is in it: one OR of the eight values and a shift decide. Nothing is subtracted. A call with any other test values keeps them broadcast in eight registers and compares at 64 bits. A block of a narrow call that holds an element outside the window (it cannot be a member, and it cannot be narrowed) is compared at 64 bits with the test values broadcast from memory one at a time, which is slow and needs no register and no stack.
- **check:** pass. Differential test outside the judge against the reference loop: 400,000 arrays of up to 600 elements, test values in windows of 3 to 65,535 and unrestricted, at small and at random bases, with repeats as the binding pads them, and elements drawn from the test values, from just outside the window, from test values plus a high bit, and at random. No mismatch.
- **compare:** not run (first committed version). `run`: 1.45x over naive C overall (`mostly_misses` 1.94x, `wide_values` 1.08x), 40.7x over NumPy.
- **Kept:** yes
- **Notes:** the narrow path takes 0.124 ns per element up to 65,536, 0.64 cycles, against 1.6 for the compiler's loop: 2.4x. The wide path is the compiler's loop written by hand and gains what a serial OR and a cheaper narrowing to bytes give: 1.05x to 1.14x.

## x86-64: where it stands

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
- **Narrow path, from 1,048,576:** memory. 9 bytes per element move, and the kernel is 1.3x to 1.4x over the naive C.
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
