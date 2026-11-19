# where_f64 attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, pinned to one performance core by the judge. The runs are `--unlocked`: `spec.py` and `naive.c` are new in the same pull request and not on `main` yet. Neither was changed after the first line of assembly was written.

What the compiler does with `naive.c` (`clang -O3 -march=native`): it vectorises the loop four elements wide. 0.23 ns per element at 1,024, 0.21 at 4,096, 0.24 at 65,536, 0.33 at 262,144 and 0.68 at 1,048,576. NumPy branches on every element: 2.7 ns per element at 65,536 with half of the condition true, 0.54 with 95% true.

Every element is 17 bytes read and 8 written, so beyond the first-level cache the kernel moves memory and its arithmetic is a small part of the time. The kernel does the same work whatever the condition holds, so the two timing distributions differ only in where their buffers happened to be placed.

## 1. Baseline: widen four bytes of c, select with and, and-not, or

- **Idea:** four elements per step. `vpmovsxbq` widens four bytes of c to four 64-bit lanes, `vpcmpeqq` against zero makes the lane mask, all ones where c is false, and the output is (mask and b) or (a and not mask), with both loads folded into the logical instructions. The elements are only ever loaded, masked and stored. A byte of c is true when it is not zero. The last 1 to 3 elements take a scalar loop with a conditional move.
- **check:** pass
- **compare:** n/a (first version). `run`: 1.14x [1.14 – 1.14] over naive C overall (half_true 1.17x, mostly_true 1.11x), 6.86x over NumPy.
- **Kept:** yes
- **Notes:** half_true: 1.07x at 16, 1.27x to 1.29x from 64 to 1,024 (0.18 ns per element), 1.43x at 4,096, 1.15x at 16,384, 1.24x at 65,536, 1.00x at 262,144, 1.05x to 1.09x from 1,048,576 up. mostly_true has 4,096 at 1.10x and 262,144 and 1,048,576 at 0.96x: the same code on other buffers. `vblendvpd` was not used for the select: measured on this core in a micro-benchmark outside the judge, a group with it takes the same 1.54 cycles as the three logical instructions (constant mask, buffers at multiples of 32).

## 2. Blocks of 16: one load of c, lane masks by `vpshufb`, the leftover by an overlapping block

- **Idea:** found with a micro-benchmark outside the judge, 1,024 elements in the first-level cache, buffers at multiples of 32: attempt 1 takes 2.4 cycles per group of four, the select alone with a constant mask 1.55, and the widening and the compare alone 1.0, because `vpmovsxbq` has one port. So the masks are made another way. `vbroadcasti128` loads 16 bytes of c into both halves of a register, one `vpcmpeqb` against zero makes them 0xff where c is false, and one `vpshufb` per group spreads four of those bytes over the four 64-bit lanes: 1.25 vector instructions and a quarter of a load per group, where attempt 1 has two and a whole load. The elements left after the whole blocks are done by one more block that ends exactly at the end of the buffers, in place of a scalar loop; out never overlaps an input, so writing some elements twice is harmless. Under 16 elements: attempt 1's groups, the last one ending at the end of the buffers; under 4, the scalar loop. Pointers are advanced and addressed with a displacement, where attempt 1 indexes.
- **check:** pass
- **compare:** significantly faster, 1.040x [1.034 – 1.046]
- **Kept:** yes
- **Notes:** 1.09x to 1.13x at 16 (the overlapping group in place of the scalar loop), 0.99x at 64, 1.05x to 1.09x from 256 to 4,096, 1.09x to 1.11x at 1,048,576, nothing at 4,194,304 and 16,777,216. In the micro-benchmark the block takes 2.02 cycles per group against 2.4, and 2.28 against 2.74 with a and b 16 bytes off a multiple of 32. Blocks of 32 were no faster there than blocks of 16 (2.04), and nor was computing the four groups before storing any (1.94 against 2.02 with everything aligned, 2.29 against 2.28 with a and b off). `vblendvpd` in place of the three logical instructions: 1.93 against 2.02 aligned, 2.28 against 2.28 off; not taken, for nothing.

## 3. Stores past the caches from 24 MB of output

- **Idea:** from 3,145,728 elements the blocks store with `vmovntdq`, which does not read the line it is about to overwrite, followed by one `sfence`. Of the 33 bytes a block moves per element (17 read, 8 read again before they are overwritten, 8 written), that takes the second 8 away. These stores need out at a multiple of 32: the first block is stored the ordinary way and the blocks after it start up to three elements in, where out is at such a multiple; the last, overlapping block is stored the ordinary way too, after the `sfence`. An out that is not a multiple of 8 keeps the ordinary stores. The threshold is this processor's third-level cache, as in `cumsum_i64` and `cumsum_bool`: below it an output written with ordinary stores is left in a cache for the caller.
- **check:** pass. The judge's correctness cases do not reach this path, so it was tested outside the judge against a scalar loop that copies each element's eight bytes: every size from 0 to 299 and 18 sizes from 33 below the threshold to 4,194,304; three placements of c, four of a, three of b and seven of out, among them an a, a b and two outs that are not multiples of 8; 64 guard bytes before and after out; once with every byte of c 0 or 1 and once with the true bytes any non-zero value (the large sizes then with one placement of a and b). 156,114 runs, no failure. The same program reports 1,638 and 2,730 failures for two kernels broken on purpose (the blocks after the first not advanced in c; one store past the end).
- **compare:** significantly faster, 1.051x [1.046 – 1.056]; a second run 1.051x [1.049 – 1.053]
- **Kept:** yes
- **Notes:** 1.31x at 4,194,304 (2.97 ms to 2.27 ms) and at 16,777,216 (12.25 ms to 9.32 ms) with half_true, 1.33x and 1.28x with mostly_true; 1.28x to 1.34x in the second run. Every other size 0.99x to 1.01x.

## 4. Bring the output to a multiple of 32 bytes at every size from 64

- **Idea:** a 32-byte store that straddles two cache lines costs more than one that does not, and with out 16 bytes off a multiple of 32, as memory from the C allocator often is, every second store does. In a micro-benchmark outside the judge at 4,096 and 16,384 elements, in the second-level cache: 0.150 ns per element with all three of a, b and out at multiples of 32, 0.176 with out alone 16 bytes off, 0.176 with a and b off and out not, 0.184 to 0.195 with all three off. So the first block is done where out is, and the blocks after it start up to three elements in, where out is a multiple of 32, as attempt 3 already did for its own stores. No branch and no scalar loop: the elements between are written twice. Where a and b sit as out does, which is the usual case for arrays that came from the same allocator, this aligns all three.
- **check:** pass. The program of attempt 3 again: 156,114 runs, no failure.
- **compare:** significantly faster, 1.048x [1.042 – 1.054]
- **Kept:** yes
- **Notes:** half_true 1.07x at 256, 1.11x at 1,024, 1.12x at 4,096, 1.27x at 16,384, 1.18x at 65,536, 1.05x at 262,144, nothing from 1,048,576, where the judge's output already is at a multiple of 32. mostly_true the same except 16,384 at 1.00x, where its output already was. **0.90x at 64** in both: five blocks where there were four.

## 5. A group of four in place of the first block, and alignment from 32 elements

- **Idea:** bringing out to a multiple of 32 takes at most three elements, so the unaligned first step need not be a block of 16: one group of four with its mask from `vpmovsxbq`, six instructions against 22. With the first step that cheap the alignment starts at 32 elements, not 64.
- **check:** pass. The program of attempt 3 again: 156,114 runs, no failure.
- **compare:** significantly faster, 1.017x [1.015 – 1.020]
- **Kept:** yes
- **Notes:** 1.19x to 1.20x at 64 (13.9 ns to 11.6 ns, and below the 12.5 ns it took before attempt 4), 1.02x at 256, 0.99x at 4,096 and 16,384, nothing elsewhere.

## 6. Select by storing twice: the elements of a, then a masked store of b (`vmaskmovpd`)

- **Idea:** in the micro-benchmark a group selected with and, and-not and or takes 2.0 cycles in the first-level cache whatever is done to it, and the same group done as two loads, a store of a's four elements and a `vmaskmovpd` store of b's over them where c is false takes 1.66 to 1.79; at 16,384 elements 0.135 ns per element against 0.150. The lane mask's top bit is all the masked store reads. The blocks stored past the caches keep the logical select, since a masked store cannot go past them.
- **check:** pass. The program of attempt 3: 156,114 runs, no failure.
- **compare:** significantly faster, 1.005x [1.001 – 1.010]; a second run 1.008x [1.003 – 1.013]
- **Kept:** no. The judge's verdict allows it, twice, by half a percent. Not kept for what comes with it: `vmaskmovpd` to memory is one of the instructions that are fast on this processor and slow on others (many times slower on AMD before Zen 4, by the published instruction tables; not measured here), the time comes to depend on the condition, and two sizes lose more than the others gain. Reported for a person to decide.
- **Notes:** both runs alike. 1.06x to 1.10x at 16, 1.05x at 256, 1.02x to 1.03x at 1,024 and 4,096; **0.88x at 64 with half_true** (11.7 ns to 13.3 ns) and 0.97x with mostly_true, the first place the same code takes a different time on the two distributions; **0.96x at 16,384** in both; 0.97x to 0.99x at 1,048,576; nothing from 4,194,304, which this change does not touch. The micro-benchmark's gain beyond the first-level cache did not appear in the judge.

## 7. All four groups of a block selected before any is stored

- **Idea:** the loads of a block ahead of its stores: the four selects first, each into its own register, then the four stores, in the ordinary blocks and in those stored past the caches. In the micro-benchmark it was 0.156 ns per element against 0.175 at 4,096 with a and b 16 bytes off a multiple of 32, and nothing at 16,384.
- **check:** pass. The program of attempt 3: 156,114 runs, no failure.
- **compare:** no significant difference, 1.001x [0.998 – 1.004]
- **Kept:** no
- **Notes:** every size 0.98x to 1.02x.

## 8. Ask for the cache lines 512 bytes ahead

- **Idea:** from 256 to 65,536 elements the judge's data sits in the second-level cache, and there the kernel is as fast as the lines arrive. Six `prefetcht0` per block ask for the two lines of a, of b and of out that the block 512 bytes on will use. A first loop does that for as long as those lines are inside the buffers and a second, the loop as it was, does the last blocks without, so nothing outside the buffers is asked for and an array of under 80 elements never prefetches. Not in the blocks stored past the caches. In the micro-benchmark, with a and b 16 bytes off a multiple of 32: 0.160 to 0.145 ns per element at 4,096 and 0.180 to 0.150 at 65,536 with 512 bytes, the same with 1,024, less with 256 (0.153 and 0.185), and nothing that stands out of the noise at 262,144 and above.
- **check:** pass. The program of attempt 3: 156,114 runs, no failure.
- **compare:** significantly faster, 1.019x [1.013 – 1.026]
- **Kept:** yes
- **Notes:** 1.03x to 1.04x at 1,024, 1.06x to 1.07x at 4,096, 16,384 and 65,536, 1.01x at 256, nothing at 16 and 64 and from 262,144 up.

## 9. Stores past the caches from 24 MB of everything the call moves, not of output

- **Idea:** this kernel moves 33 bytes per element, of which the output is 8, so a call passes the 24 MB of the third-level cache at 762,600 elements, a quarter of attempt 3's threshold. `STREAM_MIN_N` lowered to that; nothing else changed.
- **check:** pass
- **compare:** significantly faster, 1.031x [1.029 – 1.034] (one run)
- **Kept:** no. Not mine to keep: the threshold of attempt 3 is the rule the other kernels follow (`cumsum_i64`, `cumsum_bool`: stores past the caches only where the output alone fills the third-level cache), and it decides what a caller gets. With ordinary stores at 1,048,576 elements most of the 8 MB of output is still in the third-level cache when the kernel returns; with these stores none of it is, and a caller that reads the output next pays for that outside anything the judge times. Reported for a person to decide.
- **Notes:** 1.46x at 1,048,576 with half_true (489.5 us to 336.0 us) and 1.36x with mostly_true (501.4 us to 368.0 us); every other size 0.99x to 1.00x. In the micro-benchmark, which calls the kernel again and again on the same buffers and so favours the ordinary stores, the same stores were also faster below that: 0.31 to 0.32 ns per element against 0.41 at 262,144, 0.30 to 0.32 against 0.40 to 0.44 at 524,288, 0.34 to 0.35 against 0.48 at 1,048,576, 0.47 to 0.49 against 0.65 to 0.85 at 2,097,152 (two runs each, other work on the machine).

## 10. No prefetch under 2,048 elements

- **Idea:** attempt 8's six prefetches are six more instructions in a block of 22, and where the four buffers already are in the first-level cache they fetch nothing. In the micro-benchmark, which calls the kernel on one set of buffers and so has 1,024 elements (34 KB) in that cache, attempt 8 made the kernel slower there: 119.0 ns against 111.5 with a and b 16 bytes off a multiple of 32, 105.3 against 95.3 with everything at one. So the prefetching loop is entered only from 2,048 elements, where the buffers cannot all be in the first-level cache.
- **check:** pass. The program of attempt 3: 156,114 runs, no failure.
- **compare:** significantly slower, 0.997x [0.996 – 0.998]
- **Kept:** no
- **Notes:** 0.96x to 0.97x at 1,024 and 0.99x to 1.00x at 256, nothing elsewhere; in the micro-benchmark 1,024 elements were back at 110.3 ns and 93.8 ns. The judge and the micro-benchmark time different things at that size. The judge cycles through a pool of inputs of 4,096 elements in all, which for this kernel is 135 KB of buffers and does not fit the 48 KB first-level cache, so at 1,024 elements and below it times data arriving from the second-level cache, where the prefetches help. Data already in the first-level cache, 81 to about 1,400 elements, is 7% to 10% slower with attempt 8 than without, and the judge has no size that shows it.

## Where it stands

Final `judge run` (unlocked): 1.38x [1.37 – 1.38] over naive C over both distributions, 8.18x over NumPy.

`half_true`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 3.5 ns | 4.9 ns | 1.38x [1.37 – 1.38] | 442.7 ns | 126x [125 – 126] |
| 64 | 11.8 ns | 16.3 ns | 1.38x [1.38 – 1.38] | 455.8 ns | 38.5x [38.5 – 38.5] |
| 256 | 41.7 ns | 61.5 ns | 1.47x [1.47 – 1.48] | 578.3 ns | 13.9x [13.9 – 13.9] |
| 1,024 | 148.1 ns | 237.1 ns | 1.60x [1.60 – 1.60] | 965.7 ns | 6.52x [6.51 – 6.53] |
| 4,096 | 546.5 ns | 878.3 ns | 1.61x [1.60 – 1.61] | 2.4 us | 4.35x [4.35 – 4.35] |
| 16,384 | 2.3 us | 3.7 us | 1.63x [1.62 – 1.63] | 33.5 us | 14.8x [14.8 – 14.9] |
| 65,536 | 9.9 us | 16.0 us | 1.62x [1.62 – 1.63] | 171.1 us | 17.3x [17.3 – 17.4] |
| 262,144 | 86.1 us | 85.7 us | 1.00x [0.99 – 1.00] | 695.9 us | 8.08x [8.07 – 8.15] |
| 1,048,576 | 540.8 us | 632.9 us | 1.17x [1.15 – 1.20] | 2.87 ms | 5.31x [5.21 – 5.37] |
| 4,194,304 | 2.46 ms | 3.35 ms | 1.36x [1.34 – 1.38] | 13.58 ms | 5.51x [5.46 – 5.61] |
| 16,777,216 | 9.40 ms | 13.12 ms | 1.40x [1.36 – 1.44] | 53.56 ms | 5.70x [5.59 – 5.75] |
| overall |  |  | 1.40x [1.40 – 1.41] |  | 11.9x [11.8 – 11.9] |

`mostly_true`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 3.4 ns | 4.8 ns | 1.41x [1.41 – 1.41] | 416.6 ns | 121x [121 – 121] |
| 64 | 11.5 ns | 15.9 ns | 1.38x [1.38 – 1.38] | 440.3 ns | 38.2x [38.1 – 38.2] |
| 256 | 41.6 ns | 58.2 ns | 1.40x [1.40 – 1.40] | 540.1 ns | 13.0x [13.0 – 13.0] |
| 1,024 | 148.3 ns | 217.8 ns | 1.47x [1.47 – 1.47] | 875.0 ns | 5.90x [5.89 – 5.91] |
| 4,096 | 575.3 ns | 871.7 ns | 1.52x [1.51 – 1.52] | 2.1 us | 3.57x [3.57 – 3.58] |
| 16,384 | 2.4 us | 3.4 us | 1.45x [1.45 – 1.45] | 7.0 us | 2.96x [2.95 – 2.97] |
| 65,536 | 10.5 us | 14.5 us | 1.37x [1.37 – 1.38] | 35.5 us | 3.37x [3.37 – 3.38] |
| 262,144 | 109.6 us | 105.2 us | 0.96x [0.96 – 0.96] | 166.4 us | 1.52x [1.51 – 1.52] |
| 1,048,576 | 606.6 us | 656.9 us | 1.08x [1.06 – 1.10] | 905.3 us | 1.49x [1.47 – 1.52] |
| 4,194,304 | 2.43 ms | 3.60 ms | 1.48x [1.45 – 1.53] | 6.21 ms | 2.55x [2.50 – 2.61] |
| 16,777,216 | 13.35 ms | 19.32 ms | 1.45x [1.40 – 1.51] | 33.79 ms | 2.53x [2.45 – 2.64] |
| overall |  |  | 1.35x [1.34 – 1.36] |  | 5.64x [5.62 – 5.67] |

Other judges were running on other cores during this run, so the sizes from 1,048,576 up carry their share of the third-level cache and of memory (16,777,216 took 9.4 ms in one table and 13.4 ms in the other, kernel and naive C alike).

The second table, output 16 bytes behind c within the page: 0.97x to 1.04x of the first at every size except 1,048,576, where the first table's own timing was the slow one (730.8 us against 560.1 us). The close placement costs this kernel nothing.

What holds it:

- **16 to 65,536: the second-level cache.** 0.13 to 0.18 ns per element from 64 to 65,536, 0.7 to 0.95 cycles, and 0.22 at 16. The judge's pool of small inputs is 135 KB for this kernel (attempt 10), so none of these sizes is timed with its data in the first-level cache. With it there, in the micro-benchmark at 1,024 elements, the kernel takes 0.10 to 0.12 ns per element (0.09 to 0.11 before attempt 8).
- **In the first-level cache: about 2 cycles per group of four, and not the ports.** Three logical instructions alone run at one group per cycle, two 32-byte loads alone at 0.72 cycles, both together at 1.38, and with the store 1.55; the lane masks bring it to 2.0. The cost of each part adds to the others though they use different ports, and two loads into general registers beside the three logical instructions add as well (1.18). Not explained; `perf` is not available on this machine.
- **262,144: the third-level cache.** The kernel and the naive C take the same time, 0.33 to 0.42 ns per element.
- **1,048,576: memory, with ordinary stores.** Attempt 9 is 1.36x to 1.46x there and was not kept.
- **4,194,304 and 16,777,216: memory.** 0.56 to 0.59 ns per element with the stores past the caches, leaving aside the one disturbed timing.

What the kernel requires of its caller:

- out does not overlap c, a or b. Elements at the start and at the end of the arrays are selected and written twice, so every input must still hold its values after out has been written.
- A byte of c is true when it is not zero: 2, 0x80 and 0xff select a as 1 does. The judge only sends 0 and 1; the other values were checked outside it (attempt 3).
- No buffer needs any alignment. An out that is not a multiple of 8 bytes is correct but is neither brought to a multiple of 32 nor stored past the caches.
- From 3,145,728 elements, with out at a multiple of 8 bytes, the output is in no cache when the kernel returns.

Instructions that are slow on some other x86-64 processors: none in the kept kernel (`vpshufb`, `vbroadcasti128`, `prefetcht0`, `vmovntdq`, `sfence`). Attempt 6's `vmaskmovpd` store is one.

Not tried:

- **The select of attempt 6 only where its stores do not straddle a line and the size gains.** It loses at 64 and at 16,384 and gains between; a size switch for an instruction that is not kept anyway was not worth an attempt.
- **Prefetches of a and b only, or at another distance, in the judge.** In the micro-benchmark a and b alone gave about two thirds of attempt 8's gain at 4,096 and 16,384 and less at 65,536, for four instructions in place of six; 1,024 bytes was level with 512 at 4,096 and 16,384.
- **Prefetches in the blocks stored past the caches.** In the micro-benchmark at 4,194,304, with prefetches of a and b 256 to 2,048 bytes ahead: 0.54 to 0.59 ns per element against 0.56 without, inside the noise of a shared machine.
- **Blocks of 32 with one mask word** (`vpmovmskb` and a table of lane masks, as `maskfill_f64` does): the table is one more 32-byte load per group, and loads are what this kernel has most of.
- **Bringing a or b to a multiple of 32 where out cannot be made to agree with them.** One store stream against two load streams: in the micro-benchmark out alone off cost the same as a and b both off (0.176 ns per element at 4,096 and 16,384).
- **AVX-512 masked moves:** outside the instruction level.
