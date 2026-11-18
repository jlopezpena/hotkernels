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
