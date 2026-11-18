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
