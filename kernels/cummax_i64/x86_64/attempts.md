# cummax_i64 attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, pinned to one performance core by the judge. The runs are `--unlocked`: `spec.py` and `naive.c` are new in the same pull request and not on `main` yet. Neither was changed after the first line of assembly was written. Other judges were running on other cores of the machine during these runs, so the sizes from 1,048,576 up are noisy.

What the compiler does with `naive.c` (`clang -O3 -march=native`, clang 21.1.8): not a conditional move but a compare and a branch per element, unrolled eight times and laid out twice, once for "the last element was a new maximum" and once for "it was not", so that neither steady state has a taken branch. The maximum is then no chain at all: 0.15 ns per element, 0.8 of a cycle, from 64 to 65,536 elements, on random data and on rising data alike. NumPy takes 1.5 ns per element.

**That figure depends on the branches being predicted.** The judge times the same array again and again, and so does a micro-benchmark; this processor's predictor learns a repeated array of 16,384 elements nearly whole. In a micro-benchmark outside the judge, with a new maximum at a random half of the elements, the naive C takes 0.35 ns per element at 16,384 elements and 2.55 at 2,000,000, where the array is too long to learn. A walk with an upward drift: 0.20 and 1.2 to 1.8.

**What a vector can do.** AVX2 has no 64-bit integer maximum. One costs a `vpcmpgtq` (one port on this processor, three cycles) and a blend (three operations either as `vblendvpd` or as and, and-not, or): four operations on three ports. A running maximum in the register needs three per four elements, a cycle per element: slower than the naive C before anything is stored.

## 1. Baseline: blocks of 16 that are flat, rising or neither

- **Idea:** do not compute the maximum where one compare shows there is nothing to compute. With M the maximum so far, a block of 16 elements is *flat* if none is above M (one `vpcmpgtq` per vector; the block is M 16 times), *rising* if none is below the element before it and the element before the block is M (one `vpcmpgtq` per vector against the same load one element back; the block is a copy), and otherwise done the general way: four elements at a time in the integer registers, the maxima of x0, x0..x1, x0..x2, x0..x3 with compare and conditional move, then each against M, so the chain from group to group is one compare and one move and there is no branch. The flat and rising loops run until a block fails their test. After a general block the next is tried flat if M did not change, rising if the block's last element is M, and done the general way untested otherwise.
- **check:** pass
- **compare:** n/a (first version). `run`: 1.10x [1.10 – 1.11] over naive C on `random`, 1.08x [1.08 – 1.08] on `rising`, 10.9x over NumPy on both.
- **Kept:** yes
- **Notes:** `random`: 1.40x at 16, 0.83x at 64 (the first blocks of random data hold new maxima and go the general way), 1.21x to 1.25x from 1,024 to 16,384, 1.09x at 65,536, 1.02x to 1.04x above. `rising`: 1.14x to 1.21x up to 256, 1.07x to 1.09x to 65,536, 1.00x to 1.02x above. In a micro-benchmark with both buffers aligned the loops take 0.067 ns per element at 1,024 (flat) and 0.075 (rising), and 0.104 and 0.112 with the output 16 bytes off: the judge's placement. The general path alone, on 1 0 2 1 3 2 ..., takes 0.21 ns per element, 1.1 cycles, against the naive C's 0.14 on a pattern its branches predict; on the random half above it takes 0.24 at 16,384 and 0.54 at 2,000,000 against the naive C's 0.35 and 2.55.

## 2. Bring the output to a multiple of 32 bytes first

- **Idea:** up to three elements one at a time until `out` is a multiple of 32, from 19 elements up, so that no store of a block straddles two cache lines. An `out` that is not a multiple of 8 is left as it is.
- **check:** pass
- **compare:** significantly faster, 1.030x [1.025 – 1.035]
- **Kept:** yes
- **Notes:** `random`: 1.17x at 1,024, 1.16x at 4,096, 1.14x at 65,536, 1.00x at 16,384; `rising`: 1.09x at 4,096, 1.07x at 16,384; 0.98x to 0.99x at 16 and 64, which pay two more compares; nothing from 262,144 up. The judge's 128.6 ns at 1,024 random elements is still twice the micro-benchmark's 65 to 80: below 4,096 elements the judge times a pool of arrays in turn (four of 1,024, 64 of 64) and not one array over and over, so a new maximum in the flat loop costs its mispredicted exit, and after it two blocks done the general way.

## 3. After a new maximum in the flat loop, straight back to the flat loop

- **Idea:** a block that fails the flat test is done the general way and then, under attempt 1's rule, M has changed, so the next block is done the general way too before the flat loop is tried again. In data whose maximum rarely changes the next block is flat. So: after such a block go back to the flat test at once, and leave the blocks to the general loop only if that test fails as well.
- **check:** pass
- **compare:** significantly faster, 1.004x [1.003 – 1.005]
- **Kept:** yes
- **Notes:** `random`: 1.04x at 256, 1.01x at 64 and 1,024, nothing elsewhere; `rising` unchanged. Far less than expected, and it showed what attempt 2's note got wrong: the judge's pool at 256 and 1,024 elements is 4,096 elements in all, 32 KB of input and 32 KB of output, which is more than the 48 KB first-level cache. The judge never times this kernel inside the first-level cache from 64 elements up; its 0.12 ns per element at 1,024 is the second-level cache's speed, the same as at 16,384, and not the cost of new maxima.
