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

## 4. Stores past the caches from 24 MB of output

- **Idea:** from 3,145,728 elements the flat and rising loops store with `vmovntdq`, which does not read the line it is about to overwrite, with an `sfence` wherever such a loop is left. It needs the output at a multiple of 32, which attempt 2 already arranges; an output that could not be brought there keeps the ordinary stores. The general blocks keep ordinary stores. The threshold is this processor's third-level cache, as in `cumsum_i64`: below it an output written with ordinary stores is left in a cache for the caller. Whether to store that way is decided once and kept below the stack pointer, since every register the kernel may use without saving is taken.
- **check:** pass. The judge's correctness cases do not reach this path, so it was tested outside the judge against the naive C: 173 sizes, every one from 0 to 150 and 16 from 3,145,727 to 4,194,304 around the threshold and the head lengths; nine kinds of data (random, rising with ties, constant, few values, 1 0 2 1 3 2, two walks, falling, and runs of random length that are flat, rising or noisy in turn, so that the paths change at every position); four placements of `x` and seven of `out`, two of them not multiples of 8; 16 guard bytes before and after `out`, and `x` compared with what it held before: 120,204 runs, no failure.
- **compare:** significantly faster, 1.067x [1.063 – 1.071]
- **Kept:** yes
- **Notes:** `random`: 1.51x at 4,194,304 (2.34 ms to 1.57 ms) and 1.49x at 16,777,216; `rising`: 1.46x and 1.42x. More than the 1.25x the same stores gave `cumsum_i64`, less than the 2.6x of `cumsum_bool`, whose input is an eighth of its output. Nothing between 1,024 and 1,048,576. 0.92x at 64 rising elements (10.5 ns to 11.4) and 0.98x to 0.99x at 16, 64 and 256: the test on the way into a loop. Two earlier forms of the same change were slower there and are not kept: with the mark in `rdx` and M before a general block in memory, 0.92x at 256 random elements; with the mark written as a byte over a 4-byte zero, the same as this one. All three gave 1.06x to 1.07x overall, decided by the two largest sizes, which were measured three times with the same result.

## Where it stands

Final `judge run` (unlocked; the judge reported a load average of 3.0, other judges on other cores): 1.21x [1.20 – 1.21] over naive C on `random` and 1.13x [1.12 – 1.14] on `rising`, 1.17x [1.16 – 1.17] over both; 11.4x over NumPy on each.

`random`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 4.5 ns | 5.9 ns | 1.30x [1.29 – 1.31] | 276.8 ns | 61.7x [61.65 – 61.72] |
| 64 | 15.1 ns | 12.3 ns | 0.82x [0.82 – 0.82] | 332.4 ns | 22.1x [22.04 – 22.06] |
| 256 | 35.5 ns | 40.3 ns | 1.13x [1.13 – 1.14] | 649.0 ns | 18.3x [18.25 – 18.27] |
| 1,024 | 126.7 ns | 160.3 ns | 1.27x [1.26 – 1.27] | 1.8 us | 14.0x [13.98 – 14.03] |
| 4,096 | 470.7 ns | 622.9 ns | 1.32x [1.32 – 1.32] | 6.7 us | 14.3x [14.31 – 14.32] |
| 16,384 | 1.9 us | 2.4 us | 1.27x [1.27 – 1.27] | 25.8 us | 13.6x [13.55 – 13.56] |
| 65,536 | 7.6 us | 9.4 us | 1.24x [1.24 – 1.24] | 100.9 us | 13.3x [13.32 – 13.34] |
| 262,144 | 72.5 us | 75.8 us | 1.05x [1.04 – 1.05] | 405.3 us | 5.59x [5.58 – 5.60] |
| 1,048,576 | 390.7 us | 397.8 us | 1.02x [0.98 – 1.05] n.s. | 1.64 ms | 4.20x [4.04 – 4.47] |
| 4,194,304 | 2.24 ms | 3.47 ms | 1.55x [1.53 – 1.57] | 9.19 ms | 4.10x [4.03 – 4.17] |
| 16,777,216 | 6.88 ms | 10.29 ms | 1.50x [1.45 – 1.53] | 33.63 ms | 4.89x [4.79 – 4.97] |
| overall |  |  | 1.21x [1.20 – 1.21] |  | 11.4x [11.35 – 11.47] |

`rising`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 4.5 ns | 4.4 ns | 1.00x [0.99 – 1.00] n.s. | 272.9 ns | 61.2x [61.05 – 61.32] |
| 64 | 11.1 ns | 12.4 ns | 1.12x [1.12 – 1.12] | 332.9 ns | 29.9x [29.92 – 29.97] |
| 256 | 37.4 ns | 41.1 ns | 1.10x [1.08 – 1.11] | 648.8 ns | 17.3x [17.32 – 17.35] |
| 1,024 | 137.8 ns | 149.6 ns | 1.09x [1.09 – 1.09] | 1.8 us | 12.8x [12.81 – 12.84] |
| 4,096 | 536.6 ns | 588.2 ns | 1.10x [1.09 – 1.10] | 6.7 us | 12.5x [12.44 – 12.49] |
| 16,384 | 2.1 us | 2.3 us | 1.10x [1.10 – 1.10] | 25.4 us | 12.0x [12.00 – 12.02] |
| 65,536 | 8.6 us | 9.3 us | 1.09x [1.09 – 1.09] | 101.5 us | 11.9x [11.85 – 11.87] |
| 262,144 | 75.5 us | 77.1 us | 1.02x [1.02 – 1.02] | 404.8 us | 5.36x [5.35 – 5.37] |
| 1,048,576 | 348.5 us | 357.3 us | 1.03x [1.00 – 1.06] | 1.64 ms | 4.72x [4.65 – 4.78] |
| 4,194,304 | 1.81 ms | 2.63 ms | 1.45x [1.40 – 1.49] | 8.70 ms | 4.80x [4.60 – 4.92] |
| 16,777,216 | 7.42 ms | 10.75 ms | 1.45x [1.28 – 1.55] | 34.56 ms | 4.66x [4.00 – 4.84] |
| overall |  |  | 1.13x [1.12 – 1.14] |  | 11.4x [11.21 – 11.41] |

With the output buffer 16 bytes behind its input ("apart" is the same kernel with the output clear of the input), `random`:

| n | kernel | apart | speedup [95%] | naive C | speedup [95%] |
|---|---|---|---|---|---|
| 4,096 | 483.8 ns | 498.9 ns | 1.03x [1.03 – 1.03] | 611.3 ns | 1.26x [1.26 – 1.26] |
| 16,384 | 2.0 us | 1.9 us | 0.93x [0.93 – 0.93] | 3.1 us | 1.53x [1.53 – 1.54] |
| 65,536 | 7.6 us | 7.6 us | 1.00x [1.00 – 1.00] n.s. | 9.4 us | 1.23x [1.23 – 1.24] |
| 262,144 | 60.0 us | 59.8 us | 1.00x [0.99 – 1.01] n.s. | 65.7 us | 1.10x [1.09 – 1.10] |
| 1,048,576 | 410.3 us | 568.8 us | 1.39x [1.36 – 1.41] | 1.32 ms | 3.22x [3.20 – 3.24] |
| 4,194,304 | 1.55 ms | 1.70 ms | 1.10x [1.08 – 1.11] | 2.45 ms | 1.58x [1.57 – 1.60] |
| 16,777,216 | 6.55 ms | 6.72 ms | 1.03x [1.01 – 1.04] | 9.85 ms | 1.51x [1.48 – 1.55] |
| overall |  |  | 1.06x [1.05 – 1.06] |  | 1.54x [1.53 – 1.54] |

No stall: a block's four loads come before its four stores. 0.93x at 16,384 is the one size where the close placement costs.

What the kernel requires of its caller: `out` does not overlap `x`. In the rising loop a block of `out` is stored after the same block of `x` is loaded, so `out` equal to `x` happens to work there, but the general path stores a group's elements after loading only that group and nothing was tested with overlap.

What holds it at each size:

- **16 and 64:** the general path, 1.1 cycles per element against the 0.8 of the naive C's predicted branches. Random data begins with new maxima, so its first blocks go that way: 0.82x at 64.
- **256 to 65,536:** the second-level cache. The judge's pool of arrays is 64 KB of input and output from 64 elements up, so no size is timed in the first-level cache, where the micro-benchmark has the flat loop at 0.064 ns per element and the rising one at 0.076 against the naive C's 0.15. In the second-level cache the flat loop takes 0.116 ns per element, which is what a copy takes there (`cumsum_i64`'s log: 0.115). The rising loop takes 0.116 when `x` and `out` sit alike within 32 bytes and 0.129 when they do not, which is the judge's placement: with the stores aligned, two of a block's four loads straddle a cache line, and so do two of the four loads one element back.
- **262,144 and 1,048,576:** the third-level cache, 1.02x to 1.05x.
- **4,194,304 and 16,777,216:** memory, with a third less traffic than the naive C: 1.45x to 1.55x.

**Data that is neither flat nor rising** is not in the judge's distributions. In the micro-benchmark, 16,384 elements, nanoseconds per element:

| data | kernel | naive C |
|---|---|---|
| 1 0 2 1 3 2 ..., which the naive C's branches predict | 0.20 | 0.14 |
| a walk with a small upward drift | 0.19 | 0.20 |
| a walk with a strong upward drift | 0.23 | 0.27 |
| a new maximum at a random half of the elements | 0.22 | 0.35 |
| the same at 2,000,000 elements, too long for the predictor to learn | 0.72 | 2.60 |
| a walk with a small upward drift, 2,000,000 elements | 0.75 | 1.32 |

The first row is the worst found: 0.7x of the naive C on a pattern with a short period. On data a predictor cannot learn the general path is 1.8 to 3.6 times faster than the naive C, which is why it has no branch per element. (The figures at 2,000,000 elements were taken with other judges running and moved between runs: 0.54 and 0.67 for the kernel in an earlier one.)

Not tried:

- **The rising test without the second set of loads.** Building the vector one element back in the register costs two shuffles per vector on the port `vpcmpgtq` needs: three cycles per vector where the second-level cache allows 2.5. Assembling the output vectors from loads aligned to `x` (one `vperm2i128` per vector) would halve the straddling loads only when `x` and `out` differ by 16 within 32 bytes.
- **A flat test after every general block,** or one chosen by whether the last group of four changed M. For random data the gain when the next block is flat and the loss when it is not are about equal by count; for data that is neither it adds a failed test per block.
- **Blocks of 32 elements** in the flat and rising loops. The loops are at the copy's speed in the second-level cache already; it could only show inside the first-level cache, which the judge does not time.
- **Stores past the caches below 24 MB,** for the reason `cumsum_i64` gives: the caller gets an output that is in no cache.
- **The general path in the vector registers.** By count (above the first attempt) it is slower than the integer one.
