# cumfmax_f64 attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, pinned to one performance core by the judge. The runs are `--unlocked`: `spec.py` and `naive.c` are new in the same pull request and not on `main` yet. Neither was changed after the first line of assembly was written. Other judges were running on other cores of the machine during these runs, so the sizes from 1,048,576 up are noisy.

**What NumPy does, found by experiment before `naive.c` was written** (NumPy 2.5.5, which calls the C library's `fmax` for each pair, the running value first):

- A quiet NaN element changes nothing. While every element so far is NaN the output is `x[0]`, sign and payload: of two quiet NaNs `fmax` returns the first.
- Of two equal values the second is returned, so the later of `+0.0` and `-0.0` is the running value.
- An array of two elements is the exception: NumPy takes another path for a single pair and calls `fmax(x[1], x[0])`. `out[1]` is then the first of two zeros and the second of two NaNs.
- A signaling NaN is not ignored: the C library's `fmax` returns a quiet NaN for one, and the running maximum so far is lost (`[1.0, sNaN, 0.5]` gives `[1.0, NaN, 0.5]`). The spec leaves signaling NaNs out of the kernel's domain; the judge's NaN is the quiet one.

`naive.c` writes the comparisons out. `fmax()` is not declared where the package builds the C kernels (the build stops on it), and the compiler's own expansion of it would keep the earlier of two zeros. Outside the judge the naive C and the kernel were each compared with `np.fmax.accumulate`, bit for bit, on 12,216 arrays of 0 to 20,000 elements holding quiet NaNs of four payloads and both signs, infinities, both zeros, the smallest denormals and ordinary values, at eight placements: no difference.

What the compiler does with `naive.c` (`clang -O3 -march=native`, clang 21.1.8): a chain through the running value without a branch, 2.0 ns per element at every size and on every distribution, eleven cycles. NumPy takes 1.8 to 2.3 ns per element.

## 1. Baseline: NaN to minus infinity, the running maximum inside a vector, a chain of one `vmaxpd`

- **Idea:** `vmaxpd a, b` is "a if a > b, otherwise b": with the older value as a it keeps the later of two equal values, which is NumPy's rule for zeros, but it returns a NaN in b. So each element is first put through `vmaxpd` against minus infinity, which turns NaN into minus infinity and changes nothing else; once the running value M is a number that changes no output. The elements before the first number are done one at a time, and so is an array of two. Then four elements per vector: each lane against the lane before it (`vpermpd`), against the lane two before (`vpermpd`), the result against M in every lane for the output, and M against the vector's last lane, broadcast, for the next M: a chain of one `vmaxpd`, four cycles, per four elements.
- **check:** pass
- **compare:** n/a (first version). `run`: 6.99x [6.95 – 7.02] over naive C on `random`, 6.27x [6.25 – 6.29] on `rising`, 5.90x [5.87 – 5.93] on `some_nan`; 10.3x, 9.3x and 10.9x over NumPy.
- **Kept:** yes
- **Notes:** 0.19 ns per element from 256 to 65,536 on every distribution, one cycle: the chain. 10.4x over the naive C there, 4.5x at 16, 7.1x at 262,144, 3.3x and 2.4x to 2.5x at the two largest sizes. The second table of `run`, output 16 bytes behind the input: 0.88x to 0.90x of the first from 4,096 to 65,536. A vector is stored before the next is loaded.

## 2. Blocks of eight, the next block loaded before this one is stored

- **Idea:** the chain of attempt 1 is four cycles per vector. In a block of two vectors the second takes the first's last lane, broadcast, before either meets M, so the chain is one `vmaxpd` per eight elements. And the two vectors of the next block are loaded before the two of this one are stored, for the close placement.
- **check:** pass
- **compare:** significantly faster, 1.102x [1.096 – 1.108]
- **Kept:** yes
- **Notes:** on all three distributions alike: 1.20x at 1,024, 1.25x to 1.28x at 4,096, 1.23x to 1.31x at 16,384 and 65,536, 1.05x at 64, nothing from 262,144 up (0.95x and 0.96x at 1,048,576 on two of the three distributions, 1.00x on the third, with other judges running). 0.153 ns per element at 16,384, 0.8 of a cycle. By count the block has six shuffles (`vpermpd`) on the one port that takes them, six cycles per eight elements, and ten `vmaxpd` on two ports, five cycles.

## 3. Prefetch the input and the output six cache lines ahead

- **Idea:** found in a micro-benchmark outside the judge. Attempt 2 takes 0.142 ns per element at 1,024 elements, inside the first-level cache, and 0.167 at 16,384 and 65,536, where a plain copy takes 0.115: the cache is not what holds it. A block is some 30 operations, so the processor's window of instructions holds about a kilobyte of data and does not reach the cache misses ahead of it, as it does in a loop of a few instructions. One `prefetcht0` per block for each of the two buffers, 384 bytes ahead: 0.140 at 16,384 and 65,536 in the micro-benchmark, and the same at 768 bytes; 0.152 at 192. For the output alone: nothing. A prefetch reads nothing and cannot fault; near the end of the arrays its address is past them.
- **check:** pass
- **compare:** significantly faster, 1.033x [1.031 – 1.036]
- **Kept:** yes
- **Notes:** 1.04x to 1.08x from 64 to 65,536 on all three distributions, 1.02x at 262,144, 1.00x to 1.03x above; 16 unchanged. Less than the micro-benchmark's 1.2x: the judge's attempt 2 was at 0.153 ns per element at 16,384, not 0.167, and is now at 0.146. Tried in the same micro-benchmark and not kept, each a way to take shuffles off the one port that has them by sending the value through memory instead (a store below the stack pointer and a broadcast load of it): the two broadcasts of a block, 0.132 ns per element at 1,024 but 0.277 at 16,384; the shift by two lanes, 0.172 and 0.389. Beyond the first-level cache the stores of the output wait for their cache lines and the small stores queue behind them.

## 4. Bring the output to a multiple of 32 bytes first

- **Idea:** once M is a number, up to three elements one at a time until `out` is a multiple of 32, when at least 11 elements are left, so that no store of a block straddles two cache lines. An `out` that is not a multiple of 8 is left as it is.
- **check:** pass
- **compare:** significantly faster, 1.014x [1.008 – 1.019]
- **Kept:** yes
- **Notes:** 1.09x to 1.13x at 64 on all three distributions, 1.00x to 1.03x from 256 to 1,048,576, 0.97x to 1.00x at the two largest sizes, none of those significant on `rising` and `some_nan`. The block is held by its arithmetic, so a store that straddles costs little; the verdict is the 64-element row. The stores past the caches need the alignment in any case.

## 5. Stores past the caches from 24 MB of output

- **Idea:** when 3,145,728 elements or more are left as M becomes a number, the blocks store with `vmovntpd`, which does not read the line it is about to overwrite, followed by one `sfence`. It needs the output at a multiple of 32, which attempt 4 arranges; an output that could not be brought there keeps the ordinary stores. The output is no longer prefetched on that path. The threshold is this processor's third-level cache, as in `cumsum_i64`: below it an output written with ordinary stores is left in a cache for the caller.
- **check:** pass. The judge's correctness cases do not reach this path, so it was tested outside the judge against the naive C: 173 sizes, every one from 0 to 150 and 16 from 3,145,727 to 4,194,304 around the threshold and the head lengths; ten kinds of data (random, rising with ties, constant, few values, 1 0 2 1 3 2, two walks, falling, runs of random length that are flat, rising or noisy in turn, and the same with NaNs of both signs and several payloads, infinities and both zeros scattered through them, alone and in runs, with and without a run of NaN at the start); four placements of `x` and seven of `out`, two of them not multiples of 8; 16 guard bytes before and after `out`, and `x` compared with what it held before: 133,784 runs, no failure. The NaNs there include a signaling one, which the kernel ignores as the naive C does and NumPy does not.
- **compare:** significantly faster, 1.030x [1.029 – 1.031]; a second run, since the verdict rests on the two largest sizes: 1.027x [1.026 – 1.028].
- **Kept:** yes
- **Notes:** 1.23x to 1.25x at 4,194,304 (2.28 ms to 1.82 ms) and at 16,777,216 on all three distributions in both runs, nothing elsewhere.

## 6. The first shuffle inside the 128-bit halves, as an integer shuffle

- **Idea:** inside the first-level cache the block takes six cycles in the micro-benchmark, which is its six `vpermpd` on the one port that takes a shuffle across the halves of a register. The running maximum inside a vector can pair the lanes first (x0 x0 x2 x2 against the vector, a shuffle that stays inside each half) and then put the lower pair's maximum against the upper pair (a0 a0 a1 a1, across the halves). `vunpcklpd` and `vmovddup` for the first shuffle changed nothing in the micro-benchmark (145 ns at 1,024 elements, as before); `vpshufd`, the integer shuffle, has a second port: 139 ns.
- **check:** pass, and the comparison with NumPy outside the judge again: 12,216 arrays, no difference.
- **compare:** significantly faster, 1.009x [1.008 – 1.010]
- **Kept:** yes
- **Notes:** 1.02x to 1.03x from 256 to 4,096, 1.01x to 1.02x at 16,384 and 65,536, nothing at 16 and 64 or from 262,144 up. Tried in the micro-benchmark and not taken further: the second vector's first step from a load one element back, cleaned of NaN, in place of its shuffle: 0.138 ns per element at 1,024 and 16,384 against this attempt's 0.135 and 0.133 (attempt 5: 0.142), and both together 0.140. It trades a shuffle for a `vmaxpd`, and the ten of a block already fill their two ports for five cycles.

## Where it stands

Final `judge run` (unlocked): 8.73x [8.69 – 8.76] over naive C on `random`, 8.40x [8.37 – 8.42] on `rising`, 8.18x [8.16 – 8.21] on `some_nan`, 8.44x [8.42 – 8.45] over the three; 12.8x, 12.3x and 14.9x over NumPy.

`random`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 3.9 ns | 16.6 ns | 4.24x [4.24 – 4.25] | 294.3 ns | 75.2x [75.14 – 75.26] |
| 64 | 10.8 ns | 111.0 ns | 10.3x [10.31 – 10.32] | 381.2 ns | 35.4x [35.38 – 35.44] |
| 256 | 38.9 ns | 502.7 ns | 12.9x [12.89 – 13.00] | 745.7 ns | 19.2x [19.12 – 19.28] |
| 1,024 | 147.6 ns | 2.1 us | 14.1x [14.05 – 14.08] | 2.1 us | 14.4x [14.39 – 14.43] |
| 4,096 | 592.4 ns | 8.4 us | 14.1x [14.10 – 14.12] | 7.5 us | 12.7x [12.70 – 12.72] |
| 16,384 | 2.3 us | 33.5 us | 14.3x [14.24 – 14.26] | 31.3 us | 13.3x [13.30 – 13.35] |
| 65,536 | 9.3 us | 134.1 us | 14.3x [14.32 – 14.39] | 116.2 us | 12.4x [12.41 – 12.48] |
| 262,144 | 61.4 us | 529.8 us | 8.63x [8.52 – 8.69] | 490.6 us | 8.00x [7.90 – 8.04] |
| 1,048,576 | 358.5 us | 2.15 ms | 6.01x [5.75 – 6.24] | 1.99 ms | 5.54x [5.32 – 5.74] |
| 4,194,304 | 1.97 ms | 8.62 ms | 4.36x [4.31 – 4.41] | 9.50 ms | 4.81x [4.74 – 4.87] |
| 16,777,216 | 7.93 ms | 34.45 ms | 4.34x [4.25 – 4.37] | 37.71 ms | 4.75x [4.66 – 4.77] |
| overall |  |  | 8.73x [8.69 – 8.76] |  | 12.8x [12.79 – 12.89] |

`rising`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 3.7 ns | 16.6 ns | 4.55x [4.55 – 4.55] | 295.2 ns | 80.8x [80.79 – 80.89] |
| 64 | 10.8 ns | 110.8 ns | 10.3x [10.25 – 10.25] | 380.6 ns | 35.2x [35.20 – 35.25] |
| 256 | 39.2 ns | 503.0 ns | 12.8x [12.77 – 12.96] | 745.7 ns | 19.0x [18.93 – 19.21] |
| 1,024 | 147.2 ns | 2.1 us | 14.1x [14.09 – 14.11] | 2.1 us | 14.3x [14.31 – 14.34] |
| 4,096 | 601.0 ns | 8.4 us | 13.9x [13.89 – 13.92] | 7.6 us | 12.6x [12.55 – 12.58] |
| 16,384 | 2.4 us | 33.5 us | 14.2x [14.15 – 14.17] | 29.0 us | 12.3x [12.27 – 12.29] |
| 65,536 | 9.8 us | 134.4 us | 13.7x [13.71 – 13.77] | 125.0 us | 12.8x [12.75 – 12.81] |
| 262,144 | 79.4 us | 528.2 us | 6.65x [6.64 – 6.67] | 489.4 us | 6.16x [6.15 – 6.17] |
| 1,048,576 | 425.0 us | 2.16 ms | 5.08x [4.92 – 5.24] | 1.99 ms | 4.69x [4.55 – 4.83] |
| 4,194,304 | 1.96 ms | 8.57 ms | 4.38x [4.33 – 4.43] | 9.41 ms | 4.80x [4.74 – 4.86] |
| 16,777,216 | 7.95 ms | 34.48 ms | 4.34x [4.27 – 4.36] | 37.58 ms | 4.73x [4.65 – 4.76] |
| overall |  |  | 8.40x [8.37 – 8.42] |  | 12.3x [12.29 – 12.37] |

`some_nan`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 3.6 ns | 15.5 ns | 4.28x [4.28 – 4.28] | 303.7 ns | 83.6x [83.52 – 83.63] |
| 64 | 10.8 ns | 105.0 ns | 9.70x [9.70 – 9.72] | 425.5 ns | 39.3x [39.28 – 39.40] |
| 256 | 39.3 ns | 476.8 ns | 12.1x [12.09 – 12.19] | 838.2 ns | 21.3x [21.24 – 21.44] |
| 1,024 | 147.5 ns | 2.0 us | 13.3x [13.32 – 13.35] | 2.4 us | 16.3x [16.31 – 16.36] |
| 4,096 | 601.1 ns | 7.8 us | 13.0x [12.96 – 12.97] | 8.6 us | 14.3x [14.27 – 14.32] |
| 16,384 | 2.4 us | 31.7 us | 13.4x [13.40 – 13.41] | 34.7 us | 14.7x [14.65 – 14.67] |
| 65,536 | 9.6 us | 126.6 us | 13.1x [13.13 – 13.16] | 150.6 us | 15.6x [15.61 – 15.67] |
| 262,144 | 62.2 us | 501.5 us | 8.06x [8.03 – 8.10] | 608.9 us | 9.79x [9.75 – 9.82] |
| 1,048,576 | 380.1 us | 2.02 ms | 5.31x [5.18 – 5.45] | 2.47 ms | 6.49x [6.32 – 6.66] |
| 4,194,304 | 2.12 ms | 8.67 ms | 4.09x [4.02 – 4.16] | 12.14 ms | 5.73x [5.61 – 5.82] |
| 16,777,216 | 7.80 ms | 31.93 ms | 4.09x [4.04 – 4.13] | 45.41 ms | 5.82x [5.74 – 5.85] |
| overall |  |  | 8.18x [8.16 – 8.21] |  | 14.9x [14.84 – 14.93] |

With the output buffer 16 bytes behind its input ("apart" is the same kernel with the output clear of the input), `random`:

| n | kernel | apart | speedup [95%] | naive C | speedup [95%] |
|---|---|---|---|---|---|
| 4,096 | 585.5 ns | 600.6 ns | 1.03x [1.02 – 1.03] | 8.4 us | 14.3x [14.22 – 14.30] |
| 16,384 | 2.4 us | 2.3 us | 1.00x [0.99 – 1.00] | 33.5 us | 14.2x [14.17 – 14.22] |
| 65,536 | 9.4 us | 9.8 us | 1.03x [1.03 – 1.03] | 134.4 us | 14.2x [14.21 – 14.24] |
| 262,144 | 62.2 us | 78.7 us | 1.27x [1.26 – 1.27] | 534.3 us | 8.59x [8.55 – 8.63] |
| 1,048,576 | 433.2 us | 584.5 us | 1.35x [1.33 – 1.37] | 2.66 ms | 6.13x [6.08 – 6.21] |
| 4,194,304 | 1.86 ms | 1.96 ms | 1.06x [1.04 – 1.07] | 8.63 ms | 4.63x [4.58 – 4.68] |
| 16,777,216 | 7.69 ms | 7.93 ms | 1.03x [1.01 – 1.05] | 34.35 ms | 4.47x [4.38 – 4.49] |
| overall |  |  | 1.10x [1.10 – 1.11] |  | 8.48x [8.44 – 8.50] |

No stall since attempt 2: a block's loads come before the stores of the block before it. Attempt 1 was at 0.88x to 0.90x there.

**The same speed on every kind of data.** The kernel has no branch that depends on an element once the running value is a number: 0.14 to 0.15 ns per element from 1,024 to 65,536 on random, rising and NaN-laden data alike. Before the first number it takes the elements one at a time with a branch each, so an array that is all NaN is done at that loop's speed; the judge's `nan_all` cases check it and nothing times it.

What the kernel requires of its caller: `out` does not overlap `x`, and no element of `x` is a signaling NaN. With one, the kernel ignores it as it ignores any NaN (and returns it unchanged if it is `x[0]`), where NumPy's result turns into a quiet NaN at that element and forgets the maximum so far. The judge cannot see the difference: its NaN is the quiet one. Matching NumPy there would take a test of every element on every path; `spec.py` says the caller checks, which is a person's decision to confirm.

What holds it at each size:

- **16:** the entry, the head and the tail: 4.2x to 4.6x.
- **64 to 65,536:** the arithmetic of a block: ten `vmaxpd` on two ports and six shuffles, four of them on one port. 0.135 ns per element in the micro-benchmark inside the first-level cache and 0.133 to 0.140 at 16,384; 0.14 to 0.15 in the judge. A plain copy takes 0.115 beyond the first-level cache, so the cache is not the limit here as it is for `cummax_i64`.
- **262,144 and 1,048,576:** the third-level cache: 5x to 8.6x.
- **4,194,304 and 16,777,216:** memory, with a third less traffic than the naive C: 4.1x to 4.4x.

Not tried:

- **A flat loop and a rising loop, as in `cummax_i64`:** a block in which no element is above M is M eight times, at one compare per vector. For random data beyond the first-level cache that is the copy's 0.115 ns per element against 0.14 to 0.15, and inside it more. It needs a second test for a zero of the other sign (NumPy's running value changes sign there, and `x > M` does not see it), and it gives the kernel branches that depend on the data, which it does not have now.
- **Blocks of 16** with the next M taken from the last output by a shuffle in place of a `vmaxpd`. By count it saves one `vmaxpd` in 20 and no shuffle.
- **The first elements before a number in the vector registers.** An array that begins with many NaNs is done one element at a time until the first number.
- **Stores past the caches below 24 MB,** for the reason `cumsum_i64` gives: the caller gets an output that is in no cache.
- **`prefetchw` for the output.** It is not among the x86-64-v3 instructions.
