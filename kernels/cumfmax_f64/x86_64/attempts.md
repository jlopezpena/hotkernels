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
