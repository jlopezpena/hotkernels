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
