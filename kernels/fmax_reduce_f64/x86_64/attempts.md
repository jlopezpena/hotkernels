# fmax_reduce_f64 attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, pinned to one performance core by the judge. The runs are `--unlocked`: `spec.py` and `naive.c` are new in the same pull request and not on `main` yet. Neither was changed after the first line of assembly was written. Other judges were running on other cores of the machine during these runs, so the sizes from 1,048,576 up are noisy.

**What NumPy does, found by experiment before `naive.c` was written** (NumPy 2.5.5 on this machine, whose loop for this reduction is the AVX2 one, four float64 to a vector). For every length from 1 to 44, every pair of positions was given `+0.0` and `-0.0` among elements of -1 (and among NaNs), which orders the positions by whose zero wins; an array of NaNs with a different payload in every element shows which one is returned. Neither depends on where the array starts in memory (eight offsets), and `np.fmax.reduceat` gives a run the bits `np.fmax.reduce` gives the same elements.

- The order of the positions is that of a loop with four running maxima, one per lane, all starting at `x[0]`, taking `x[1..4]`, `x[5..8]` and so on, then combining the lanes, then taking the `(n - 1) mod 4` leftover elements one at a time. At 14 elements the zero that wins is the first present of 12, 8, 4, 0, 10, 6, 2, 11, 7, 3, 9, 5, 1, 13: lane 3 from its end back to `x[0]`, then lanes 1, 2 and 0 from their ends, then the leftover element. Up to four elements it is the first zero of the array.
- In a lane, and between lanes, the later of two equal values is kept (the lanes are combined 0 with 2, 1 with 3, then those two). Among the leftover elements the earlier is kept: they go through the C library's `fmax` with the element as its first argument.
- An array of NaNs gives `x[0]`, sign and payload, when `(n - 1) mod 4` is 0, and its last element otherwise.
- A model of that loop in Python agreed with NumPy on 5,820 arrays of 1 to 100,003 elements holding zeros of both signs, quiet NaNs of four payloads and both signs, minus infinity and negative values; `naive.c`, which is the same loop, on 60,000 arrays of 1 to 20,040 elements outside the judge, at eight placements, with infinities, denormals and ordinary values as well.

This is the answer of NumPy's AVX2 loop. A NumPy that runs its AVX-512 loop (eight to a vector) or its NEON loop will order the positions differently; nothing here was run on such a machine.

What the compiler does with `naive.c` (`clang -O3`): it turns the four lanes into vector code, 0.29 to 0.45 ns per element, and the leftover loop into branches on the data, which random data mispredicts: called on runs of 2, 3 and 4 elements it takes 5, 7.5 and 10.7 ns, on runs of 5 it takes 2.2 ns.

Figures "per call on runs of L" are from a driver outside the judge: 8,192 elements in the first-level cache, cut into consecutive runs of L, the kernel called once per run as a binding of `np.fmax.reduceat` would call it, the best of 400 passes. The judge's smallest timed size is 16.

## 1. Baseline: NumPy's loop step for step, NaN to minus infinity, a chain of one `vmaxpd`

- **Idea:** `vmaxpd a, b` is "a if a > b, otherwise b", so with the running value as a it keeps the later of two equal values, which is the rule of NumPy's lanes; `vmaxsd v, r` keeps the earlier, the rule of the leftover elements. Neither ignores a NaN, so every element first goes through a `vmaxpd` against minus infinity. The result is then NumPy's unless every element is a NaN; a result of minus infinity is looked at again, one element at a time, and an array of NaNs returns `x[0]` or its last element. One element returns at once.
- **check:** pass. Outside the judge: 60,000 arrays against NumPy, no difference.
- **compare:** n/a (first version). `run`: 2.38x [2.38 – 2.39] over naive C on `random`, 2.37x [2.36 – 2.37] on `some_nan`; 2.26x and 2.54x over NumPy.
- **Kept:** yes
- **Notes:** 2.9 ns at 16 (1.80x over the naive C), 8.3 ns at 64, 0.18 ns per element from 4,096 to 1,048,576, which is the chain: one cycle per element. NumPy takes 0.096 ns per element at 16,384 to 262,144, so this is at 0.5x of it there. Per call on runs of 2, 3, 4, 8, 16, 32: 1.1, 1.2, 1.5, 1.8, 2.3, 3.2 ns (naive C: 5.2, 7.5, 10.7, 3.9, 5.6, 9.7).

## 2. Blocks of four vectors, combined with one another before they meet the running maximum

- **Idea:** the chain of attempt 1 is four cycles per vector. Combining is associative as long as the earlier value is the a of each `vmaxpd` and the later one the b (the later of two equal values survives either way), which is how NumPy's own loop goes through eight vectors at a time. Four cleaned vectors are combined in pairs, then the pairs, then the result with the running maximum: one `vmaxpd` on the chain per 16 elements. Whole blocks first, then single vectors, then the leftover elements.
- **check:** pass. Outside the judge: 60,000 arrays against NumPy, no difference.
- **compare:** significantly faster, 1.974x [1.968 – 1.980]
- **Kept:** yes
- **Notes:** both distributions alike: 1.91x at 64, 2.0x at 256, 2.8x at 1,024, 3.66x at 4,096, 2.9x at 16,384 and 65,536, 2.4x at 262,144, 1.9x at 1,048,576, 1.12x at 4,194,304, nothing at 16 and at 16,777,216. 0.049 ns per element at 4,096: eight `vmaxpd` per block on two ports, four cycles per 16 elements, which is also the chain. Per call on runs of 8, 16, 32, 64: 1.8, 2.4, 2.9, 4.9 ns.
