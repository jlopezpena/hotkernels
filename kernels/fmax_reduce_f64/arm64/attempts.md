# fmax_reduce_f64 attempts (arm64)

All runs on an Apple M1 Pro (performance core), macOS, native. **Every judge command in this run takes `--unlocked`**, a departure from `AGENTS.md` decided by the maintainer: the kernel's `spec.py` on this branch differs from `main`. Until pull request #177 the locked reference failed the judge on ARM64 (issue 162: NumPy's ARM64 build treats zeros of both signs and NaNs of several bit patterns differently from the x86-64 build the spec described); #177 narrows the spec's domain to what the binding passes, and is not merged. `spec.py`, `naive.c` and `judge/` were not edited. Another kernel run and CI jobs shared the machine, so sizes from about 100,000 elements up are noisy.

**Where the order of the elements is relied on.** NumPy's ARM64 loop is the chain `m = fmaxnm(m, x[i])` from `x[0]` (`docs/fmax-references-design.md`). Within the spec's domain any order of `fmaxnm` gives the same bits: `fmaxnm` of `+0.0` and `-0.0` is `+0.0` in either order, and a zero maximum has no `-0.0` beside it anyway; an all-NaN run has one bit pattern, which every order returns (two quiet NaNs give the first operand, quieted, and `FPCR.DN` is clear on macOS); no NaN is signalling. Every attempt from 2 on reorders the elements, and reads some of them twice, on that ground.

## 1. Baseline: the scalar `fmaxnm` chain

- **Idea:** NumPy's own ARM64 rule, one `fmaxnm` per element from `x[0]`.
- **check:** pass
- **compare:** n/a (first version)
- **Kept:** yes
- **Notes:** starting point; the chain is one `fmaxnm` latency per element.

## 2. Vectors: overlapping first and last loads up to 32 elements, eight accumulators beyond

- **Idea:** the order does not matter within the domain (see above), so: up to 32 elements no loop, the first and the last 2, 4, 8 or 16 elements as vectors that overlap in the middle, folded as a tree of `fmaxnm` and one `fmaxnmp`; the class is found from the bits of n - 1. From 33 elements eight two-lane accumulators start as the first 16 elements, take 16 elements per step, and the last step is the last 16 elements, overlapping the one before it.
- **check:** pass
- **compare:** significantly faster, 6.93x [6.85 – 7.02]
- **Kept:** yes
- **Notes:** 1.3 ns at 16 (5.2x), 4.4 ns at 64, 0.054 ns per element at 4,096 and 16,384: 2.8 cycles per 16 elements, the load limit of three 16-byte loads per cycle on this core. 0.095 ns per element from 65,536 (second-level cache), 110 us at 1,048,576, where NumPy takes 119 us.
