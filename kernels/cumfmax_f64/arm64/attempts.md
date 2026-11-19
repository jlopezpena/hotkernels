# cumfmax_f64 (arm64) attempts

Apple M1 Pro, macOS, native.

**Every judge command in this run took `--unlocked`,** a departure from AGENTS.md decided by the maintainer for this run. The kernel's `spec.py` on this branch differs from `main`: until pull request #177 the locked reference failed the judge on ARM64 ([issue 162](https://github.com/sofeikov/hotkernels/issues/162): NumPy's ARM64 build treats zeros of both signs and NaNs of several bit patterns differently from the x86-64 build the spec described). #177 narrows the spec's domain to what the binding passes (no `-0.0` in the input, NaNs of one bit pattern, no signalling NaN) and was not merged when the run was made. `spec.py`, `naive.c` and `judge/` were not edited.

## 1. Baseline: one scalar `fmaxnm` per element

- **Idea:** NumPy's ARM64 loop itself, `m = fmaxnm(m, x[i])` from `x[0]`, in the simplest form: `fmaxnm` ignores a quiet NaN operand, and of two NaNs returns one with the same bits, so the all-NaN prefix is `x[0]`'s NaN.
- **check:** pass (20,368 cases)
- **compare:** n/a (first version). `run`: 3.96x [3.96 – 3.96] over naive C on all three distributions, 8.01x over NumPy. 0.64 ns per element at every size from 1,024 up: the chain's two cycles per element. The naive C, whose compares and selects make a longer chain, takes 2.6 ns.
- **Kept:** yes (baseline)

## 2. Blocks of 16: a prefix maximum in the vector registers, M brought in last

- **Idea:** out[0] = x[0] = M, then blocks of 16 from x[1], so the element before each block exists. Each vector against the input one element back (`ldur`), then Sklansky steps over the eight vectors as in `cummax_i64`, then every vector against M; the next M is M against the block's last lane, so the chain from block to block is one `fmaxnm`. `fmaxnm` is a single instruction where the integer kernel needed `cmgt` and `bit`. Including the element before the block is harmless: it is already in M. Tail: one vector (two elements) at a time against the element before, then one scalar.
- **check:** pass
- **compare:** significantly faster, 2.624x [2.615 – 2.633]; 3.0x to 3.2x from 1,024 to 262,144 on every distribution, 2.6x to 2.8x from 4,194,304 up, 1.5x at 16.
- **Kept:** yes
- **Notes:** 0.20 ns per element at 4,096: about ten cycles per block of 16, for 33 vector operations, near the four floating-point pipes' limit. The next gain has to come from doing fewer of them.
