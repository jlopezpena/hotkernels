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

## 3. A flat loop: a block that does not change M is M sixteen times

- **Idea:** as in `cummax_i64`, but only the flat way: the block's maximum (a tree of seven `fmaxnm`, an `fmaxnmp`, one `fmaxnm` against M) compared with M; equal, the block is M sixteen times. The blocks start in a general loop and stay there while each block changes M (so the all-NaN prefix, where the compare is unordered, stays general too); after one that does not, flat; a block that fails the flat test goes the general way, and the next is tried flat again. Once M is a number it never becomes NaN again.
- **check:** pass
- **compare:** significantly faster, 1.099x [1.083 – 1.115]: `random` 1.16x, `some_nan` 1.14x (1.1x to 1.4x from 256 up), `rising` 1.01x.
- **Kept:** yes
- **Notes:** outside the judge (a C loop timing the kernel on 4,096 elements in the first-level cache) a flat block takes 7.6 cycles at 3.228 GHz, a general one 10.2. Without its stores a flat block takes 3.2 cycles, four `stp` of M alone 4.1, and a plain copy of 128 bytes (four `ldp`, four `stp`) 5.1 to 5.6: this core's first-level cache moves about 48 bytes per cycle read and written together, and the stores are the larger part of the flat block's cost.

## 4. Flat test: `fcmgt` against M and `umaxv` in place of `fmaxnmp`, `fmaxnm` and `fcmp`

- **Idea:** the tree's two lanes compared with M as a vector, the mask reduced with `umaxv` and moved to an integer register; a NaN lane compares false, which is right once M is a number, as it is in the flat loop. A shorter path to the branch.
- **check:** pass
- **compare:** no significant difference, 1.000x [0.999 – 1.002]
- **Kept:** no

## 5. A rising loop: a block that never falls and starts on M is a copy

- **Idea:** as in `cummax_i64`. After a general block that does not leave M as it was, if the block's last element is M, try the next as rising: every element `fcmge` the one before it (half the shifted vectors from `ldur`, half from `ext`, as `cummax_i64` found best), the masks `and`ed and `uminv`. A NaN compares false and fails the test, which is what makes `fcmge` and not `fcmgt`-the-other-way the test here. The block is then the input, stored as it is. A block that fails goes the general way with its vectors already loaded; M is reloaded from the element before it.
- **check:** pass
- **compare:** significantly faster, 1.049x [1.041 – 1.057]: `rising` 1.16x (1.23x at 256, 1.10x to 1.26x from 4,096 up), `random` and `some_nan` 1.00x, apart from 0.99x and 0.97x at 64.
- **Kept:** yes

## 6. Blocks from x[0], not from x[1]

- **Idea:** the blocks started at x[1] so that the element before each existed, which left 15 elements to the tail at every size the judge times and put every block 8 bytes off the buffers' alignment. Start them at x[0], with x[0] standing in for the element before the first block (it is M, so it changes nothing); below 16 elements, x[0] and then the tail from x[1] as before.
- **check:** pass
- **compare:** significantly faster, 1.174x [1.170 – 1.178]: 1.16x at 16, 1.04x to 1.07x at 64, 1.22x to 1.33x at 256, 1.34x to 1.73x at 1,024 and 4,096, 1.04x to 1.11x from 65,536 up.
- **Kept:** yes
- **Notes:** the gain at 1,024 and 4,096 is far more than the tail's: it is the alignment. A flat block now takes 0.106 ns per element at 4,096, 5.4 cycles per block, the speed of a copy in the first-level cache; with each `stp` of 32 bytes crossing a 64-byte line every other time it was 7.6. So an `out` that is not a multiple of 64 bytes (or at least 32) costs the kernel up to 1.7x in the first-level cache; the judge's timed buffers are aligned.
- **Later, outside the judge:** with `x` at a page and `out` at a page plus 0, 16, 32 and 8 bytes, a random 4,096 takes 0.094, 0.088, 0.099 and 0.155 ns per element. Only an `out` that is not a multiple of 16 bytes pays: every other `stp` of 32 bytes then crosses a line.

## 7. `stnp` in the flat and rising loops

- **Idea:** as in `cummax_i64`: the non-temporal pair store for the two loops that store whole blocks without a prefix maximum.
- **check:** pass
- **compare:** significantly faster, 1.020x [1.017 – 1.024]; a second run, since nearly every size read 1.00x: 1.021x [1.017 – 1.025]. The gain is all at 16,384 elements, 1.36x on `random` and `some_nan` and 1.07x on `rising` in both runs; 0.98x to 0.99x at 1,024 and 4,096 on `random` and `some_nan`; 1.00x elsewhere.
- **Kept:** yes
- **Notes:** at 16,384 elements `x` is 128 KB, the size of this core's first-level data cache, and `x` and `out` together do not fit. With `stnp` the output seemingly does not displace the input, and a repeated call finds `x` where the last one left it: 0.11 ns per element there, against 0.15 before.

## 8. After a general block: rising if it ends on M, flat otherwise

- **Idea:** as attempt 13 of `cummax_i64`. The general loop went on while each block changed M; random data changes M often in its first blocks, and the branch that kept it general went either way. Now a general block is followed by the rising test if it ends on M, by the flat test otherwise, and there is no general loop: a general block is only ever the first one or one that failed a test. While M is NaN the flat test fails (its compare is unordered), so an all-NaN prefix still goes the general way, at the cost of a failed test per block. The copy of M before each general block is gone with the loop.
- **check:** pass
- **compare:** significantly faster, 1.015x [1.014 – 1.016]: 1.12x to 1.14x at 64 on `random` and `some_nan`, 1.05x at 64 on `rising`, 1.04x to 1.05x at 256, 1.00x from 4,096 up.
- **Kept:** yes

## 9. The first block without M

- **Idea:** in the first block x[0] stands in for the element before it, so after the Sklansky steps every lane's run already starts at x[0], which is M: the eight `fmaxnm` against M and the one that makes the next M are skipped, and M is the block's last lane. One step less of latency, which is what a call of 16 elements is.
- **check:** pass
- **compare:** significantly faster, 1.027x [1.026 – 1.028]: 1.20x at 16, 1.04x to 1.11x at 64, 1.02x to 1.06x at 256.
- **Kept:** yes

## 10. Rising test: every shifted vector from `ldur`

- **Idea:** the four shifted vectors made with `ext` loaded instead, taking four operations off the floating-point pipes, which the test's compares and `and`s keep busy.
- **check:** pass
- **compare:** significantly slower, 0.987x [0.977 – 0.998]: `rising` 0.96x (0.87x at 64), the others 1.00x. As on `cummax_i64` (its attempt 17), half and half is better.
- **Kept:** no

## 11. Prefetch the input 1,024 bytes ahead in the flat and rising loops

- **Idea:** one `prfm pldl1keep` per block, eight blocks ahead, for the sizes beyond the caches.
- **check:** pass
- **compare:** significantly slower, 0.986x [0.984 – 0.988]: 1.00x to 1.01x from 1,048,576 up, 0.92x to 0.96x from 16,384 to 262,144 on `some_nan`. This core's own prefetchers already follow a forward stream.
- **Kept:** no

## Where it stands

`uv run judge run cumfmax_f64 --unlocked` at the end of the run, with another kernel run and CI jobs on the machine: 14.0x [13.83 – 14.13] over the naive C and 28.4x [27.94 – 28.57] over NumPy; `random` 14.3x, `rising` 14.1x, `some_nan` 13.7x. 5.3x at 16 elements, 12x to 14x at 64, 20x to 25x from 256 to 16,384 (0.10 to 0.12 ns per element, the speed of a copy in the first-level cache), 14x to 15x at 65,536 and 262,144, 9x to 14x from 1,048,576 up, where the figures are noisy. The naive C takes 2.5 ns per element on this machine at every size. With the output 16 bytes behind its input the kernel is no slower: 1.24x [1.23 – 1.24] over the same placement apart.

**Outside the domain** (a script against `np.fmax.accumulate` on this machine, NumPy 2.5.5, arrays of 1 to 80 elements; no change made for it):

- `-0.0`: 1,035 of 3,000 arrays of zeros of both signs and `-1.0` differ. The flat test takes `fmaxnm(T, M) == M` as "nothing changes", and `+0.0 == -0.0`; the rising test takes `-0.0 >= +0.0` as rising and copies the input. Both can leave a `-0.0` where NumPy's `fmaxnm` chain has `+0.0`.
- NaNs of several bit patterns: 371 of 3,000 differ, and 1,796 of 2,000 arrays that start with a run of NaNs of mixed patterns. In the all-NaN prefix the kernel gives the later NaN where NumPy keeps `x[0]`'s: `fmaxnm` of two NaNs returns its first operand, and the prefix steps put the later element first. Swapping the operands of those steps would likely match NumPy here at no cost; it was not tried, being outside the spec.
- One NaN pattern, `0xfff8...` (negative): no difference in 3,000 arrays, so the default-NaN mode is off and `fmaxnm` passes the pattern through.
- A signalling NaN: 2,619 of 3,000 differ. NumPy's output becomes a quiet NaN and the maximum so far is forgotten; the kernel mostly ignores it (in a flat block or once quietened inside a prefix step) and may give a quiet NaN where a scalar step meets it directly.
