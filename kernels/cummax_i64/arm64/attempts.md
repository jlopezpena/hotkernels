# cummax_i64 attempts (arm64)

All timings on an Apple M1 Pro, macOS, native. Another kernel run was timing
in another worktree during these runs, so sizes from about 100,000 elements up
are noisier than on a quiet machine.

Reference point: the naive C (`clang -O3`) keeps the running maximum in one
register, a `cmp` and a `csel` per element on one chain: 0.65 ns per element,
two cycles, at every size from 1,024 up, on both distributions. NumPy takes
2.5 ns per element.

## 1. Simplest scalar loop

- **Idea:** one `ldr`, `cmp`, `csel`, `str` per element with post-indexed
  pointers; the starting point, to get `check` passing.
- **check:** pass
- **compare:** n/a (first version). `run`: 0.98x [0.98 – 0.98] of naive C
  overall, on both distributions.
- **Kept:** yes (baseline)
- **Notes:** 0.90x at 16, 1.00x from 4,096 up: both are held by the
  two-cycle `cmp`/`csel` chain, 0.65 ns per element.

## 2. Scalar groups of four with M off the chain inside the group

- **Idea:** the maxima of x0, x0..x1, x0..x2, x0..x3 of a group of four do
  not depend on M, so each output is one `cmp`/`csel` against M and the chain
  from group to group is one `cmp` and one `csel` (two cycles per four
  elements instead of per element). Two groups per iteration, one more group,
  then the scalar loop of attempt 1 for 0 to 3 elements.
- **check:** pass
- **compare:** significantly faster, 1.34x [1.34 – 1.34]
- **Kept:** yes
- **Notes:** 1.40x to 1.57x from 256 up on both distributions, 0.41 ns per
  element (1.3 cycles): no longer the chain but the 14 `cmp`/`csel` per four
  elements. 0.72x at 16 (10.1 ns against 7.3) and 1.11x at 64.

## 3. Blocks of 16 that are flat, rising or general

- **Idea:** the idea of the x86-64 kernel. With M the maximum so far, a block
  of 16 elements is *flat* if no element is above M (eight `cmgt` against M
  in both lanes, an `orr` tree, `umaxv`, `fmov`, `cbnz`; the block is M
  sixteen times, four `stp` of q registers), *rising* if no element is below
  the one before it and the element before the block is M (the same test
  against eight overlapping `ldur` one element back; the block is a copy),
  and otherwise done the general way of attempt 2, four groups of four. The
  flat and rising loops run until a block fails their test. After a general
  block the next is tried flat if M did not change, rising if the block's
  last element is M, and done the general way untested otherwise; after a
  block that failed the flat test the flat test is tried again at once. Fewer
  than 16 elements: groups of four, then single elements.
- **check:** pass
- **compare:** significantly faster, 2.33x [2.31 – 2.35]
- **Kept:** yes
- **Notes:** `random`: 1.11x at 16 (one general block either way), 1.24x at
  64, 2.4x to 3.8x from 256 up; the flat loop runs at 0.104 ns per element at
  4,096 (5.3 cycles per block, about what eight 16-byte stores cost on this
  core by `cumsum_i64`'s log). `rising`: 2.3x to 3.0x from 64 up; the rising
  loop runs at 0.135 ns per element at 4,096 (6.9 cycles per block). 2.0x to
  2.3x at 4,194,304 and 16,777,216, about 0.19 ns per element, which is the
  speed of a copy from memory.

## 4. Rising test: half the shifted vectors from `ext`

- **Idea:** the rising loop loads 12 vectors per block (eight for the block,
  eight shifted back one element, as four `ldp` and eight `ldur`). The
  shifted vector of the second register of each `ldp` pair is the `ext` of
  the pair, so build those four with `ext` and load only the other four.
- **check:** pass
- **compare:** significantly faster, 1.03x [1.03 – 1.04]
- **Kept:** yes
- **Notes:** `rising` 1.14x to 1.16x from 256 to 16,384 (0.118 ns per
  element at 4,096, 6.0 cycles per block against 6.9), 1.08x at 64, nothing
  from 65,536 up; `random` unchanged.
